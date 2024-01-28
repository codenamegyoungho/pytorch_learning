#%%
import torch 
import torchvision 
from torchinfo import summary 
import matplotlib.pyplot as plt 
from torch import nn 
from torchvision import transforms 

# Setup device agnostic code 
import os 
import zipfile 

from pathlib import Path 
import requests 

# Setup path to data folder 
device = "cuda" if torch.cuda.is_available() else "cpu"
data_path = Path("data/")
image_path = Path(data_path/ "pizza_steak_sushi")

# If the image folder doesn't exist, download it and prepare it...
if image_path.is_dir():
    print(f"{image_path} directory exists.")
else:
    print(f"Did not find {image_path} directory, creating one...")
    image_path.mkdir(parents=True, exist_ok=True)

    # Download pizza, steak, sushi data 
    with open(data_path / "pizza_steak_sushi.zip",'wb') as f:
        request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
        print("Downloading pizza, steak, sushi data")
        f.write(request.content)
    
    # Unzip pizza, steak, sushi data 
    with zipfile.ZipFile(data_path / 'pizza_steak_sushi.zip','r') as zip_ref:
        print('Unzipping pizza, steak, sushi data...')
        zip_ref.extractall(image_path)
    
    # Remove zip file
    os.remove(data_path / "pizza_steak_sushi.zip")

# %%
train_dir = image_path / "train"
test_dir = image_path / "test"

# %%
manual_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485,0.456,0.406],
                         std = [0.229,0.224,0.225])
])
#%%
from going_modular import data_setup
# Create training and testing DataLoaders as well as get a list of class names 
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir = train_dir,
                                                                               test_dir= test_dir,
                                                                               transform= manual_transforms,
                                                                               batch_size= 32,
                                                                               )
train_dataloader , test_dataloader, class_names
# %%
# Get a set of pretrained model weights 
weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT

# %%
auto_transforms = weights.transforms()
auto_transforms
# %%
# Create training and testing DataLoaders as well as get a list of class names 
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir = train_dir, 
                                                                               test_dir= test_dir, 
                                                                               transform= auto_transforms,
                                                                               batch_size=32)
train_dataloader, test_dataloader, class_names
# %%
# New : Setup the model with pretrained weights and send it to the target device (torchvision v0 . 13+)
weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
model = torchvision.models.efficientnet_b0(weights=weights).to(device)
#%%
model
# %%
# Print a summary using torchinfo (uncomment for actual output)
summary(model = model,
        input_size= (32,3,224,224),
        col_names = ["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=['var_names'])
# %%
# Freeze all base layers in the "features" section of the model ( the feature extractor) by setting requires_grad = False

for param in model.features.parameters():
    param.requires_grad = False 

# %%
# Set the manual seeds 
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Get the length of class names (one output unit for each class)
output_shape = len(class_names)

# Recreate the classifier layer and seed it to the target device
model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(p=0.2, inplace=True),
    torch.nn.Linear(in_features = 1280,
                    out_features= output_shape,
                    bias=True
                    )
).to(device)
# %%
# # Do a summary *after* freezing the features and changing the output classifier layer (uncomment for actual output)
summary(model,
        input_size=(32,3,224,224),
        verbose=0,
        col_names=["input_size","output_size","num_params", "trainable"],
        col_width=20,
        row_settings=['var_names'])

# %%
# Define loss and optimizer 
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr = 0.001)

# %%
from going_modular import engine
# Set the random seeds 
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Start the timer 
from timeit import default_timer as timer 
start_time = timer()

# Setup training and save the results 
results = engine.train(model=model,
                       train_dataloader=train_dataloader,
                       test_dataloader= test_dataloader,
                       optimizer= optimizer,
                       loss_fn = loss_fn,
                       epochs=5,
                       device=device)

# End the timer and print out how long it took
end_time = timer()
print(f"[INFO] Total training time : {end_time - start_time:.3f} seconds")

# %%
# Get the plot_loss_curves() function from helper_functions.py, download the file if we don't have it 
from helper_functions import plot_loss_curves
plot_loss_curves(results)
# %%
from typing import List, Tuple 
from PIL import Image 

# 1. Take in a trained model, class names, image path, image size, a transform and target device 

def pred_and_plot_image(model: torch.nn.Module,
                        image_path: str,
                        class_names: List[str],
                        image_size: Tuple[int, int] = (224,224),
                        transform: torchvision.transforms = None,
                        device: torch.device = device):
    
    # 2. Open image 
    img = Image.open(image_path)

    # 3. Create transformation for image (if one doesn't exist)
    if transform is not None:
        image_transform = transform 
    else:
        image_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485,0.456,0.406],std=[0.229,0.224,0.225]),
        ])
    
    ### Predict on image ###
    
    # 4. Make sure the model evaluation mode and inference mode
    model.to(device)
    model.eval()
    with torch.inference_mode():
        transformed_image = image_transform(img).unsqueeze(dim=0)
        target_image_pred = model(transformed_image.to(device))
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)
    
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    # 10. Plot image with predicted label and probability 
    plt.figure()
    plt.imshow(img)
    plt.title(f"Pred : {class_names[target_image_pred_label]} | Prob : {target_image_pred_probs.max():.3f}")
    plt.axis(False);

# %%
# Get a random list of image paths from test set 
import random 
num_images_to_plot = 3 
test_image_path_list = list(Path(test_dir).glob("*/*.jpg"))
test_image_path_sample = random.sample(population=test_image_path_list, k=num_images_to_plot)

# Make predictions on and plot the images 
for image_path in test_image_path_sample:
    pred_and_plot_image(model = model,
                        image_path = image_path,
                        class_names = class_names,
                        image_size = (224,224))
# %%
# Download custom image
import requests 

# Setup custom image path 
custom_image_path = Path(data_path / "04-pizza-dad.jpeg")

# Download the image if it doesn't already exist 
if not custom_image_path.is_file():
    with open(custom_image_path, "wb") as f:
        request = requests.get('https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/04-pizza-dad.jpeg')
        print(f"Downloading {custom_image_path}...")
        f.write(request.content)
else:
    print(f"{custom_image_path} already exists, Skipping download.")

# Predict on custom image 
pred_and_plot_image(model = model,
                    image_path= custom_image_path,
                    class_names=class_names,)
# %%
