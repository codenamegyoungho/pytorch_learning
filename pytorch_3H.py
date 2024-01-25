#%%
import torch 
from torch import nn 
import matplotlib.pyplot as plt 
import torchvision 
from torchvision import datasets 
from torchvision.transforms import ToTensor 


# %%
# Setup Training data 
train_data = datasets.FashionMNIST(
    root='data',
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=None 
)
test_data = datasets.FashionMNIST(
    root = 'data',
    train=False,
    transform=ToTensor(),
    download=True
)
# %%
class_names = train_data.classes 
# %%
torch.manual_seed(42)
fig = plt.figure(figsize=(9,9))
rows, cols = 4,4 
for i in range(1, rows * cols +1 ):
    random_idx = torch.randint(0,len(train_data),size=[1]).item()
    img, label = train_data[random_idx]
    fig.add_subplot(rows, cols, i)
    plt.imshow(img.squeeze(),cmap='gray')
    plt.title(class_names[label])
    plt.axis(False);
# %%
from torch.utils.data import DataLoader 
BATCH_SIZE = 32 
train_dataloader = DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    shuffle = True
)

test_dataloader = DataLoader(
    test_data,
    batch_size=BATCH_SIZE,
    shuffle=False
)
### Check out what's inside the training dataloader 
train_features_batch, train_labels_batch = next(iter(train_dataloader))
train_features_batch.shape, train_labels_batch.shape
# %%
torch.manual_seed(42)
random_idx = torch.randint(0, len(train_features_batch), size= [1]).item()
img, label = train_features_batch[random_idx], train_labels_batch[random_idx]
plt.imshow(img.squeeze(),cmap = 'gray')
plt.title(class_names[label])
plt.axis('Off')

# %%
class FashionMNISTModelV0(nn.Module):
    def __init__(self, input_shape:int, hidden_units : int, output_shape:int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape,
                      out_features=hidden_units),
            nn.Linear(in_features=hidden_units,
                      out_features=output_shape)
        )
    def forward(self, x):
        return self.layer_stack(x)
            
            
# %%
torch.manual_seed(42)
model_0 = FashionMNISTModelV0(
    input_shape=784,
    hidden_units=10,
    output_shape=len(class_names)
)
model_0.to('cpu')

# %%
from helper_functions import accuracy_fn 
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    params=model_0.parameters(),
    lr = 0.1
)

# %%
from tqdm.auto import tqdm
torch.manual_seed(42)
epochs = 3 
for epoch in tqdm(range(epochs)):
    print(f"Epoch : {epoch}\n-----")
    train_loss = 0 
    for batch , (X,y) in enumerate(train_dataloader):
        model_0.train()
        y_pred = model_0(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    train_loss /= len(train_dataloader)

    # Testing 
    test_loss, test_acc = 0, 0 
    model_0.eval()
    with torch.inference_mode():
        for X, y in test_dataloader:
            test_pred = model_0(X)
            test_loss = loss_fn(test_pred,y)
            test_acc += accuracy_fn(y_true=y,
                                    y_pred = test_pred.argmax(dim=1))   
        test_loss /= len(test_dataloader)
        test_acc /= len(test_dataloader)
    print(f"\nTrain loss: {train_loss:.5f} | Test loss : {test_loss:.5f}, Test acc : {test_acc:.2f}%\n")    

# %%
torch.manual_seed(42)
def eval_model(model : torch.nn.Module,
               data_loader : torch.utils.data.DataLoader,
               loss_fn : torch.nn.Module,
               accuracy_fn):
    loss , acc = 0, 0 
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            y_pred = model(X)
            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y_true = y,
                               y_pred = y_pred.argmax(dim=1))
        loss /= len(data_loader)
        acc /= len(data_loader)
    return {'model_name': model.__class__.__name__,
            'model_loss' : loss.item(),
            "model_acc" : acc}
model_0_results = eval_model(model = model_0,
                             data_loader=test_dataloader,
                             loss_fn=loss_fn,
                             accuracy_fn=accuracy_fn)
model_0_results

# %%
