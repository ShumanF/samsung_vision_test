import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets
Minst = datasets.MNIST
from random import randrange

# create transforms and load MNIST train/test datasets (uses Minst defined in previous cell)
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5,), (0.5,))
])

train_dataset = Minst(root='./data', train=True, download=False, transform=transform)
test_dataset = Minst(root='./data', train=False, download=False, transform=transform)

# optional: create dataloaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = Net()
print(model)

# 1. Define the file path where the model was saved.
model_save_path = 'mnist_cnn_model.pth'

# 2. Create a new instance of the model with the same architecture as the saved model.
model = Net() # Assuming the Net class is defined in this environment

# 3. Load the state dictionary from the saved file.
model.load_state_dict(torch.load(model_save_path))

# 4. Set the loaded model to evaluation mode
model.eval()

print(f"Model loaded successfully from: {model_save_path}")

image, label = train_dataset[randrange(100, 1000)]  # image shape: [1, 28, 28]

def mviz_single28(img: Tensor):
    # Accepts [1,28,28] or [28,28]
    if img.ndim == 3 and img.shape[0] == 1:
        img = img.squeeze(0)
    elif img.ndim != 2:
        raise ValueError(f"Expected [1,28,28] or [28,28], got {img.shape}")

    # Normalize from [-1,1] → [0,1]
    img = ((img + 1) / 2).clamp(0, 1)

    for row in img.tolist():
        line = [f"\033[38;5;{232 + int(val * 23)}m██" for val in row]
        print("".join(line) + "\033[0m")

    
mviz_single28(image)
print(image.shape)

with torch.no_grad():
    outputs = model(image)
    _, predicted = torch.max(outputs.data, 1)
    print(f"Label: {label}")
    print(f"Predicted Label: {predicted.item()}")