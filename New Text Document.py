import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets
Minst = datasets.MNIST

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

print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
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
print("CNN model created:")
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

image, label = train_dataset[5]  # image shape: [1, 28, 28]

# Print label and image
print(f"Label: {label}")
mviz_single28(image)
print(image.shape)
with torch.no_grad():
    outputs = model(image)
    _, predicted = torch.max(outputs.data, 1)
    print(f"Predicted Label: {predicted.item()}")