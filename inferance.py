import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets
Minst = datasets.MNIST
from random import randrange
import csv
import numpy as np


# We'll read test images from a CSV instead of the raw MNIST files in ./data.
# CSV format expected (common Kaggle style): first column is label (optional),
# remaining 784 columns are pixel values (0-255) in row-major order.

# Normalizer: accepts a tensor in [C,H,W] with values in [0,1]
#normalize = torchvision.transforms.Normalize((0.5,), (0.5,))

def normalize(t: np.array) -> np.array:
    # match normalization used in training ((x - 0.5) / 0.5 => [-1, 1])
    return (t - 0.5) / 0.5

def load_csv_samples(csv_path: str, max_samples: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Load samples from a CSV file.

    Returns:
        (images, labels): tuple of NumPy arrays
        - images shape: [N, 1, 28, 28], dtype=float32, normalized to [-1,1]
        - labels shape: [N], dtype=int or None
    """
    images = []
    labels = []

    with open(csv_path, 'r', newline='') as f:
        first = f.readline()
        f.seek(0)
        first_cols = first.strip().split(',')
        has_header = any(any(ch.isalpha() for ch in col) for col in first_cols)
        reader = csv.reader(f)
        if has_header:
            next(reader)

        for i, row in enumerate(reader):
            if max_samples is not None and len(images) >= max_samples:
                break
            if not row:
                continue

            if len(row) >= 785:
                label = int(row[0])
                pixels = row[1:785]
            elif len(row) == 784:
                label = None
                pixels = row
            else:
                continue  # skip malformed rows

            pix = np.array([int(p) for p in pixels], dtype=np.float32) / 255.0
            img = pix.reshape(1, 28, 28)
            img = normalize(img)
            images.append(img)
            labels.append(label)

    # Convert to NumPy arrays
    images_np = np.stack(images, axis=0)
    labels_np = np.array(labels, dtype=object)  # allows None values

    return images_np, labels_np

from model import Net

model = Net()
print(model)

# 1. Define the file path where the model was saved.
model_save_path = 'mnist_cnn_model.pth'

# 2. Create a new instance of the model with the same architecture as the saved model.
model = Net() # Assuming the Net class is defined in this environment

# 3. Load the state dictionary from the saved file. Use CPU map_location to be safe
model.load_state_dict(torch.load(model_save_path, map_location=torch.device('cpu')))

# 4. Set the loaded model to evaluation mode
model.eval()

print(f"Model loaded successfully from: {model_save_path}")

# Load samples from CSV and pick a random one for quick testing
csv_path = 'mnist_test.csv'
samples = load_csv_samples(csv_path,10)
if len(samples) == 0:
    raise FileNotFoundError(f"No samples found in CSV: {csv_path}")

idx = randrange(len(samples))
images, labels = samples # images shaped: [1, 28, 28], array of labess

def mviz_single28(img):
    # Accepts [1,28,28] or [28,28]
    if img.ndim == 3 and img.shape[0] == 1:
        img = img.squeeze(0)
    elif img.ndim != 2:
        raise ValueError(f"Expected [1,28,28] or [28,28], got {img.shape}")

    # Normalize from [-1,1] → [0,1]
    img = ((img + 1) / 2).clip(0, 1)

    for row in img.tolist():
        line = [f"\033[38;5;{232 + int(val * 23)}m██" for val in row]
        print("".join(line) + "\033[0m")

broj = 5
mviz_single28(images[broj])
print(images[broj].shape)

with torch.no_grad():
    # model expects input shape [N,C,H,W]
    #inputs = images[0].unsqueeze(0)
    inputs = torch.from_numpy(images[broj]).unsqueeze(0)
    outputs = model(inputs)
    _, predicted = torch.max(outputs.data, 1)
    print(f"Label: {labels[broj]}")
    print(f"Predicted Label: {predicted.item()}")