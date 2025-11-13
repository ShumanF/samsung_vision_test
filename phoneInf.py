import numpy as np
import onnxruntime as ort
import csv

batch_size = 64
max_samples = 100
onnx_path = r'C:\Users\Suman\Desktop\samsung\mnist_cnn_model.onnx'
mnist_test_path = "mnist_test.csv"

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

def run_inference(onnx_path: str, images: np.ndarray, batch_size: int) -> np.ndarray:
    """Run ONNX Runtime inference and return logits array of shape (N, num_classes)."""
    sess = ort.InferenceSession(onnx_path)
    input_name = sess.get_inputs()[0].name
    outputs = []
    N = images.shape[0]
    for i in range(0, N, batch_size):
        batch = images[i : i + batch_size]
        # ONNX model expects shape (N,C,H,W); images are already (N,1,28,28)
        out = sess.run(None, {input_name: batch})
        logits = out[0]
        outputs.append(logits)
    return np.concatenate(outputs, axis=0)

images, labels = load_csv_samples(str(mnist_test_path), max_samples=1000)

broj = 2
logits = run_inference(str(onnx_path), images[broj:1000+1], batch_size=1000)
preds = np.argmax(logits, axis=1)

#mviz_single28(images[broj])
#print(images[broj].shape)
print(f"Label: {labels[broj]}")
print(f"Predicted Label: {preds}")