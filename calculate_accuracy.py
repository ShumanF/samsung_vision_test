"""Calculate and print accuracy of first 1000 images using ONNX model."""
import csv
import numpy as np
import torch
import onnxruntime as ort
from pathlib import Path


def normalize(t: torch.Tensor) -> torch.Tensor:
    """Normalize from [0,1] to [-1,1] using (x-0.5)/0.5"""
    return (t - 0.5) / 0.5


def load_csv_samples(csv_path: str, max_samples: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    """Load samples from CSV file.
    
    Returns:
        (images, labels): tuple of NumPy arrays
        - images shape: [N, 1, 28, 28], dtype=float32, normalized to [-1,1]
        - labels shape: [N], dtype=int
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
            if len(images) >= max_samples:
                break
            if not row or len(row) < 785:
                continue

            label = int(row[0])
            pixels = row[1:785]
            
            pix = torch.tensor([int(p) for p in pixels], dtype=torch.float32).div_(255.0)
            img = pix.view(1, 28, 28)
            img = normalize(img)
            images.append(img.numpy())
            labels.append(label)

    images_np = np.stack(images, axis=0) if images else np.array([])
    labels_np = np.array(labels, dtype=int)

    return images_np, labels_np


def run_inference(onnx_path: str, images: np.ndarray, batch_size: int = 64) -> np.ndarray:
    """Run ONNX inference and return predictions."""
    sess = ort.InferenceSession(onnx_path)
    input_name = sess.get_inputs()[0].name
    outputs = []
    N = images.shape[0]
    
    for i in range(0, N, batch_size):
        batch = images[i:i + batch_size]
        out = sess.run(None, {input_name: batch})
        logits = out[0]
        outputs.append(logits)
    
    logits = np.concatenate(outputs, axis=0)
    return np.argmax(logits, axis=1)


def main():
    onnx_path = 'mnist_cnn_model.onnx'
    csv_path = 'mnist_test.csv'
    max_samples = 1000
    batch_size = 64

    # Check files exist
    if not Path(onnx_path).exists():
        print(f"Error: ONNX model not found: {onnx_path}")
        return
    if not Path(csv_path).exists():
        print(f"Error: CSV file not found: {csv_path}")
        return

    print(f"Loading first {max_samples} images from {csv_path}...")
    images, labels = load_csv_samples(csv_path, max_samples=max_samples)
    
    if images.shape[0] == 0:
        print("Error: No images loaded from CSV")
        return

    print(f"Loaded {images.shape[0]} images with labels")
    print(f"Running ONNX inference on {images.shape[0]} images (batch_size={batch_size})...")
    predictions = run_inference(onnx_path, images, batch_size=batch_size)

    # Calculate accuracy
    accuracy = (predictions == labels).mean()
    correct = (predictions == labels).sum()
    total = len(labels)

    print("\n" + "="*60)
    print(f"ACCURACY RESULTS")
    print("="*60)
    print(f"Total images:     {total}")
    print(f"Correct:          {correct}")
    print(f"Incorrect:        {total - correct}")
    print(f"Accuracy:         {accuracy * 100:.2f}%")
    print("="*60)

    # Show first 20 predictions
    print("\nFirst 20 predictions:")
    print(f"{'Index':<8}{'True Label':<15}{'Predicted':<15}{'Correct':<10}")
    print("-" * 50)
    for i in range(min(50, total)):
        correct_mark = "✓" if predictions[i] == labels[i] else "✗"
        print(f"{i:<8}{labels[i]:<15}{predictions[i]:<15}{correct_mark:<10}")


if __name__ == '__main__':
    main()
