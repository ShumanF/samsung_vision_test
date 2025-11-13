"""Simple tester for an MNIST ONNX model using ONNX Runtime and a CSV test file.

Usage:
  python test_onnx.py --onnx mnist_cnn_model.onnx --csv mnist_test.csv --max 1000

The script expects the CSV to be either:
 - 785 columns per row: label (0-9) followed by 784 pixel values (0-255)
 - 784 columns per row: only pixels

The script normalizes pixels to [0,1] then to [-1,1] using (x-0.5)/0.5 to match
the preprocessing used when the PyTorch model was trained/exported.
"""
import argparse
import csv
import numpy as np
import onnxruntime as ort
from pathlib import Path
from typing import Tuple, List, Optional


import csv
import torch
import numpy as np

def normalize(t: torch.Tensor) -> torch.Tensor:
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

            pix = torch.tensor([int(p) for p in pixels], dtype=torch.float32).div_(255.0)
            img = pix.view(1, 28, 28)
            img = normalize(img)
            images.append(img.numpy())
            labels.append(label)

    # Convert to NumPy arrays
    images_np = np.stack(images, axis=0)
    labels_np = np.array(labels, dtype=object)  # allows None values

    return images_np, labels_np


def run_inference(onnx_path: str, images: np.ndarray, batch_size: int = 64) -> np.ndarray:
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--onnx', default='mnist_cnn_model.onnx', help='Path to ONNX model')
    p.add_argument('--csv', default='mnist_test.csv', help='CSV file with test images')
    p.add_argument('--max', type=int, default=None, help='Max number of samples to load')
    p.add_argument('--batch', type=int, default=64, help='Batch size for ONNX Runtime')
    args = p.parse_args()

    onnx_path = Path(args.onnx)
    csv_path = Path(args.csv)
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    print(f"Loading CSV: {csv_path} (max={args.max})")
    images, labels = load_csv_samples(str(csv_path), max_samples=args.max)
    if images.shape[0] == 0:
        print("No images loaded. Exiting.")
        return

    print(f"Loaded {images.shape[0]} images. Running ONNX inference...")
    logits = run_inference(str(onnx_path), images, batch_size=args.batch)
    preds = np.argmax(logits, axis=1)

    # show some sample predictions
    print("Sample predictions:")
    nshow = min(100, images.shape[0])
    for i in range(nshow):
        lbl = labels[i]
        print(f"#{i}: label={lbl}  pred={int(preds[i])}")

    # If labels are present, compute accuracy
    if labels and any(lbl is not None for lbl in labels):
        true = np.array([lbl if lbl is not None else -1 for lbl in labels], dtype=int)
        mask = true >= 0
        acc = (preds[mask] == true[mask]).mean() if mask.any() else float('nan')
        print(f"Accuracy on {mask.sum()} labeled samples: {acc*100:.2f}%")
    else:
        print("No labels in CSV, accuracy not computed.")


if __name__ == '__main__':
    main()
