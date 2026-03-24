#!/usr/bin/env python3
# scripts/prepare_mnist.py
"""
Download MNIST, resize images to 14x14, normalize to float32 [0, 1], and
partition into the v0 dataset (10% of training samples) and a remaining pool.

A UUID is assigned to every sample at creation time and saved as uuids.npy
alongside images.npy and labels.npy.  Downstream scripts (seed_dataset.py,
load_test.py) read these files so that each sample carries the same identifier
throughout the entire pipeline.

Output layout:
    data/
    ├── v0/
    │   ├── train/   images.npy (4800×14×14)   labels.npy (4800,)   uuids.npy (4800,)
    │   ├── val/     images.npy (600×14×14)     labels.npy (600,)    uuids.npy (600,)
    │   └── test/    images.npy (600×14×14)     labels.npy (600,)    uuids.npy (600,)
    ├── remaining/   images.npy (54000×14×14)   labels.npy (54000,)  uuids.npy (54000,)
    └── mnist_test/  images.npy (10000×14×14)   labels.npy (10000,)  uuids.npy (10000,)

Usage:
    PYTHONPATH=. python scripts/prepare_mnist.py
"""

import os
import uuid

import numpy as np
from PIL import Image

TARGET_H, TARGET_W = 14, 14
V0_FRACTION = 0.10
TRAIN_RATIO, VAL_RATIO = 0.80, 0.10  # remaining 0.10 → test
SEED = 42
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def _resize_images(raw_images):
    """raw_images: numpy uint8 array of shape (N, H, W). Returns float32 (N, 14, 14) in [0,1]."""
    out = np.empty((len(raw_images), TARGET_H, TARGET_W), dtype=np.float32)
    for i, img in enumerate(raw_images):
        pil = Image.fromarray(img).resize((TARGET_W, TARGET_H), Image.BILINEAR)
        out[i] = np.array(pil, dtype=np.float32) / 255.0
    return out


def _make_uuids(n: int) -> np.ndarray:
    """Return a (n,) array of UUID strings, one per sample."""
    return np.array([str(uuid.uuid4()) for _ in range(n)])


def _save(path, images, labels, uuids):
    os.makedirs(path, exist_ok=True)
    np.save(os.path.join(path, "images.npy"), images)
    np.save(os.path.join(path, "labels.npy"), labels)
    np.save(os.path.join(path, "uuids.npy"), uuids)
    print(f"  Saved {len(images)} samples → {path}")


def main():
    try:
        from torchvision.datasets import MNIST
    except ImportError:
        raise SystemExit(
            "torchvision is required to download MNIST.\n"
            "Install it with: pip install torchvision"
        )

    raw_dir = os.path.join(DATA_DIR, "raw")
    print("Downloading MNIST...")
    train_ds = MNIST(root=raw_dir, train=True, download=True)
    test_ds = MNIST(root=raw_dir, train=False, download=True)

    # ── Official test set → data/mnist_test/ ─────────────────────
    print("Processing official test set (10,000 samples)...")
    test_images = _resize_images(test_ds.data.numpy())
    test_labels = test_ds.targets.numpy().astype(np.int64)
    test_uuids = _make_uuids(len(test_images))
    _save(os.path.join(DATA_DIR, "mnist_test"), test_images, test_labels, test_uuids)

    # ── Training set: shuffle, take 10% for v0 ───────────────────
    print("Processing training set (60,000 samples)...")
    all_images = _resize_images(train_ds.data.numpy())
    all_labels = train_ds.targets.numpy().astype(np.int64)
    all_uuids = _make_uuids(len(all_images))

    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(all_images))
    n_v0 = int(len(idx) * V0_FRACTION)
    v0_idx = idx[:n_v0]          # 6,000 samples
    remaining_idx = idx[n_v0:]   # 54,000 samples

    # ── Remaining pool ────────────────────────────────────────────
    print(f"Saving remaining pool ({len(remaining_idx)} samples)...")
    _save(
        os.path.join(DATA_DIR, "remaining"),
        all_images[remaining_idx],
        all_labels[remaining_idx],
        all_uuids[remaining_idx],
    )

    # ── v0: split 80/10/10 ────────────────────────────────────────
    n_train = int(n_v0 * TRAIN_RATIO)
    n_val = int(n_v0 * VAL_RATIO)
    # remaining goes to test

    v0_images = all_images[v0_idx]
    v0_labels = all_labels[v0_idx]
    v0_uuids = all_uuids[v0_idx]

    train_end = n_train
    val_end = n_train + n_val

    splits = {
        "train": (v0_images[:train_end], v0_labels[:train_end], v0_uuids[:train_end]),
        "val":   (v0_images[train_end:val_end], v0_labels[train_end:val_end], v0_uuids[train_end:val_end]),
        "test":  (v0_images[val_end:], v0_labels[val_end:], v0_uuids[val_end:]),
    }

    print(f"\nPartitioning v0 ({n_v0} samples, 80/10/10 split)...")
    for split, (imgs, lbls, uids) in splits.items():
        _save(os.path.join(DATA_DIR, "v0", split), imgs, lbls, uids)

    print("\nDone.")
    print(f"  v0/train : {len(splits['train'][0])} samples")
    print(f"  v0/val   : {len(splits['val'][0])} samples")
    print(f"  v0/test  : {len(splits['test'][0])} samples")
    print(f"  remaining: {len(remaining_idx)} samples")
    print(f"  mnist_test: {len(test_images)} samples")


if __name__ == "__main__":
    main()
