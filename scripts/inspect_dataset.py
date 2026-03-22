#!/usr/bin/env python3
# scripts/inspect_dataset.py
"""
Fetch 16 random training samples from the DatasetController and plot
them in a 4x4 grid with their labels.

Usage:
    DATA_CONTROLLER_DB_URL=postgresql://mlflow:mlflow@localhost:5432/mlflow \
    DATASET_S3_ENDPOINT_URL=http://localhost:9000 \
    DATASET_BUCKET=mnist-dataset \
    PYTHONPATH=. python scripts/inspect_dataset.py [--split train] [--out grid.png]
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared.data_controller.dataset import DatasetController  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--out", default=None, help="Save path (e.g. grid.png). Shows interactively if omitted.")
    args = parser.parse_args()

    print(f"Fetching {args.split} samples from DatasetController...")
    ctrl = DatasetController()
    samples = ctrl.get_dataset_split(args.split)

    if not samples:
        print(f"No samples found in split '{args.split}'. Run make data.seed first.")
        sys.exit(1)

    print(f"  {len(samples)} samples available")

    rng = np.random.default_rng(42)
    chosen = rng.choice(len(samples), size=min(16, len(samples)), replace=False)

    fig, axes = plt.subplots(4, 4, figsize=(6, 6))
    fig.suptitle(f"MNIST 14×14 — split: {args.split}", fontsize=12)

    for ax, idx in zip(axes.flat, chosen):
        s = samples[idx]
        img = np.array(s["image"], dtype=np.float32)
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(str(s["label"]), fontsize=10)
        ax.axis("off")

    # Hide any unused axes if fewer than 16 samples
    for ax in list(axes.flat)[len(chosen):]:
        ax.axis("off")

    plt.tight_layout()

    if args.out:
        plt.savefig(args.out, dpi=120)
        print(f"Saved → {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
