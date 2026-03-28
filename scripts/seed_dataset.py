#!/usr/bin/env python3
# scripts/seed_dataset.py
"""
Seed the MNIST dataset into Postgres (metadata) and MinIO (image files).

Creates a dataset version (default: v0) containing train/val/test splits from
data/v0/.  Each sample is stored once per version/split in the dataset_samples
table.

Schema:
  dataset_samples  — (uuid, version_id, split, label, minio_path)

Idempotent: ON CONFLICT (uuid, version_id) DO NOTHING in Postgres.

Prerequisites:
  - make data.prepare (creates data/v0/ files)
  - make dc.infra.up (Postgres + MinIO running)
  - DATA_CONTROLLER_DB_URL, DATASET_S3_ENDPOINT_URL, DATASET_BUCKET,
    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY set in environment

Usage:
    DATA_CONTROLLER_DB_URL=postgresql://mlflow:mlflow@localhost:5432/mlflow \\
    DATASET_S3_ENDPOINT_URL=http://localhost:9000 \\
    DATASET_BUCKET=mnist-dataset \\
    PYTHONPATH=. python scripts/seed_dataset.py
"""

import os
import sys
from datetime import date
from uuid import UUID

import numpy as np

# Project root on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared.data_controller.dataset import DatasetController  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "v0")
SPLITS = ["train", "val", "test"]

# Dataset version created by this script.  Override via environment variable
# if seeding a second version from a different data source.
DATASET_VERSION = os.environ.get("DATASET_VERSION", "v0")


def seed_split(
    ctrl: DatasetController,
    version_id: str,
    split: str,
    date_prefix: str,
) -> int:
    """Store all samples for *split* in MinIO + Postgres under *version_id*."""
    images_path = os.path.join(DATA_DIR, split, "images.npy")
    labels_path = os.path.join(DATA_DIR, split, "labels.npy")
    uuids_path = os.path.join(DATA_DIR, split, "uuids.npy")

    if not os.path.exists(images_path):
        print(f"  [{split}] SKIP — {images_path} not found (run data.prepare first)")
        return 0

    images = np.load(images_path)  # (N, 14, 14) float32
    labels = np.load(labels_path)  # (N,) int64
    uuids = np.load(uuids_path)    # (N,) str — assigned at prepare time

    if not (len(images) == len(labels) == len(uuids)):
        raise ValueError(
            f"[{split}] Inconsistent array lengths: "
            f"images={len(images)} labels={len(labels)} uuids={len(uuids)}"
        )

    count = 0
    for i, (img, label, uuid_str) in enumerate(zip(images, labels, uuids, strict=True)):
        uuid = UUID(str(uuid_str))
        minio_path = f"{date_prefix}/{uuid}.npy"
        ctrl.store_sample(
            uuid=uuid,
            version_id=version_id,
            split=split,
            label=int(label),
            image_2d=img.tolist(),
            minio_path=minio_path,
        )
        count += 1
        if (i + 1) % 500 == 0 or (i + 1) == len(images):
            print(f"  [{split}] {i + 1}/{len(images)} seeded...", end="\r")

    print()  # newline after \r progress
    return count


def main():
    print("Connecting to DatasetController...")
    ctrl = DatasetController()

    date_prefix = date.today().strftime("%Y%m%d")
    print(f"Dataset version : {DATASET_VERSION}")
    print(f"MinIO date prefix: {date_prefix}")
    print()

    totals = {}
    for split in SPLITS:
        print(f"Seeding {split}...")
        n = seed_split(ctrl, DATASET_VERSION, split, date_prefix)
        totals[split] = n

    print()
    print("Seeding complete:")
    for split, n in totals.items():
        print(f"  {split}: {n} samples")


if __name__ == "__main__":
    main()
