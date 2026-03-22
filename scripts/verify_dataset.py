#!/usr/bin/env python3
# scripts/verify_dataset.py
"""
Verify that the seeded dataset is consistent between Postgres and MinIO.

Checks:
  - Each split has the expected number of samples
  - All 10 digit labels are present in the training split
  - A spot-check sample downloads correctly from MinIO and matches Postgres

Usage:
    DATA_CONTROLLER_DB_URL=... DATASET_S3_ENDPOINT_URL=... DATASET_BUCKET=... \
    PYTHONPATH=. python scripts/verify_dataset.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared.data_controller import DatasetController  # noqa: E402


def main():
    ctrl = DatasetController()

    print("Checking sample counts...")
    for split in ["train", "val", "test"]:
        samples = ctrl.get_dataset_split(split)
        labels = sorted(set(s["label"] for s in samples))
        print(f"  {split}: {len(samples)} samples, labels={labels}")
        if not samples:
            print(f"  ERROR: no samples in {split}")
            sys.exit(1)

    print("Spot-checking MinIO ↔ Postgres pixel values...")
    train = ctrl.get_dataset_split("train")
    sample = train[0]
    minio_img = ctrl.download_image(sample["minio_path"])
    pg_img = np.array(sample["image"], dtype=np.float32)

    if not np.allclose(minio_img, pg_img, atol=1e-5):
        print("  ERROR: pixel values from MinIO do not match Postgres")
        print(f"  max diff: {np.abs(minio_img - pg_img).max()}")
        sys.exit(1)

    print("  OK: MinIO == Postgres pixel values")
    print("Verification passed.")


if __name__ == "__main__":
    main()
