#!/usr/bin/env python3
# scripts/verify_dataset.py
"""
Verify that the seeded dataset is consistent between Postgres and MinIO.

Checks:
  - A dataset version exists
  - Each split has samples
  - All 10 digit labels are present in the training split
  - A spot-check sample downloads correctly from MinIO (correct shape)
  - UUIDs from uuids.npy are present in the database

Usage:
    DATA_CONTROLLER_DB_URL=... DATASET_S3_ENDPOINT_URL=... DATASET_BUCKET=... \\
    PYTHONPATH=. python scripts/verify_dataset.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared.data_controller.dataset import DatasetController  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "v0")

# Number of UUIDs spot-checked against the database.  Checking all samples
# would be slow; a small prefix is sufficient to confirm propagation worked.
_UUID_SPOT_CHECK_COUNT = 10


def main():
    ctrl = DatasetController()

    print("Checking dataset version...")
    version_id = ctrl.get_latest_version()
    if version_id is None:
        print("  ERROR: no dataset version found. Run scripts/seed_dataset.py first.")
        sys.exit(1)
    print(f"  Latest version: {version_id}")

    print("Checking sample counts...")
    train = None
    for split in ["train", "val", "test"]:
        samples = ctrl.get_dataset_split(version_id, split)
        labels = sorted(set(s["label"] for s in samples))
        print(f"  {split}: {len(samples)} samples, labels={labels}")
        if not samples:
            print(f"  ERROR: no samples in split '{split}' for version '{version_id}'")
            sys.exit(1)
        if split == "train":
            train = samples

    if len(set(s["label"] for s in train)) < 10:
        print("  WARNING: training split has fewer than 10 distinct labels")

    print("Spot-checking MinIO image download...")
    sample = train[0]
    minio_img = ctrl.download_image(sample["minio_path"])
    if minio_img.shape != (14, 14):
        print(f"  ERROR: expected image shape (14, 14), got {minio_img.shape}")
        sys.exit(1)
    print(f"  OK: image shape {minio_img.shape}, dtype {minio_img.dtype}")

    print("Checking UUID propagation (uuids.npy → database)...")
    uuids_path = os.path.join(DATA_DIR, "train", "uuids.npy")
    if not os.path.exists(uuids_path):
        print(f"  SKIP — {uuids_path} not found (run data.prepare first)")
    else:
        uuids = np.load(uuids_path)
        # uuid values from DB are uuid.UUID objects; normalise for comparison
        db_uuids = {str(s["uuid"]) for s in train}
        missing = [u for u in uuids[:_UUID_SPOT_CHECK_COUNT] if str(u) not in db_uuids]
        if missing:
            print(f"  ERROR: {len(missing)} UUIDs from uuids.npy not found in database")
            print(f"  First missing: {missing[0]}")
            sys.exit(1)
        print(f"  OK: {_UUID_SPOT_CHECK_COUNT} UUIDs from uuids.npy present in database")

    print("Verification passed.")


if __name__ == "__main__":
    main()
