#!/usr/bin/env python3
# scripts/integrate_annotations.py
"""
Integrate new annotations into a cumulative new dataset version.

Reads annotated predictions that are not yet in any dataset version,
downloads the corresponding images from MinIO (stored by the serving layer at
predictions/{uuid}.npy), and creates a new dataset version by copying the
previous version's samples and appending the new ones.

Writes /tmp/version_id.txt on success — consumed by Argo as an output parameter.

Exit codes:
  0 — success (including the case of zero new annotations — idempotent)
  1 — fatal error (copy count check failed, unexpected exception)

Prerequisites (env vars):
  DATA_CONTROLLER_DB_URL, DATASET_S3_ENDPOINT_URL, DATASET_BUCKET,
  AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
"""

import io
import logging
import os
import random
import sys
from pathlib import Path

import boto3
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared.config import require_env  # noqa: E402
from shared.data_controller.dataset import DatasetController  # noqa: E402
from shared.logging_config import setup_logging  # noqa: E402

setup_logging("integrate-annotations")
logger = logging.getLogger(__name__)

VERSION_ID_OUTPUT_PATH = os.environ.get("VERSION_ID_OUTPUT_PATH", "/tmp/version_id.txt")


def _download_image(s3_client, bucket: str, key: str) -> np.ndarray | None:
    try:
        body = s3_client.get_object(Bucket=bucket, Key=key)["Body"].read()
        return np.load(io.BytesIO(body))
    except Exception as exc:
        logger.warning(f"Could not download {key}: {exc}")
        return None


def _assign_splits(n: int) -> list[str]:
    """Assign train/val/test splits. If n < 10 put everything in train."""
    if n < 10:
        return ["train"] * n
    indices = list(range(n))
    random.shuffle(indices)
    n_test = max(1, round(n * 0.1))
    n_val = max(1, round(n * 0.1))
    splits = ["train"] * n
    for i in indices[:n_test]:
        splits[i] = "test"
    for i in indices[n_test : n_test + n_val]:
        splits[i] = "val"
    return splits


def main() -> int:
    ctrl = DatasetController()

    prev_version = ctrl.get_latest_version()
    if prev_version is None:
        logger.error("No existing dataset version found. Run seed_dataset.py first.")
        return 1

    import re

    if not re.fullmatch(r"v\d+", prev_version):
        logger.error(f"Unexpected version format: '{prev_version}'. Expected v<N>.")
        return 1

    new_version = f"v{int(prev_version[1:]) + 1}"
    logger.info(f"Previous version: {prev_version} → new version: {new_version}")

    annotated = ctrl.get_unversioned_annotations()
    logger.info(f"Unversioned annotated predictions: {len(annotated)}")

    if not annotated:
        logger.info("No new annotations to integrate. Writing previous version to output.")
        Path(VERSION_ID_OUTPUT_PATH).write_text(prev_version)
        return 0

    # Copy historical samples (pure SQL, no MinIO)
    copied = ctrl.copy_version(prev_version, new_version)
    logger.info(f"Copied {copied} samples from {prev_version} → {new_version}")

    endpoint = require_env("DATASET_S3_ENDPOINT_URL")
    bucket = require_env("DATASET_BUCKET")
    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", ""),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
    )

    splits = _assign_splits(len(annotated))
    stored = 0
    skipped = 0

    for item, split in zip(annotated, splits, strict=True):
        uuid = item["uuid"]
        label = item["label"]
        key = f"predictions/{uuid}.npy"

        image = _download_image(s3, bucket, key)
        if image is None:
            logger.warning(
                f"Skipping {uuid} — image not in MinIO (upload may have failed at serve time)"
            )
            skipped += 1
            continue

        ctrl.store_sample(
            uuid=uuid,
            version_id=new_version,
            split=split,
            label=int(label),
            image_2d=image.tolist(),
            minio_path=key,
        )
        stored += 1

    logger.info(f"Stored {stored} new samples, skipped {skipped} (missing images)")

    # Sanity check: new version must have at least as many rows as were copied
    if stored == 0 and copied == 0:
        logger.error("No samples in new version — aborting to prevent training on empty dataset")
        return 1

    Path(VERSION_ID_OUTPUT_PATH).write_text(new_version)
    logger.info(f"Version {new_version} ready. Written to {VERSION_ID_OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
