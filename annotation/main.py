# annotation/main.py
"""Annotation job — labels candidate predictions using a file-based ground truth oracle.

For each prediction with annotation_status='candidate' in Postgres, this job:
  1. Looks up the UUID in the file-based oracle (uuids.npy + labels.npy loaded
     at startup) to retrieve the ground truth label.
  2. Writes the label back using AnnotationDataController.write_label(), which
     also advances annotation_status to 'annotated'.
  3. Logs a warning and skips predictions whose UUID is absent from the oracle
     (e.g. predictions from unknown images).

Environment variables:
  DATA_CONTROLLER_DB_URL       Required — PostgreSQL DSN.
  ANNOTATION_ORACLE_PATH       Required — directory containing uuids.npy and labels.npy.
  ANNOTATION_SAMPLES_PER_RUN   Max samples to annotate per execution (default: 10).
"""

from __future__ import annotations

import logging
import os
import sys

import numpy as np

from shared.config import require_env
from shared.data_controller._base import DataControllerError
from shared.data_controller.annotation import AnnotationDataController
from shared.logging_config import setup_logging

setup_logging("annotation")
logger = logging.getLogger(__name__)


def _load_oracle(oracle_path: str) -> dict[str, int]:
    """Load the ground truth oracle from uuids.npy + labels.npy.

    Args:
        oracle_path: Directory containing ``uuids.npy`` and ``labels.npy``.

    Returns:
        Mapping from UUID string to integer ground truth label.
    """
    uuids_path = os.path.join(oracle_path, "uuids.npy")
    labels_path = os.path.join(oracle_path, "labels.npy")
    try:
        uuids = np.load(uuids_path)
        labels = np.load(labels_path)
    except FileNotFoundError as exc:
        logger.critical("Oracle file not found: %s", exc)
        sys.exit(1)
    oracle = {str(u): int(lbl) for u, lbl in zip(uuids, labels)}
    logger.info(
        "Loaded annotation oracle: %d entries from %s", len(oracle), oracle_path
    )
    return oracle


def main() -> None:
    samples_per_run_str = os.environ.get("ANNOTATION_SAMPLES_PER_RUN", "10")
    try:
        samples_per_run = int(samples_per_run_str)
    except ValueError:
        logger.critical(
            "ANNOTATION_SAMPLES_PER_RUN must be a positive integer, got: %r",
            samples_per_run_str,
        )
        sys.exit(1)

    oracle_path = require_env("ANNOTATION_ORACLE_PATH")
    oracle = _load_oracle(oracle_path)

    logger.info("Annotation job starting. Samples per run: %d", samples_per_run)

    try:
        ctrl = AnnotationDataController()
    except DataControllerError as exc:
        logger.critical("Failed to initialise AnnotationDataController: %s", exc)
        sys.exit(1)

    try:
        candidates = ctrl.get_candidates(limit=samples_per_run)
    except DataControllerError as exc:
        logger.error("Failed to fetch candidates: %s", exc)
        sys.exit(1)

    if not candidates:
        logger.info("No candidate predictions found. Nothing to annotate.")
        return

    logger.info("Found %d candidate(s) to annotate.", len(candidates))

    annotated = 0
    skipped = 0
    for uuid in candidates:
        label = oracle.get(str(uuid))
        if label is None:
            logger.warning(
                "UUID %s not found in annotation oracle — skipping.", uuid
            )
            skipped += 1
            continue
        try:
            ctrl.write_label(uuid, label)
            logger.info("Annotated %s → label=%d", uuid, label)
            annotated += 1
        except DataControllerError as exc:
            logger.error("Failed to annotate %s: %s", uuid, exc)

    logger.info(
        "Annotation job complete. %d/%d annotated, %d skipped (not in oracle).",
        annotated,
        len(candidates),
        skipped,
    )


if __name__ == "__main__":
    main()
