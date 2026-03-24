# annotation/main.py
"""Annotation job — simulates ground truth labeling of MNIST predictions.

For each prediction with annotation_status='candidate' in Postgres, this job:
  1. Looks up the ground truth label from the dataset_samples table by matching
     the prediction_id (UUID) to the original MNIST sample's sample_id.
  2. Writes the label back using AnnotationDataController.write_label(), which
     also advances annotation_status to 'annotated'.

Environment variables:
  DATA_CONTROLLER_DB_URL       Required — PostgreSQL DSN.
  ANNOTATION_SAMPLES_PER_RUN   Max samples to annotate per execution (default: 10).
                               Set to a higher value to annotate more samples at once.
"""

from __future__ import annotations

import logging
import os
import sys

from shared.data_controller._base import DataControllerError
from shared.data_controller.annotation import AnnotationDataController
from shared.logging_config import setup_logging

setup_logging("annotation")
logger = logging.getLogger(__name__)


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
    for prediction_id, label in candidates:
        try:
            ctrl.write_label(prediction_id, label)
            logger.info("Annotated %s → label=%d", prediction_id, label)
            annotated += 1
        except DataControllerError as exc:
            logger.error("Failed to annotate %s: %s", prediction_id, exc)

    logger.info(
        "Annotation job complete. %d/%d samples annotated.",
        annotated,
        len(candidates),
    )


if __name__ == "__main__":
    main()
