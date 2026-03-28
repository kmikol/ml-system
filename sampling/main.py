# sampling/main.py
"""Sampling job — selects unannotated predictions and marks them as annotation candidates.

Queries predictions with annotation_status='none' and atomically advances up to
SAMPLING_CANDIDATES_PER_RUN rows to annotation_status='candidate' so the
annotation job can resolve their ground truth labels from the file-based oracle.

This job is the first step of the sample-and-label Argo workflow. It must run
before the annotation job, which processes only 'candidate' rows.

Environment variables:
  DATA_CONTROLLER_DB_URL       Required — PostgreSQL DSN.
  SAMPLING_CANDIDATES_PER_RUN  Max predictions to mark per execution (default: 50).
"""

from __future__ import annotations

import logging
import os
import sys

from shared.data_controller._base import DataControllerError
from shared.data_controller.sampling import SamplingDataController
from shared.logging_config import setup_logging

setup_logging("sampling")
logger = logging.getLogger(__name__)


def main() -> None:
    candidates_per_run_str = os.environ.get("SAMPLING_CANDIDATES_PER_RUN", "50")
    try:
        candidates_per_run = int(candidates_per_run_str)
    except ValueError:
        logger.critical(
            "SAMPLING_CANDIDATES_PER_RUN must be a positive integer, got: %r",
            candidates_per_run_str,
        )
        sys.exit(1)

    logger.info("Sampling job starting. Candidates per run: %d", candidates_per_run)

    try:
        ctrl = SamplingDataController()
    except DataControllerError as exc:
        logger.critical("Failed to initialise SamplingDataController: %s", exc)
        sys.exit(1)

    try:
        marked = ctrl.select_and_mark_candidates(limit=candidates_per_run)
    except DataControllerError as exc:
        logger.error("Failed to select candidates: %s", exc)
        sys.exit(1)

    if not marked:
        logger.info("No unannotated predictions found. Nothing to mark.")
        return

    logger.info("Marked %d prediction(s) as candidate.", len(marked))


if __name__ == "__main__":
    main()
