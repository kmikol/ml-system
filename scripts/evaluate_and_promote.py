#!/usr/bin/env python3
# scripts/evaluate_and_promote.py
"""
Metric-gated conditional promotion of the newly trained model to Production.

Promotion decision (in priority order):
  1. No existing Production model → promote unconditionally.
  2. Baseline available (old Production model scored on new val split) →
     promote if new_val_acc >= baseline_val_acc - MIN_VAL_ACC_IMPROVEMENT.
     Both models are evaluated on the same data, so the comparison is fair
     even when the dataset changes between versions.
  3. Baseline computation failed (e.g. MinIO unavailable) → fall back to
     cross-version comparison: new_val_acc vs old run's logged val_acc.
     This is the original unfair comparison, used only as a last resort.

The baseline is logged to MLflow on the new run as "baseline_val_acc" so
every run carries a record of what the bar was at promotion time.

Exit codes:
  0 — always (not promoting is a valid outcome, not an error)

Prerequisites (env vars):
  MLFLOW_TRACKING_URI, MODEL_NAME, NEW_RUN_ID, VERSION_ID,
  DATA_CONTROLLER_DB_URL, DATASET_S3_ENDPOINT_URL, DATASET_BUCKET,
  AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY,
  MIN_VAL_ACC_IMPROVEMENT (optional, default 0.0)
"""

import logging
import os
import sys
import tempfile

import mlflow
import mlflow.tracking
import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared.config import require_env  # noqa: E402
from shared.data_controller.dataset import DatasetController  # noqa: E402
from shared.logging_config import setup_logging  # noqa: E402
from shared.model_artifact_controller import (  # noqa: E402
    ModelArtifactController,
    ModelArtifactError,
)

setup_logging("evaluate-and-promote")
logger = logging.getLogger(__name__)


def _score_on_val_split(classifier_path: str, val_samples: list[dict]) -> float:
    """Run the classifier ONNX on val_samples and return accuracy.

    Batches all samples into a single ONNX call for efficiency.
    Softmax is computed with the log-sum-exp trick (no scipy needed).
    """
    images = np.stack([np.array(s["image"]).flatten().astype(np.float32) for s in val_samples])
    labels = np.array([s["label"] for s in val_samples], dtype=np.int64)

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(classifier_path, opts)

    logits = session.run(["logits"], {"features": images})[0]
    # log-sum-exp softmax, then argmax
    shifted = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
    predictions = np.argmax(probs, axis=1)

    return float((predictions == labels).mean())


def main() -> int:
    try:
        return _main()
    except Exception as exc:
        logger.error(f"Unexpected error in evaluate-and-promote: {exc}", exc_info=True)
        return 0  # never fail the workflow — not promoting is a valid outcome


def _main() -> int:
    new_run_id = require_env("NEW_RUN_ID")
    version_id = require_env("VERSION_ID")
    model_name = require_env("MODEL_NAME")
    min_improvement = float(os.environ.get("MIN_VAL_ACC_IMPROVEMENT", "0.0"))
    tracking_uri = require_env("MLFLOW_TRACKING_URI")

    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()
    controller = ModelArtifactController()

    # ── Get new run's val_acc ─────────────────────────────────────────────────
    try:
        new_run = client.get_run(new_run_id)
        new_val_acc = new_run.data.metrics.get("val_acc", 0.0)
        logger.info(f"New run {new_run_id}: val_acc={new_val_acc:.4f}")
    except Exception as exc:
        logger.error(f"Could not retrieve new run {new_run_id}: {exc}")
        return 0

    # ── Get current Production run_id + logged val_acc ────────────────────────
    prod_run_id: str | None = None
    prod_val_acc: float | None = None
    try:
        prod_run_id = controller.get_production_run_id(model_name, "Production")
        prod_run = client.get_run(prod_run_id)
        prod_val_acc = prod_run.data.metrics.get("val_acc", 0.0)
        logger.info(f"Current Production run {prod_run_id}: val_acc={prod_val_acc:.4f}")
    except ModelArtifactError:
        logger.info("No current Production model — will promote unconditionally")
    except Exception as exc:
        logger.warning(f"Could not retrieve Production run metrics: {exc}")

    # ── Compute baseline: score Production model on the new val split ─────────
    # Both the new model and the baseline are now scored on the same data,
    # making the comparison fair even when the dataset changed between versions.
    baseline_val_acc: float | None = None
    if prod_run_id is not None:
        try:
            dataset_ctrl = DatasetController()
            val_samples = dataset_ctrl.get_dataset_split(version_id, "val")
            if not val_samples:
                logger.warning(f"Val split for {version_id} is empty — skipping baseline")
            else:
                with tempfile.TemporaryDirectory(prefix="eval_baseline_") as tmpdir:
                    classifier_path, _, _ = controller.download_serving_bundle(prod_run_id, tmpdir)
                    baseline_val_acc = _score_on_val_split(classifier_path, val_samples)

                logger.info(
                    f"Baseline: Production model on {version_id} val "
                    f"({len(val_samples)} samples): {baseline_val_acc:.4f}"
                )
                client.log_metric(new_run_id, "baseline_val_acc", baseline_val_acc)
        except Exception as exc:
            logger.warning(
                f"Baseline computation failed: {exc}. "
                "Falling back to cross-version val_acc comparison."
            )

    # ── Promotion decision ────────────────────────────────────────────────────
    if prod_run_id is None:
        should_promote = True
        reason = "no existing Production model"
    elif baseline_val_acc is not None:
        # Fair: both models evaluated on the same val split
        should_promote = new_val_acc >= baseline_val_acc - min_improvement
        reason = (
            f"new {new_val_acc:.4f} vs baseline {baseline_val_acc:.4f} "
            f"on {version_id} val (fair comparison)"
        )
    else:
        # Fallback: cross-version comparison — unfair but better than no gate
        ref = prod_val_acc if prod_val_acc is not None else 0.0
        should_promote = new_val_acc >= ref - min_improvement
        reason = (
            f"new {new_val_acc:.4f} vs old-run val_acc {ref:.4f} "
            f"(cross-version fallback — baseline unavailable)"
        )

    logger.info(f"Promote decision: {should_promote} ({reason})")

    if not should_promote:
        logger.info("Skipping promotion.")
        return 0

    # ── Find registered version and promote ───────────────────────────────────
    try:
        versions = client.search_model_versions(f"name='{model_name}' and run_id='{new_run_id}'")
        if not versions:
            logger.warning(
                f"No registered version found for run {new_run_id}. "
                "Training may have failed to register — skipping promotion."
            )
            return 0
        version = versions[0].version
        logger.info(f"Found registered version {version} for run {new_run_id}")
    except Exception as exc:
        logger.error(f"Failed to search model versions: {exc}")
        return 0

    try:
        controller.promote_model(model_name, version)
        logger.info(f"Promoted {model_name} v{version} → Production")
    except ModelArtifactError as exc:
        logger.error(f"Promotion failed: {exc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
