# shared/artifact_paths.py
"""
Single source of truth for MLflow artifact paths and filenames.

MLflow artifact tree for a training run:
    {run_id}/artifacts/
        onnx/
            classifier/
                model.onnx
                model.onnx.data     (may exist depending on exporter)
            embedder/
                model.onnx
                model.onnx.data     (may exist depending on exporter)
        reference_distribution.json
        class_gaussians.json
        feature_schema.json

Usage in training (logging):
    mlflow.log_artifacts(local_classifier_dir, MLFLOW_PATH_CLASSIFIER)

Usage in serving (downloading):
    onnx_dir = client.download_artifacts(run_id, MLFLOW_PATH_ONNX_ROOT, dst)
    path = resolve_classifier_path(onnx_dir)
"""

import os


# ── Filenames ────────────────────────────────────────────────────
ONNX_FILENAME = "model.onnx"
REFERENCE_DIST_FILENAME = "reference_distribution.json"
CLASS_GAUSSIANS_FILENAME = "class_gaussians.json"
FEATURE_SCHEMA_FILENAME = "feature_schema.json"
EVAL_REPORT_FILENAME = "evaluation_report.json"

# ── MLflow artifact paths (for log_artifacts / download_artifacts) ─
MLFLOW_PATH_ONNX_ROOT = "onnx"
MLFLOW_PATH_CLASSIFIER = "onnx/classifier"
MLFLOW_PATH_EMBEDDER = "onnx/embedder"


# ── Path resolution after download ──────────────────────────────
def resolve_classifier_path(onnx_download_dir: str) -> str:
    """
    Given the local dir returned by download_artifacts(run_id, "onnx", dst),
    return the full path to classifier/model.onnx.
    """
    path = os.path.join(onnx_download_dir, "classifier", ONNX_FILENAME)
    if not os.path.isfile(path):
        _dump_tree(onnx_download_dir, "classifier/" + ONNX_FILENAME)
    return path


def resolve_embedder_path(onnx_download_dir: str) -> str:
    """
    Given the local dir returned by download_artifacts(run_id, "onnx", dst),
    return the full path to embedder/model.onnx.
    """
    path = os.path.join(onnx_download_dir, "embedder", ONNX_FILENAME)
    if not os.path.isfile(path):
        _dump_tree(onnx_download_dir, "embedder/" + ONNX_FILENAME)
    return path


def _dump_tree(root: str, expected: str):
    """On failure, print what's actually on disk so debugging is immediate."""
    import sys
    print(f"\n  FATAL: Expected file '{expected}' not found in '{root}'", file=sys.stderr)
    print(f"         Actual contents:", file=sys.stderr)
    for dirpath, dirnames, filenames in os.walk(root):
        level = dirpath.replace(root, "").count(os.sep)
        indent = "           " + "  " * level
        print(f"{indent}{os.path.basename(dirpath)}/", file=sys.stderr)
        for f in filenames:
            fpath = os.path.join(dirpath, f)
            size = os.path.getsize(fpath)
            print(f"{indent}  {f}  ({size} bytes)", file=sys.stderr)
    raise FileNotFoundError(f"'{expected}' not found in '{root}'")
