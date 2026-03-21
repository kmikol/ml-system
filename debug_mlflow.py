#!/usr/bin/env python3
"""
Diagnostic: inspect MLflow artifacts and test ONNX loading.

From host:   MLFLOW_S3_ENDPOINT_URL=http://localhost:9000 python debug_mlflow.py
From devcontainer: python debug_mlflow.py
"""

import os
import sys
import tempfile

import mlflow
from shared.artifact_paths import (
    MLFLOW_PATH_ONNX_ROOT,
    resolve_classifier_path,
    resolve_embedder_path,
)
from shared.config import require_env

MLFLOW_TRACKING_URI = require_env("MLFLOW_TRACKING_URI")
MODEL_NAME = require_env("MODEL_NAME")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = mlflow.tracking.MlflowClient()


def section(title):
    print(f"\n{'=' * 60}\n  {title}\n{'=' * 60}")


section("Registered Models")
models = client.search_registered_models()
if not models:
    print("  No models found. Run training first.")
    sys.exit(1)
for m in models:
    print(f"  {m.name}")
    for v in m.latest_versions:
        print(f"    v{v.version}: stage={v.current_stage} run={v.run_id}")


section("Production Model")
versions = client.search_model_versions(f"name='{MODEL_NAME}'")
prod = next((v for v in versions if v.current_stage == "Production"), None)
if not prod:
    print(f"  No Production model for '{MODEL_NAME}'")
    sys.exit(1)
run_id = prod.run_id
print(f"  Run:     {run_id}")
print(f"  Version: {prod.version}")
print(f"  Source:  {prod.source}")


section(f"Artifacts for {run_id}")


def list_artifacts(path="", indent=0):
    for a in client.list_artifacts(run_id, path):
        pfx = "  " + "  " * indent
        sz = f" ({a.file_size}b)" if a.file_size else " [dir]"
        print(f"{pfx}{a.path}{sz}")
        if a.is_dir:
            list_artifacts(a.path, indent + 1)


list_artifacts()


section("Download + Load Test")
with tempfile.TemporaryDirectory() as tmpdir:
    onnx_dir = client.download_artifacts(run_id, MLFLOW_PATH_ONNX_ROOT, tmpdir)
    print(f"  Downloaded to: {onnx_dir}")

    print("\n  Local files:")
    for root, _dirs, files in os.walk(onnx_dir):
        lvl = root.replace(onnx_dir, "").count(os.sep)
        print(f"  {'  ' * lvl}{os.path.basename(root)}/")
        for f in files:
            sz = os.path.getsize(os.path.join(root, f))
            print(f"  {'  ' * (lvl + 1)}{f}  ({sz}b)")

    cls_path = resolve_classifier_path(onnx_dir)
    emb_path = resolve_embedder_path(onnx_dir)
    print(f"\n  Classifier: {cls_path}")
    print(f"  Embedder:   {emb_path}")

    try:
        import numpy as np
        import onnxruntime as ort

        for label, path in [("Classifier", cls_path), ("Embedder", emb_path)]:
            sess = ort.InferenceSession(path)
            inp = sess.get_inputs()[0]
            out = sess.get_outputs()
            dummy = np.random.randn(1, inp.shape[1]).astype(np.float32)
            result = sess.run(None, {inp.name: dummy})
            print(f"\n  {label}:")
            print(f"    Input:  {inp.name} {inp.shape}")
            print(f"    Output: {[(o.name, o.shape) for o in out]}")
            print(f"    Test:   shape={result[0].shape} ✅")
    except ImportError:
        print("\n  onnxruntime not installed, skipping load test")

print(f"\n{'=' * 60}\n  Done.\n{'=' * 60}")
