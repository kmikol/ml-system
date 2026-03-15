# training/main.py
"""
Training pipeline: generate data → train → export ONNX → register in MLflow.
Usage: python -m training.main
"""

import os
import json
import tempfile
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
import mlflow
import onnx

from shared.config import require_env
from shared.artifact_paths import (
    ONNX_FILENAME,
    MLFLOW_PATH_CLASSIFIER,
    MLFLOW_PATH_EMBEDDER,
    REFERENCE_DIST_FILENAME,
    CLASS_GAUSSIANS_FILENAME,
    FEATURE_SCHEMA_FILENAME,
)
from shared.schemas.feature_schema import (
    FEATURE_NAMES, FEATURE_SCHEMA, INPUT_DIM, EMBEDDING_DIM, NUM_CLASSES,
)
from training.model import Classifier, ClassifierWrapper, EmbedderWrapper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── All config from env, no defaults ─────────────────────────────
MLFLOW_TRACKING_URI = require_env("MLFLOW_TRACKING_URI")
MODEL_NAME = require_env("MODEL_NAME")
MAX_EPOCHS = int(require_env("TRAINING_MAX_EPOCHS"))
SEED = int(require_env("TRAINING_SEED"))
BATCH_SIZE = int(require_env("TRAINING_BATCH_SIZE"))
LR = float(require_env("TRAINING_LR"))


def generate_synthetic_data(n_samples: int, seed: int):
    rng = np.random.default_rng(seed)
    class_means = {
        0: np.array([30.0, 35000.0, 620.0, 1.8, 3.0]),
        1: np.array([45.0, 65000.0, 710.0, 1.0, 8.0]),
        2: np.array([60.0, 95000.0, 780.0, 0.4, 15.0]),
    }
    class_stds = {
        0: np.array([8.0, 12000.0, 50.0, 0.5, 2.0]),
        1: np.array([10.0, 18000.0, 40.0, 0.3, 3.0]),
        2: np.array([7.0, 15000.0, 30.0, 0.2, 4.0]),
    }
    class_priors = np.array([0.4, 0.35, 0.25])

    labels = rng.choice(NUM_CLASSES, size=n_samples, p=class_priors)
    features = np.zeros((n_samples, INPUT_DIM))
    for cls in range(NUM_CLASSES):
        mask = labels == cls
        features[mask] = rng.normal(class_means[cls], class_stds[cls], (mask.sum(), INPUT_DIM))

    for i, name in enumerate(FEATURE_NAMES):
        spec = FEATURE_SCHEMA[name]
        features[:, i] = np.clip(features[:, i], spec["min"], spec["max"])

    return features, labels


def make_dataloaders(features, labels, seed, batch_size):
    n = len(features)
    n_val = int(n * 0.2)
    indices = np.random.default_rng(seed).permutation(n)

    def to_loader(idx, shuffle):
        x = torch.tensor(features[idx], dtype=torch.float32)
        y = torch.tensor(labels[idx], dtype=torch.long)
        return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle)

    return to_loader(indices[n_val:], True), to_loader(indices[:n_val], False)


def export_onnx(model: Classifier, export_dir: str):
    """
    Export classifier and embedder as ONNX into separate subdirectories.
    Each subdir is logged to MLflow as a directory artifact, so any
    companion .data files are included automatically.
    """
    model.eval()
    dummy = torch.randn(1, INPUT_DIM)

    # ── Classifier ──
    cls_dir = os.path.join(export_dir, "model")
    os.makedirs(cls_dir)
    cls_path = os.path.join(cls_dir, ONNX_FILENAME)
    wrapper = ClassifierWrapper(model)
    wrapper.eval()
    torch.onnx.export(
        wrapper, dummy, cls_path,
        input_names=["features"], output_names=["logits"],
        dynamic_axes={"features": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    onnx.checker.check_model(onnx.load(cls_path, load_external_data=True))
    logger.info(f"Exported classifier: {os.listdir(cls_dir)}")

    # ── Embedder ──
    emb_dir = os.path.join(export_dir, "embedder")
    os.makedirs(emb_dir)
    emb_path = os.path.join(emb_dir, ONNX_FILENAME)
    wrapper = EmbedderWrapper(model)
    wrapper.eval()
    torch.onnx.export(
        wrapper, dummy, emb_path,
        input_names=["features"], output_names=["embedding"],
        dynamic_axes={"features": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=17,
    )
    onnx.checker.check_model(onnx.load(emb_path, load_external_data=True))
    logger.info(f"Exported embedder: {os.listdir(emb_dir)}")

    return cls_dir, emb_dir


def compute_reference_distributions(model, features, labels):
    model.eval()
    x = torch.tensor(features, dtype=torch.float32)
    with torch.no_grad():
        logits, embeddings = model(x)
    embeddings_np = embeddings.numpy()
    logits_np = logits.numpy()

    feature_histograms = {}
    for i, name in enumerate(FEATURE_NAMES):
        col = features[:, i]
        counts, bin_edges = np.histogram(col, bins=50)
        feature_histograms[name] = {
            "bin_edges": bin_edges.tolist(),
            "counts": (counts / counts.sum()).tolist(),
            "mean": float(col.mean()),
            "std": float(col.std()),
        }

    class_gaussians = {}
    for cls in range(NUM_CLASSES):
        mask = labels == cls
        cls_emb = embeddings_np[mask]
        mean = cls_emb.mean(axis=0)
        cov = np.cov(cls_emb.T) + np.eye(EMBEDDING_DIM) * 1e-6
        class_gaussians[str(cls)] = {
            "mean": mean.tolist(),
            "precision": np.linalg.inv(cov).tolist(),
            "num_samples": int(mask.sum()),
        }

    preds = np.argmax(logits_np, axis=1)
    pred_freq = np.bincount(preds, minlength=NUM_CLASSES) / len(preds)

    return {
        "reference_distribution": {
            "num_samples": len(features),
            "feature_histograms": feature_histograms,
            "embedding_mean": embeddings_np.mean(axis=0).tolist(),
            "embedding_cov": np.cov(embeddings_np.T).tolist(),
            "prediction_class_frequencies": pred_freq.tolist(),
        },
        "class_gaussians": {"classes": class_gaussians},
    }


def main():
    pl.seed_everything(SEED, workers=True)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("ml_system_training")
    logger.info(f"MLflow: {MLFLOW_TRACKING_URI}")

    features, labels = generate_synthetic_data(5000, SEED)
    train_loader, val_loader = make_dataloaders(features, labels, SEED, BATCH_SIZE)
    logger.info(f"Dataset: {len(features)} samples, {NUM_CLASSES} classes, {INPUT_DIM} features")

    model = Classifier(input_dim=INPUT_DIM, embedding_dim=EMBEDDING_DIM, num_classes=NUM_CLASSES, lr=LR)

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        callbacks=[
            pl.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min"),
            pl.callbacks.ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1),
        ],
        enable_progress_bar=True,
        deterministic=True,
    )

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logger.info(f"Run ID: {run_id}")

        mlflow.log_params({
            "input_dim": INPUT_DIM, "embedding_dim": EMBEDDING_DIM,
            "num_classes": NUM_CLASSES, "lr": LR,
            "batch_size": BATCH_SIZE, "max_epochs": MAX_EPOCHS, "seed": SEED,
        })

        trainer.fit(model, train_loader, val_loader)

        val_metrics = trainer.callback_metrics
        mlflow.log_metrics({
            "val_loss": float(val_metrics.get("val_loss", 0)),
            "val_acc": float(val_metrics.get("val_acc", 0)),
            "val_f1": float(val_metrics.get("val_f1", 0)),
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            # ── Export ONNX ──
            cls_dir, emb_dir = export_onnx(model, tmpdir)
            mlflow.log_artifacts(cls_dir, MLFLOW_PATH_CLASSIFIER)
            mlflow.log_artifacts(emb_dir, MLFLOW_PATH_EMBEDDER)

            # ── Reference distributions ──
            refs = compute_reference_distributions(model, features, labels)

            ref_path = os.path.join(tmpdir, REFERENCE_DIST_FILENAME)
            with open(ref_path, "w") as f:
                json.dump(refs["reference_distribution"], f)
            mlflow.log_artifact(ref_path)

            gauss_path = os.path.join(tmpdir, CLASS_GAUSSIANS_FILENAME)
            with open(gauss_path, "w") as f:
                json.dump(refs["class_gaussians"], f)
            mlflow.log_artifact(gauss_path)

            schema_path = os.path.join(tmpdir, FEATURE_SCHEMA_FILENAME)
            with open(schema_path, "w") as f:
                json.dump(FEATURE_SCHEMA, f, indent=2)
            mlflow.log_artifact(schema_path)

        # ── Register model ──
        model_uri = f"runs:/{run_id}/{MLFLOW_PATH_CLASSIFIER}"
        result = mlflow.register_model(model_uri, MODEL_NAME)
        logger.info(f"Registered version {result.version}")

        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=MODEL_NAME, version=result.version,
            stage="Production", archive_existing_versions=True,
        )
        logger.info(f"Version {result.version} → Production")

    logger.info("Done.")


if __name__ == "__main__":
    main()
