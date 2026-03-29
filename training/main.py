# training/main.py
"""
Training pipeline: load MNIST from DatasetController → train → export ONNX → register in MLflow.
Usage: python -m training.main
"""

import logging
import os
from pathlib import Path

import numpy as np
import onnx
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset

from shared.config import require_env
from shared.data_controller.dataset import DatasetController
from shared.logging_config import setup_logging
from shared.model_artifact_controller import ModelArtifactController
from shared.schemas.feature_schema import (
    EMBEDDING_DIM,
    INPUT_DIM,
    NUM_CLASSES,
)
from training.model import Classifier, ClassifierWrapper, EmbedderWrapper

setup_logging("training")
logger = logging.getLogger(__name__)

# ── All config from env, no defaults ─────────────────────────────
MODEL_NAME = require_env("MODEL_NAME")
MAX_EPOCHS = int(require_env("TRAINING_MAX_EPOCHS"))
SEED = int(require_env("TRAINING_SEED"))
BATCH_SIZE = int(require_env("TRAINING_BATCH_SIZE"))
LR = float(require_env("TRAINING_LR"))
AUTO_PROMOTE = os.environ.get("AUTO_PROMOTE", "true").lower() == "true"
ONNX_FILENAME = "model.onnx"


def make_dataloaders(train_samples, val_samples, batch_size):
    def to_tensor_loader(samples, shuffle):
        images = np.stack([np.array(s["image"]).flatten() for s in samples]).astype(np.float32)
        labels = np.array([s["label"] for s in samples], dtype=np.int64)
        x = torch.tensor(images)
        y = torch.tensor(labels)
        return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle)

    return to_tensor_loader(train_samples, True), to_tensor_loader(val_samples, False)


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
        wrapper,
        dummy,
        cls_path,
        input_names=["features"],
        output_names=["logits"],
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
        wrapper,
        dummy,
        emb_path,
        input_names=["features"],
        output_names=["embedding"],
        dynamic_axes={"features": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=17,
    )
    onnx.checker.check_model(onnx.load(emb_path, load_external_data=True))
    logger.info(f"Exported embedder: {os.listdir(emb_dir)}")

    return cls_dir, emb_dir


def compute_reference_distributions(model, train_samples):
    model.eval()
    images = np.stack([np.array(s["image"]).flatten() for s in train_samples]).astype(np.float32)
    labels = np.array([s["label"] for s in train_samples], dtype=np.int64)

    x = torch.tensor(images)
    with torch.no_grad():
        logits, embeddings = model(x)
    embeddings_np = embeddings.numpy()
    logits_np = logits.numpy()

    pixel_statistics = {
        "mean": float(images.mean()),
        "std": float(images.std()),
    }

    class_gaussians = {}
    for cls in range(NUM_CLASSES):
        mask = labels == cls
        cls_emb = embeddings_np[mask]
        if len(cls_emb) < 2:
            # np.cov requires at least 2 observations (ddof=1); skip classes with
            # insufficient samples rather than producing a NaN precision matrix.
            continue
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
            "num_samples": len(images),
            "pixel_statistics": pixel_statistics,
            "embedding_mean": embeddings_np.mean(axis=0).tolist(),
            "embedding_cov": np.cov(embeddings_np.T).tolist(),
            "prediction_class_frequencies": pred_freq.tolist(),
        },
        "class_gaussians": {"classes": class_gaussians},
    }


def main():
    pl.seed_everything(SEED, workers=True)

    logger.info("Loading dataset from DatasetController...")
    dataset_ctrl = DatasetController()
    version_id = dataset_ctrl.get_latest_version()
    if version_id is None:
        raise RuntimeError("No dataset version found. Run scripts/seed_dataset.py before training.")
    logger.info(f"Using dataset version: {version_id}")
    train_samples = dataset_ctrl.get_dataset_split(version_id, "train")
    val_samples = dataset_ctrl.get_dataset_split(version_id, "val")
    logger.info(
        f"Dataset: {len(train_samples)} train, {len(val_samples)} val, "
        f"{NUM_CLASSES} classes, {INPUT_DIM} input_dim"
    )

    if not train_samples or not val_samples:
        raise RuntimeError(
            "Dataset split is empty. Expected non-empty 'train' and 'val' splits in the "
            f"latest dataset version '{version_id}'. Run scripts/seed_dataset.py first."
        )

    train_loader, val_loader = make_dataloaders(train_samples, val_samples, BATCH_SIZE)

    model = Classifier(
        input_dim=INPUT_DIM, embedding_dim=EMBEDDING_DIM, num_classes=NUM_CLASSES, lr=LR
    )

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        callbacks=[
            pl.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min"),
            pl.callbacks.ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1),
        ],
        enable_progress_bar=True,
        deterministic=True,
    )

    controller = ModelArtifactController()
    with controller.start_run("ml_system_training") as run_id:
        logger.info(f"Run ID: {run_id}")

        controller.log_params(
            run_id,
            {
                "input_dim": INPUT_DIM,
                "embedding_dim": EMBEDDING_DIM,
                "num_classes": NUM_CLASSES,
                "lr": LR,
                "batch_size": BATCH_SIZE,
                "max_epochs": MAX_EPOCHS,
                "seed": SEED,
                "train_samples": len(train_samples),
                "val_samples": len(val_samples),
            },
        )

        trainer.fit(model, train_loader, val_loader)

        val_metrics = trainer.callback_metrics
        controller.log_metrics(
            run_id,
            {
                "val_loss": float(val_metrics.get("val_loss", 0)),
                "val_acc": float(val_metrics.get("val_acc", 0)),
                "val_f1": float(val_metrics.get("val_f1", 0)),
            },
        )

        # ── Export ONNX ──
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            cls_dir, emb_dir = export_onnx(model, tmpdir)

            # ── Reference distributions ──
            refs = compute_reference_distributions(model, train_samples)
            feature_schema = {
                "image_size": [14, 14],
                "num_classes": NUM_CLASSES,
                "input_dim": INPUT_DIM,
            }

            controller.log_training_outputs(
                run_id=run_id,
                classifier_dir=cls_dir,
                embedder_dir=emb_dir,
                reference_distribution=refs["reference_distribution"],
                class_gaussians=refs["class_gaussians"],
                feature_schema=feature_schema,
            )

        # ── Register model ──
        version = controller.register_model(run_id, MODEL_NAME)
        logger.info(f"Registered version {version}")
        if AUTO_PROMOTE:
            controller.promote_model(MODEL_NAME, version)
            logger.info(f"Version {version} → Production")
        else:
            logger.info(f"Version {version} registered, promotion deferred to evaluate step")

    run_id_output_path = os.environ.get("RUN_ID_OUTPUT_PATH", "")
    if run_id_output_path:
        os.makedirs(os.path.dirname(run_id_output_path) or ".", exist_ok=True)
        Path(run_id_output_path).write_text(run_id)
        logger.info(f"run_id written to {run_id_output_path}")

    logger.info("Done.")


if __name__ == "__main__":
    main()
