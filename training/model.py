# training/model.py
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy, F1Score


class Classifier(pl.LightningModule):
    def __init__(
        self, input_dim: int, embedding_dim: int, num_classes: int, lr: float, hidden_dim: int = 256
    ):
        super().__init__()
        self.save_hyperparameters()

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
        )
        self.classifier_head = nn.Linear(embedding_dim, num_classes)

        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")

    def forward(self, x):
        embedding = self.feature_extractor(x)
        logits = self.classifier_head(embedding)
        return logits, embedding

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits, _ = self(x)
        loss = F.cross_entropy(logits, y)
        self.train_acc(torch.argmax(logits, dim=1), y)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits, _ = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.val_acc(preds, y)
        self.val_f1(preds, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_acc, prog_bar=True)
        self.log("val_f1", self.val_f1, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


class ClassifierWrapper(nn.Module):
    """For ONNX export: input → logits only."""

    def __init__(self, model: Classifier):
        super().__init__()
        self.feature_extractor = model.feature_extractor
        self.classifier_head = model.classifier_head

    def forward(self, x):
        return self.classifier_head(self.feature_extractor(x))


class EmbedderWrapper(nn.Module):
    """For ONNX export: input → embedding only."""

    def __init__(self, model: Classifier):
        super().__init__()
        self.feature_extractor = model.feature_extractor

    def forward(self, x):
        return self.feature_extractor(x)


class UnifiedWrapper(nn.Module):
    """For ONNX export: input → (logits, embedding) as tuple."""

    def __init__(self, model: Classifier):
        super().__init__()
        self.feature_extractor = model.feature_extractor
        self.classifier_head = model.classifier_head

    def forward(self, x):
        embedding = self.feature_extractor(x)
        logits = self.classifier_head(embedding)
        return logits, embedding
