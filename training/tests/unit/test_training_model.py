# training/tests/unit/test_training_model.py
"""
Unit tests for training/model.py.

No env vars, DB, or MLflow needed — only PyTorch.
"""

from __future__ import annotations

import pytest
import torch

from training.model import Classifier, ClassifierWrapper, EmbedderWrapper, UnifiedWrapper

INPUT_DIM = 196
EMBEDDING_DIM = 64
NUM_CLASSES = 10
LR = 1e-3


@pytest.fixture
def model() -> Classifier:
    return Classifier(
        input_dim=INPUT_DIM,
        embedding_dim=EMBEDDING_DIM,
        num_classes=NUM_CLASSES,
        lr=LR,
    )


# ── Classifier.forward ────────────────────────────────────────────────────────


class TestClassifierForward:
    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    def test_output_shapes(self, model, batch_size):
        # BatchNorm1d requires > 1 sample in training mode; use eval for shape tests.
        model.eval()
        with torch.no_grad():
            x = torch.randn(batch_size, INPUT_DIM)
            logits, embedding = model(x)
        assert logits.shape == (batch_size, NUM_CLASSES)
        assert embedding.shape == (batch_size, EMBEDDING_DIM)

    def test_logits_and_embedding_are_distinct(self, model):
        model.eval()
        x = torch.randn(4, INPUT_DIM)
        with torch.no_grad():
            logits, embedding = model(x)
        assert logits.shape != embedding.shape

    def test_output_is_float32(self, model):
        model.eval()
        x = torch.randn(2, INPUT_DIM)
        with torch.no_grad():
            logits, embedding = model(x)
        assert logits.dtype == torch.float32
        assert embedding.dtype == torch.float32

    def test_deterministic_in_eval_mode(self, model):
        model.eval()
        x = torch.randn(4, INPUT_DIM)
        with torch.no_grad():
            logits1, emb1 = model(x)
            logits2, emb2 = model(x)
        assert torch.allclose(logits1, logits2)
        assert torch.allclose(emb1, emb2)


# ── Classifier.training_step ──────────────────────────────────────────────────


class TestClassifierTrainingStep:
    def test_returns_scalar_loss(self, model):
        x = torch.randn(8, INPUT_DIM)
        y = torch.randint(0, NUM_CLASSES, (8,))
        loss = model.training_step((x, y), batch_idx=0)
        assert loss.ndim == 0
        assert loss.item() > 0

    def test_loss_is_differentiable(self, model):
        x = torch.randn(4, INPUT_DIM)
        y = torch.randint(0, NUM_CLASSES, (4,))
        loss = model.training_step((x, y), batch_idx=0)
        loss.backward()
        # Gradients should exist on at least one parameter
        has_grad = any(p.grad is not None for p in model.parameters())
        assert has_grad


# ── Classifier.validation_step ────────────────────────────────────────────────


class TestClassifierValidationStep:
    def test_returns_scalar_loss(self, model):
        x = torch.randn(8, INPUT_DIM)
        y = torch.randint(0, NUM_CLASSES, (8,))
        loss = model.validation_step((x, y), batch_idx=0)
        assert loss.ndim == 0
        assert loss.item() > 0

    def test_does_not_raise(self, model):
        x = torch.randn(4, INPUT_DIM)
        y = torch.randint(0, NUM_CLASSES, (4,))
        # Should complete without error
        model.validation_step((x, y), batch_idx=0)


# ── Classifier.configure_optimizers ──────────────────────────────────────────


class TestClassifierConfigureOptimizers:
    def test_returns_adam_optimizer(self, model):
        opt = model.configure_optimizers()
        assert isinstance(opt, torch.optim.Adam)

    def test_optimizer_has_model_parameters(self, model):
        opt = model.configure_optimizers()
        param_ids = {id(p) for p in model.parameters()}
        opt_param_ids = {id(p) for pg in opt.param_groups for p in pg["params"]}
        assert opt_param_ids == param_ids

    def test_learning_rate_matches_hparam(self, model):
        opt = model.configure_optimizers()
        lr = opt.param_groups[0]["lr"]
        assert lr == pytest.approx(LR)


# ── ClassifierWrapper ─────────────────────────────────────────────────────────


class TestClassifierWrapper:
    @pytest.fixture
    def wrapper(self, model) -> ClassifierWrapper:
        model.eval()
        return ClassifierWrapper(model)

    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    def test_output_shape(self, wrapper, batch_size):
        with torch.no_grad():
            x = torch.randn(batch_size, INPUT_DIM)
            out = wrapper(x)
        assert out.shape == (batch_size, NUM_CLASSES)

    def test_output_matches_classifier_logits(self, model, wrapper):
        model.eval()
        x = torch.randn(4, INPUT_DIM)
        with torch.no_grad():
            logits_direct, _ = model(x)
            logits_wrapper = wrapper(x)
        assert torch.allclose(logits_direct, logits_wrapper)

    def test_returns_single_tensor(self, wrapper):
        x = torch.randn(2, INPUT_DIM)
        out = wrapper(x)
        assert isinstance(out, torch.Tensor)


# ── EmbedderWrapper ───────────────────────────────────────────────────────────


class TestEmbedderWrapper:
    @pytest.fixture
    def wrapper(self, model) -> EmbedderWrapper:
        model.eval()
        return EmbedderWrapper(model)

    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    def test_output_shape(self, wrapper, batch_size):
        with torch.no_grad():
            x = torch.randn(batch_size, INPUT_DIM)
            out = wrapper(x)
        assert out.shape == (batch_size, EMBEDDING_DIM)

    def test_output_matches_classifier_embedding(self, model, wrapper):
        model.eval()
        x = torch.randn(4, INPUT_DIM)
        with torch.no_grad():
            _, embedding_direct = model(x)
            embedding_wrapper = wrapper(x)
        assert torch.allclose(embedding_direct, embedding_wrapper)

    def test_returns_single_tensor(self, wrapper):
        x = torch.randn(2, INPUT_DIM)
        out = wrapper(x)
        assert isinstance(out, torch.Tensor)


# ── UnifiedWrapper ────────────────────────────────────────────────────────────


class TestUnifiedWrapper:
    @pytest.fixture
    def wrapper(self, model) -> UnifiedWrapper:
        model.eval()
        return UnifiedWrapper(model)

    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    def test_output_shapes(self, wrapper, batch_size):
        with torch.no_grad():
            x = torch.randn(batch_size, INPUT_DIM)
            logits, embedding = wrapper(x)
        assert logits.shape == (batch_size, NUM_CLASSES)
        assert embedding.shape == (batch_size, EMBEDDING_DIM)

    def test_output_matches_classifier(self, model, wrapper):
        model.eval()
        x = torch.randn(4, INPUT_DIM)
        with torch.no_grad():
            logits_direct, embedding_direct = model(x)
            logits_wrapper, embedding_wrapper = wrapper(x)
        assert torch.allclose(logits_direct, logits_wrapper)
        assert torch.allclose(embedding_direct, embedding_wrapper)

    def test_returns_two_tensors(self, wrapper):
        x = torch.randn(2, INPUT_DIM)
        logits, embedding = wrapper(x)
        assert isinstance(logits, torch.Tensor)
        assert isinstance(embedding, torch.Tensor)
