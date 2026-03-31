"""Unit tests for MLflowModelArtifactController.

All MLflow calls are mocked — no running MLflow server required.
The fixture replaces ctrl._mlflow and ctrl._client with MagicMocks after
construction, so each test starts with a clean, controllable state.
"""

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from shared.model_artifact_controller import ModelArtifactError
from shared.model_artifact_controller.mlflow import MLflowModelArtifactController

_MLFLOW_PATH_MODEL = "onnx/model"


@pytest.fixture
def ctrl(monkeypatch):
    """Construct controller with mocked mlflow, then replace internals with clean mocks."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    with patch("mlflow.set_tracking_uri"), patch("mlflow.tracking.MlflowClient"):
        instance = MLflowModelArtifactController()
    instance._mlflow = MagicMock()
    instance._client = MagicMock()
    return instance


# ── start_run ─────────────────────────────────────────────────────────────────


class TestStartRun:
    def test_yields_run_id(self, ctrl):
        mock_run = MagicMock()
        mock_run.info.run_id = "abc-123"

        @contextmanager
        def fake_start_run():
            yield mock_run

        ctrl._mlflow.start_run = fake_start_run

        with ctrl.start_run("my_experiment") as run_id:
            assert run_id == "abc-123"

    def test_sets_experiment_name(self, ctrl):
        mock_run = MagicMock()
        mock_run.info.run_id = "abc-123"

        @contextmanager
        def fake_start_run():
            yield mock_run

        ctrl._mlflow.start_run = fake_start_run

        with ctrl.start_run("my_experiment"):
            pass

        ctrl._mlflow.set_experiment.assert_called_once_with("my_experiment")

    def test_wraps_set_experiment_error(self, ctrl):
        ctrl._mlflow.set_experiment.side_effect = Exception("connection refused")

        with (
            pytest.raises(ModelArtifactError, match="my_experiment"),
            ctrl.start_run("my_experiment"),
        ):
            pass

    def test_wraps_start_run_error(self, ctrl):
        ctrl._mlflow.set_experiment.return_value = None
        ctrl._mlflow.start_run.side_effect = Exception("server error")

        with pytest.raises(ModelArtifactError), ctrl.start_run("my_experiment"):
            pass

    def test_does_not_suppress_model_artifact_error_from_body(self, ctrl):
        mock_run = MagicMock()
        mock_run.info.run_id = "abc-123"

        @contextmanager
        def fake_start_run():
            yield mock_run

        ctrl._mlflow.start_run = fake_start_run

        with pytest.raises(ModelArtifactError, match="inner error"), ctrl.start_run("exp"):
            raise ModelArtifactError("inner error")


# ── log_params ────────────────────────────────────────────────────────────────


class TestLogParams:
    def test_calls_log_param_for_each_entry(self, ctrl):
        ctrl.log_params("run-1", {"lr": 0.001, "batch_size": 256})

        assert ctrl._client.log_param.call_count == 2
        ctrl._client.log_param.assert_any_call("run-1", "lr", "0.001")
        ctrl._client.log_param.assert_any_call("run-1", "batch_size", "256")

    def test_empty_params_calls_nothing(self, ctrl):
        ctrl.log_params("run-1", {})
        ctrl._client.log_param.assert_not_called()

    def test_wraps_client_error(self, ctrl):
        ctrl._client.log_param.side_effect = Exception("network error")

        with pytest.raises(ModelArtifactError, match="run-1"):
            ctrl.log_params("run-1", {"lr": 0.001})


# ── log_metrics ───────────────────────────────────────────────────────────────


class TestLogMetrics:
    def test_calls_log_metric_for_each_entry(self, ctrl):
        ctrl.log_metrics("run-2", {"val_loss": 0.42, "val_acc": 0.91})

        assert ctrl._client.log_metric.call_count == 2
        ctrl._client.log_metric.assert_any_call("run-2", "val_loss", 0.42)
        ctrl._client.log_metric.assert_any_call("run-2", "val_acc", 0.91)

    def test_wraps_client_error(self, ctrl):
        ctrl._client.log_metric.side_effect = Exception("timeout")

        with pytest.raises(ModelArtifactError, match="run-2"):
            ctrl.log_metrics("run-2", {"val_loss": 0.5})


# ── log_artifact ──────────────────────────────────────────────────────────────


class TestLogArtifact:
    def test_calls_client_with_all_args(self, ctrl):
        ctrl.log_artifact("run-3", "/tmp/model.onnx", "onnx/classifier")
        ctrl._client.log_artifact.assert_called_once_with(
            "run-3", "/tmp/model.onnx", "onnx/classifier"
        )

    def test_artifact_path_defaults_to_none(self, ctrl):
        ctrl.log_artifact("run-3", "/tmp/schema.json")
        ctrl._client.log_artifact.assert_called_once_with("run-3", "/tmp/schema.json", None)

    def test_wraps_client_error(self, ctrl):
        ctrl._client.log_artifact.side_effect = Exception("S3 unavailable")

        with pytest.raises(ModelArtifactError, match="model.onnx"):
            ctrl.log_artifact("run-3", "/tmp/model.onnx")


# ── log_artifacts ─────────────────────────────────────────────────────────────


class TestLogArtifacts:
    def test_calls_client_with_all_args(self, ctrl):
        ctrl.log_artifacts("run-4", "/tmp/cls_dir", "onnx/classifier")
        ctrl._client.log_artifacts.assert_called_once_with(
            "run-4", "/tmp/cls_dir", "onnx/classifier"
        )

    def test_wraps_client_error(self, ctrl):
        ctrl._client.log_artifacts.side_effect = Exception("upload failed")

        with pytest.raises(ModelArtifactError, match="cls_dir"):
            ctrl.log_artifacts("run-4", "/tmp/cls_dir")


# ── register_model ────────────────────────────────────────────────────────────


class TestRegisterModel:
    def test_returns_version_string(self, ctrl):
        result = MagicMock()
        result.version = 3
        ctrl._mlflow.register_model.return_value = result

        version = ctrl.register_model("run-5", "my_model")

        assert version == "3"

    def test_uses_model_path_in_uri(self, ctrl):
        result = MagicMock()
        result.version = 1
        ctrl._mlflow.register_model.return_value = result

        ctrl.register_model("run-5", "my_model")

        expected_uri = f"runs:/run-5/{_MLFLOW_PATH_MODEL}"
        ctrl._mlflow.register_model.assert_called_once_with(expected_uri, "my_model")

    def test_wraps_client_error(self, ctrl):
        ctrl._mlflow.register_model.side_effect = Exception("registry unavailable")

        with pytest.raises(ModelArtifactError, match="my_model"):
            ctrl.register_model("run-5", "my_model")


# ── promote_model ─────────────────────────────────────────────────────────────


class TestPromoteModel:
    def test_sets_production_alias(self, ctrl):
        ctrl.promote_model("my_model", "3")

        ctrl._client.set_registered_model_alias.assert_called_once_with(
            name="my_model",
            alias="Production",
            version="3",
        )

    def test_wraps_client_error(self, ctrl):
        ctrl._client.set_registered_model_alias.side_effect = Exception("permission denied")

        with pytest.raises(ModelArtifactError, match="my_model"):
            ctrl.promote_model("my_model", "3")


# ── get_production_run_id ─────────────────────────────────────────────────────


class TestGetProductionRunId:
    def test_returns_run_id_for_alias(self, ctrl):
        mv = MagicMock()
        mv.run_id = "prod-run-99"
        ctrl._client.get_model_version_by_alias.return_value = mv

        result = ctrl.get_production_run_id("my_model", "Production")

        assert result == "prod-run-99"
        ctrl._client.get_model_version_by_alias.assert_called_once_with(
            name="my_model", alias="Production"
        )

    def test_raises_when_alias_not_found(self, ctrl):
        ctrl._client.get_model_version_by_alias.side_effect = Exception(
            "RESOURCE_DOES_NOT_EXIST: Registered model alias Production not found"
        )

        with pytest.raises(ModelArtifactError, match="my_model"):
            ctrl.get_production_run_id("my_model", "Production")

    def test_wraps_client_error(self, ctrl):
        ctrl._client.get_model_version_by_alias.side_effect = Exception("DB timeout")

        with pytest.raises(ModelArtifactError, match="my_model"):
            ctrl.get_production_run_id("my_model", "Production")


# ── download_artifacts ────────────────────────────────────────────────────────


class TestDownloadArtifacts:
    def test_returns_local_path(self, ctrl):
        ctrl._client.download_artifacts.return_value = "/tmp/ml_model_xyz/onnx"

        result = ctrl.download_artifacts("run-6", "onnx", "/tmp/ml_model_xyz")

        assert result == "/tmp/ml_model_xyz/onnx"
        ctrl._client.download_artifacts.assert_called_once_with(
            "run-6", "onnx", "/tmp/ml_model_xyz"
        )

    def test_wraps_client_error(self, ctrl):
        ctrl._client.download_artifacts.side_effect = Exception("S3 key not found")

        with pytest.raises(ModelArtifactError, match="run-6"):
            ctrl.download_artifacts("run-6", "onnx", "/tmp/ml_model_xyz")


# ── get_run_metrics ──────────────────────────────────────────────────────────


class TestGetRunMetrics:
    def test_returns_metrics_dict(self, ctrl):
        mock_run = MagicMock()
        mock_run.data.metrics = {"val_acc": 0.95, "val_loss": 0.05}
        ctrl._client.get_run.return_value = mock_run

        result = ctrl.get_run_metrics("run-7")

        assert result == {"val_acc": 0.95, "val_loss": 0.05}
        ctrl._client.get_run.assert_called_once_with("run-7")

    def test_returns_empty_dict_when_no_metrics(self, ctrl):
        mock_run = MagicMock()
        mock_run.data.metrics = {}
        ctrl._client.get_run.return_value = mock_run

        result = ctrl.get_run_metrics("run-7")
        assert result == {}

    def test_wraps_client_error(self, ctrl):
        ctrl._client.get_run.side_effect = Exception("run not found")

        with pytest.raises(ModelArtifactError, match="run-7"):
            ctrl.get_run_metrics("run-7")


# ── search_version_by_run ────────────────────────────────────────────────────


class TestSearchVersionByRun:
    def test_returns_version_when_found(self, ctrl):
        mock_version = MagicMock()
        mock_version.version = "5"
        ctrl._client.search_model_versions.return_value = [mock_version]

        result = ctrl.search_version_by_run("my_model", "run-8")

        assert result == "5"
        ctrl._client.search_model_versions.assert_called_once_with(
            "name='my_model' and run_id='run-8'"
        )

    def test_returns_none_when_not_found(self, ctrl):
        ctrl._client.search_model_versions.return_value = []

        result = ctrl.search_version_by_run("my_model", "run-8")
        assert result is None

    def test_wraps_client_error(self, ctrl):
        ctrl._client.search_model_versions.side_effect = Exception("DB error")

        with pytest.raises(ModelArtifactError, match="run-8"):
            ctrl.search_version_by_run("my_model", "run-8")
