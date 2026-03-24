# shared/model_artifact_controller/tests/integration/test_mlflow.py
"""
Integration tests for MLflowModelArtifactController against a local SQLite-backed
MLflow instance. No Docker or external services required.

The `mlflow_tracking` and `experiment_name` fixtures are provided by conftest.py.
"""

from __future__ import annotations

import json
import tempfile
import uuid
from pathlib import Path

import mlflow
import pytest

from shared.artifact_paths import MLFLOW_PATH_CLASSIFIER
from shared.model_artifact_controller import ModelArtifactError
from shared.model_artifact_controller.mlflow import MLflowModelArtifactController


class _DummyModel(mlflow.pyfunc.PythonModel):
    """Minimal pyfunc model used to create a valid MLmodel manifest."""

    def predict(self, context, model_input, params=None):
        return model_input


# ── start_run ─────────────────────────────────────────────────────────────────


class TestStartRun:
    def test_yields_a_valid_run_id(self, ctrl, experiment_name):
        with ctrl.start_run(experiment_name) as run_id:
            assert isinstance(run_id, str)
            assert len(run_id) > 0

    def test_creates_experiment_if_absent(self, ctrl):
        name = f"auto_created_{uuid.uuid4().hex[:8]}"
        with ctrl.start_run(name) as run_id:
            pass
        client = mlflow.tracking.MlflowClient()
        assert client.get_experiment_by_name(name) is not None

    def test_run_is_finished_after_context_exit(self, ctrl, experiment_name):
        with ctrl.start_run(experiment_name) as run_id:
            pass
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(run_id)
        assert run.info.status == "FINISHED"

    def test_run_is_failed_on_exception(self, ctrl, experiment_name):
        # start_run wraps non-ModelArtifactError exceptions, so capture run_id first
        captured = []
        with pytest.raises(ModelArtifactError):
            with ctrl.start_run(experiment_name) as run_id:
                captured.append(run_id)
                raise RuntimeError("deliberate failure")
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(captured[0])
        assert run.info.status == "FAILED"


# ── log_params ────────────────────────────────────────────────────────────────


class TestLogParams:
    def test_params_are_readable_from_mlflow(self, ctrl, experiment_name):
        with ctrl.start_run(experiment_name) as run_id:
            ctrl.log_params(run_id, {"lr": 0.001, "epochs": 10})

        client = mlflow.tracking.MlflowClient()
        params = client.get_run(run_id).data.params  # already a dict[str, str]
        assert params["lr"] == "0.001"
        assert params["epochs"] == "10"


# ── log_metrics ───────────────────────────────────────────────────────────────


class TestLogMetrics:
    def test_metrics_are_readable_from_mlflow(self, ctrl, experiment_name):
        with ctrl.start_run(experiment_name) as run_id:
            ctrl.log_metrics(run_id, {"val_loss": 0.42, "val_acc": 0.91})

        client = mlflow.tracking.MlflowClient()
        metrics = client.get_run(run_id).data.metrics
        assert metrics["val_loss"] == pytest.approx(0.42)
        assert metrics["val_acc"] == pytest.approx(0.91)


# ── log_artifact / log_artifacts ──────────────────────────────────────────────


class TestLogArtifact:
    def test_single_artifact_is_downloadable(self, ctrl, experiment_name):
        with ctrl.start_run(experiment_name) as run_id:
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
                f.write(json.dumps({"key": "value"}).encode())
                local_path = f.name
            ctrl.log_artifact(run_id, local_path, artifact_path="metadata")

        with tempfile.TemporaryDirectory() as dst:
            local = ctrl.download_artifacts(run_id, "metadata", dst)
            files = list(Path(local).iterdir())
            assert len(files) == 1
            assert json.loads(files[0].read_text()) == {"key": "value"}


class TestLogArtifacts:
    def test_directory_artifacts_are_downloadable(self, ctrl, experiment_name):
        with tempfile.TemporaryDirectory() as src_dir:
            (Path(src_dir) / "a.json").write_text('{"a": 1}')
            (Path(src_dir) / "b.json").write_text('{"b": 2}')

            with ctrl.start_run(experiment_name) as run_id:
                ctrl.log_artifacts(run_id, src_dir, artifact_path="config")

        with tempfile.TemporaryDirectory() as dst:
            local = ctrl.download_artifacts(run_id, "config", dst)
            names = {f.name for f in Path(local).iterdir()}
            assert names == {"a.json", "b.json"}


# ── register_model / promote_model / get_production_run_id ───────────────────


class TestModelRegistry:
    """Tests the full model registry lifecycle: register → promote → lookup."""

    @pytest.fixture
    def registered_model(self, ctrl, experiment_name):
        """Log a pyfunc model at the classifier path, register it, and return (run_id, version)."""
        model_name = f"test_model_{uuid.uuid4().hex[:8]}"

        with ctrl.start_run(experiment_name) as run_id:
            # Use pyfunc.log_model so MLflow writes an MLmodel manifest, which
            # is required by register_model in MLflow >= 2.14.
            mlflow.pyfunc.log_model(
                artifact_path=MLFLOW_PATH_CLASSIFIER,
                python_model=_DummyModel(),
            )

        version = ctrl.register_model(run_id, model_name)
        return model_name, run_id, version

    def test_register_model_returns_version_string(self, registered_model):
        _, _, version = registered_model
        assert isinstance(version, str)
        assert version.isdigit()

    def test_promote_model_sets_production_alias(self, ctrl, registered_model):
        model_name, run_id, version = registered_model
        ctrl.promote_model(model_name, version)

        client = mlflow.tracking.MlflowClient()
        mv = client.get_model_version_by_alias(model_name, "Production")
        assert str(mv.version) == version

    def test_get_production_run_id_returns_correct_run(self, ctrl, registered_model):
        model_name, run_id, version = registered_model
        ctrl.promote_model(model_name, version)

        found_run_id = ctrl.get_production_run_id(model_name, "Production")
        assert found_run_id == run_id

    def test_get_production_run_id_raises_when_none_promoted(self, ctrl, experiment_name):
        model_name = f"unregistered_{uuid.uuid4().hex[:8]}"
        with pytest.raises(ModelArtifactError, match=model_name):
            ctrl.get_production_run_id(model_name, "Production")


# ── download_artifacts ────────────────────────────────────────────────────────


class TestDownloadArtifacts:
    def test_downloaded_content_matches_uploaded(self, ctrl, experiment_name):
        payload = {"model": "test", "version": 1}

        with ctrl.start_run(experiment_name) as run_id:
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
                json.dump(payload, f)
                local_path = f.name
            ctrl.log_artifact(run_id, local_path, artifact_path="outputs")

        with tempfile.TemporaryDirectory() as dst:
            local = ctrl.download_artifacts(run_id, "outputs", dst)
            files = list(Path(local).iterdir())
            assert len(files) == 1
            assert json.loads(files[0].read_text()) == payload

    def test_download_nonexistent_artifact_raises(self, ctrl, experiment_name):
        with ctrl.start_run(experiment_name) as run_id:
            pass
        with pytest.raises(ModelArtifactError):
            with tempfile.TemporaryDirectory() as dst:
                ctrl.download_artifacts(run_id, "does_not_exist", dst)
