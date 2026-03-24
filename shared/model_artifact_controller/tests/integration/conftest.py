# shared/model_artifact_controller/tests/integration/conftest.py
"""
Integration test fixtures for MLflowModelArtifactController.

These tests are fully self-contained: a temporary SQLite database is created
per session and cleaned up automatically. No external MLflow service or Docker
infrastructure is required — SQLite covers all behaviour tested here.
"""

import os
import tempfile
import uuid

import mlflow
import pytest

from shared.model_artifact_controller.mlflow import MLflowModelArtifactController


@pytest.fixture(scope="session")
def mlflow_tracking():
    """Point MLflow at a temporary SQLite database for the session.

    Overrides any MLFLOW_TRACKING_URI already in the environment (e.g. the
    docker-compose http://mlflow:5000 value) so that running these tests
    locally never writes to mlruns/ and never requires a running MLflow server.
    """
    with tempfile.TemporaryDirectory(prefix="mlflow_test_") as tmpdir:
        # file:// URI stores both tracking metadata and artifacts inside tmpdir,
        # so nothing is written to ./mlruns in the workspace.
        uri = f"file://{tmpdir}/mlruns"
        prev = os.environ.get("MLFLOW_TRACKING_URI")
        os.environ["MLFLOW_TRACKING_URI"] = uri
        mlflow.set_tracking_uri(uri)
        try:
            yield uri
        finally:
            if prev is None:
                del os.environ["MLFLOW_TRACKING_URI"]
            else:
                os.environ["MLFLOW_TRACKING_URI"] = prev


@pytest.fixture(scope="session")
def experiment_name(mlflow_tracking):
    """Create a dedicated MLflow experiment for the test session."""
    name = f"integration_test_{uuid.uuid4().hex[:8]}"
    mlflow.create_experiment(name)
    return name


@pytest.fixture
def ctrl(mlflow_tracking):
    """A fresh MLflowModelArtifactController for each test."""
    return MLflowModelArtifactController()
