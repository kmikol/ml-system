# shared/model_artifact_controller/tests/integration/conftest.py
"""
Integration test fixtures for MLflowModelArtifactController.

MLflow is provided by docker-compose.test.yml (service: mlflow, backed by
Postgres + MinIO). MLFLOW_TRACKING_URI is injected as an environment variable
by the compose file — MLflowModelArtifactController reads it on construction.
"""

import uuid

import mlflow
import pytest

from shared.model_artifact_controller.mlflow import MLflowModelArtifactController


@pytest.fixture(scope="session")
def experiment_name():
    """Create a dedicated MLflow experiment for the test session."""
    name = f"integration_test_{uuid.uuid4().hex[:8]}"
    mlflow.create_experiment(name)
    return name


@pytest.fixture
def ctrl():
    """A fresh MLflowModelArtifactController for each test."""
    return MLflowModelArtifactController()
