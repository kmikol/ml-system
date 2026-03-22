import os
import tempfile

from shared.artifact_paths import (
    ONNX_FILENAME,
    resolve_classifier_path,
    resolve_embedder_path,
)


def test_resolve_classifier():
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "classifier"))
        path = os.path.join(d, "classifier", ONNX_FILENAME)
        open(path, "w").close()
        assert resolve_classifier_path(d) == path


def test_resolve_embedder():
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "embedder"))
        path = os.path.join(d, "embedder", ONNX_FILENAME)
        open(path, "w").close()
        assert resolve_embedder_path(d) == path


def test_resolve_missing_crashes():
    import pytest

    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "classifier"))
        # no model.onnx file
        with pytest.raises(FileNotFoundError):
            resolve_classifier_path(d)
