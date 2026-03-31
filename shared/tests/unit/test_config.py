# shared/tests/unit/test_config.py
"""Unit tests for shared/config.py::require_env."""

from __future__ import annotations

import pytest

from shared.config import require_env


class TestRequireEnv:
    def test_returns_value_when_set(self, monkeypatch):
        monkeypatch.setenv("TEST_VAR_CONFIG", "hello")
        assert require_env("TEST_VAR_CONFIG") == "hello"

    def test_exits_when_missing(self, monkeypatch):
        monkeypatch.delenv("TEST_VAR_CONFIG", raising=False)
        with pytest.raises(SystemExit) as exc_info:
            require_env("TEST_VAR_CONFIG")
        assert exc_info.value.code == 1

    def test_does_not_modify_env(self, monkeypatch):
        monkeypatch.setenv("TEST_VAR_CONFIG", "val")
        require_env("TEST_VAR_CONFIG")
        import os

        assert os.environ.get("TEST_VAR_CONFIG") == "val"

    def test_returns_string_type(self, monkeypatch):
        monkeypatch.setenv("TEST_VAR_CONFIG", "123")
        result = require_env("TEST_VAR_CONFIG")
        assert isinstance(result, str)
        assert result == "123"

    def test_empty_string_is_not_missing(self, monkeypatch):
        # An empty string is a valid value — only None triggers exit
        monkeypatch.setenv("TEST_VAR_CONFIG", "")
        result = require_env("TEST_VAR_CONFIG")
        assert result == ""
