# shared/data_controller/tests/integration/conftest.py
"""
Pytest fixtures for data controller integration tests.

Provides a raw psycopg2 connection for test-setup helpers that need to insert
rows directly without going through a controller.
"""

from __future__ import annotations

import os

import psycopg2
import psycopg2.extras
import pytest


@pytest.fixture
def db_conn():
    """Yield a direct psycopg2 connection for test-setup SQL.

    UUID adaptation is registered so that uuid.UUID objects round-trip
    correctly through queries in test helpers.
    """
    conn = psycopg2.connect(os.environ["DATA_CONTROLLER_DB_URL"])
    psycopg2.extras.register_uuid(conn)
    yield conn
    conn.close()
