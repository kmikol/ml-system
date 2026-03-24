# shared/logging_config.py
"""Centralised logging setup.

Call ``setup_logging(service)`` once at process start.

- Always attaches a StreamHandler (stdout) with a human-readable format.
- When ``LOKI_URL`` is set, also attaches a LokiHandler that ships logs to
  Grafana Loki. The ``service`` label identifies the origin in Loki queries.
- ``LOG_LEVEL`` env var controls the root level (default: INFO).
"""

import logging
import os


def setup_logging(service: str) -> None:
    """Configure the root logger for *service*.

    Args:
        service: Short name used as the ``service`` label in Loki and as the
                 log-record extra field (e.g. ``"serving"``, ``"training"``).
    """
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    fmt = logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(level)

    # Console handler — always present; add only if root logger has no handlers yet
    # (avoids duplicate output when called more than once, e.g. in tests)
    if not root.handlers:
        console = logging.StreamHandler()
        console.setFormatter(fmt)
        root.addHandler(console)

    # Loki handler — only when LOKI_URL is configured
    loki_url = os.getenv("LOKI_URL", "").rstrip("/")
    if loki_url:
        try:
            import logging_loki  # python-logging-loki

            loki_handler = logging_loki.LokiHandler(
                url=f"{loki_url}/loki/api/v1/push",
                tags={"service": service},
                version="1",
            )
            root.addHandler(loki_handler)
            logging.getLogger(__name__).info(
                "Loki handler configured: %s (service=%s)", loki_url, service
            )
        except ImportError:
            logging.getLogger(__name__).warning(
                "LOKI_URL is set but 'python-logging-loki' is not installed; "
                "Loki handler skipped."
            )
