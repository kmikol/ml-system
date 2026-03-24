# shared/config.py
"""
Strict environment variable reader.
No defaults anywhere. Missing = crash with a clear message.
"""

import logging
import os
import sys

logger = logging.getLogger(__name__)


def require_env(name: str) -> str:
    """Read an environment variable or crash immediately."""
    value = os.environ.get(name)
    if value is None:
        logger.critical(
            "Required environment variable '%s' is not set. "
            "Set it in .env or pass it to the container.",
            name,
        )
        sys.exit(1)
    return value
