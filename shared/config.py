# shared/config.py
"""
Strict environment variable reader.
No defaults anywhere. Missing = crash with a clear message.
"""

import os
import sys


def require_env(name: str) -> str:
    """Read an environment variable or crash immediately."""
    value = os.environ.get(name)
    if value is None:
        print(
            f"\n  FATAL: Required environment variable '{name}' is not set.\n"
            f"         Set it in .env or pass it to the container.\n",
            file=sys.stderr,
        )
        sys.exit(1)
    return value
