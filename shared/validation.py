# shared/validation.py
from __future__ import annotations

from shared.schemas.feature_schema import IMAGE_SIZE


def validate_image(image: object) -> list[dict] | None:
    """Returns None if valid, list of error dicts if invalid."""
    h, w = IMAGE_SIZE

    if not isinstance(image, list) or len(image) != h:
        return [{"field": "image", "error": f"must be a list of {h} rows"}]

    for i, row in enumerate(image):
        if not isinstance(row, list) or len(row) != w:
            return [{"field": f"image[{i}]", "error": f"must be a list of {w} values"}]
        for j, val in enumerate(row):
            try:
                v = float(val)
            except (TypeError, ValueError):
                return [{"field": f"image[{i}][{j}]", "error": "value is not a number"}]
            if v < 0.0 or v > 1.0:
                return [{"field": f"image[{i}][{j}]", "error": f"value {v} not in [0, 1]"}]

    return None
