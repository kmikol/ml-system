# shared/validation.py
from shared.schemas.feature_schema import FEATURE_SCHEMA


def validate_features(features: dict[str, float]) -> list[dict] | None:
    """Returns None if valid, list of error dicts if invalid."""
    errors = []

    for name in FEATURE_SCHEMA:
        if name not in features:
            errors.append({"field": name, "error": "missing required feature"})
    for name in features:
        if name not in FEATURE_SCHEMA:
            errors.append({"field": name, "error": "unknown feature"})

    if errors:
        return errors

    for name, spec in FEATURE_SCHEMA.items():
        value = features[name]
        if value is None:
            if not spec["nullable"]:
                errors.append({"field": name, "error": "null not allowed"})
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            errors.append({"field": name, "error": f"cannot convert to float"})
            continue
        if spec["min"] is not None and value < spec["min"]:
            errors.append({"field": name, "error": f"value {value} below minimum {spec['min']}"})
        if spec["max"] is not None and value > spec["max"]:
            errors.append({"field": name, "error": f"value {value} above maximum {spec['max']}"})

    return errors if errors else None
