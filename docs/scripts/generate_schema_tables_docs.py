#!/usr/bin/env python3
"""Generate schema field tables for docs from source files.

Avoids hardcoding schema tables in markdown by extracting model fields and
field descriptions from Pydantic schema modules.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCHEMAS = ROOT / "shared" / "schemas"
OUT_DIR = ROOT / "docs" / "schemas" / "generated"


def _escape(text: str) -> str:
    return text.replace("|", "\\|")


def _unparse(node: ast.AST | None) -> str:
    if node is None:
        return ""
    try:
        return ast.unparse(node)
    except Exception:
        return ""


def _field_description_from_call(call: ast.Call) -> str:
    for kw in call.keywords:
        if kw.arg == "description" and isinstance(kw.value, ast.Constant):
            if isinstance(kw.value.value, str):
                return kw.value.value
    return ""


def _field_default_from_call(call: ast.Call) -> str:
    for kw in call.keywords:
        if kw.arg == "default":
            return _unparse(kw.value)
    if call.args:
        first = call.args[0]
        if isinstance(first, ast.Constant) and first.value is Ellipsis:
            return "required"
        return _unparse(first)
    return ""


def _is_basemodel_subclass(node: ast.ClassDef) -> bool:
    for base in node.bases:
        name = _unparse(base)
        if name.endswith("BaseModel"):
            return True
    return False


def _extract_model_tables(path: Path) -> list[tuple[str, list[tuple[str, str, str]]]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    result: list[tuple[str, list[tuple[str, str, str]]]] = []

    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        if not _is_basemodel_subclass(node):
            continue

        rows: list[tuple[str, str, str]] = []
        for item in node.body:
            if not isinstance(item, ast.AnnAssign):
                continue
            if not isinstance(item.target, ast.Name):
                continue

            field_name = item.target.id
            field_type = _unparse(item.annotation)
            description = ""

            if isinstance(item.value, ast.Call) and _unparse(item.value.func).endswith("Field"):
                description = _field_description_from_call(item.value)
                if not description:
                    description = "TODO: add field description"
            else:
                description = "TODO: add field description"

            rows.append((field_name, field_type, description))

        result.append((node.name, rows))

    return result


def _extract_constants_table(path: Path) -> list[tuple[str, str, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    rows: list[tuple[str, str, str]] = []

    default_descriptions = {
        "IMAGE_SIZE": "Input image dimensions used by the model pipeline.",
        "INPUT_DIM": "Flattened input size derived from IMAGE_SIZE.",
        "NUM_CLASSES": "Number of output digit classes.",
        "EMBEDDING_DIM": "Embedding vector dimension used in model outputs.",
    }

    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if not name.isupper():
                continue
            value = _unparse(node.value)
            desc = default_descriptions.get(name, "TODO: add constant description")
            rows.append((name, value, desc))

    return rows


def _render_model_tables(models: list[tuple[str, list[tuple[str, str, str]]]]) -> str:
    lines: list[str] = []
    for model_name, rows in models:
        lines.append(f"## {model_name}")
        lines.append("")
        lines.append("### Fields")
        lines.append("")
        lines.append("| Field | Type | Description |")
        lines.append("|------|------|-------------|")
        for field_name, field_type, desc in rows:
            lines.append(
                f"| {_escape(field_name)} | {_escape(field_type)} | {_escape(desc)} |"
            )
        lines.append("")
    return "\n".join(lines)


def _render_constants_table(rows: list[tuple[str, str, str]]) -> str:
    lines = [
        "## Feature Constants",
        "",
        "| Constant | Value | Description |",
        "|----------|-------|-------------|",
    ]
    for name, value, desc in rows:
        lines.append(f"| {_escape(name)} | {_escape(value)} | {_escape(desc)} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    api_models = _extract_model_tables(SCHEMAS / "api.py")
    (OUT_DIR / "api-fields.md").write_text(_render_model_tables(api_models), encoding="utf-8")

    record_models = _extract_model_tables(SCHEMAS / "predict_record.py")
    inference_models = _extract_model_tables(SCHEMAS / "inference_event.py")
    constants = _extract_constants_table(SCHEMAS / "feature_schema.py")
    records_body = _render_model_tables(record_models + inference_models)
    records_body += "\n" + _render_constants_table(constants)
    (OUT_DIR / "records-fields.md").write_text(records_body, encoding="utf-8")


if __name__ == "__main__":
    main()