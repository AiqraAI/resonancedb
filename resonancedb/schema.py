"""Validation of ResonanceDB sample files against the data format.

The rules here ARE the open data standard (docs/DATA_FORMAT.md); every
consumer, the CLI, CI checks on data PRs, and any hosted service, should
validate through this module rather than re-implementing the rules.
"""

import json
from pathlib import Path

REQUIRED_FIELDS = ["material", "vibration", "sample_rate_hz", "excitation", "source"]


def _is_number(x) -> bool:
    # bool is a subclass of int; JSON true/false must not count as numbers
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def validate_sample_dict(data: dict) -> list[str]:
    """Validate a parsed sample. Returns a list of errors (empty = valid)."""
    errors = []

    missing = [f for f in REQUIRED_FIELDS if f not in data]
    if missing:
        return [f"Missing fields: {missing}"]

    if not isinstance(data["material"], str) or not data["material"].strip():
        errors.append("'material' must be a non-empty string")

    vibration = data["vibration"]
    if not isinstance(vibration, (list, tuple)):
        errors.append("'vibration' must be a list of numbers")
    elif len(vibration) == 0:
        errors.append("'vibration' array is empty")
    elif not all(_is_number(x) for x in vibration):
        errors.append("'vibration' must contain only numbers")

    if not _is_number(data["sample_rate_hz"]):
        errors.append("'sample_rate_hz' must be a number")
    elif data["sample_rate_hz"] <= 0:
        errors.append("'sample_rate_hz' must be greater than 0")

    if not isinstance(data["excitation"], str) or not data["excitation"].strip():
        errors.append("'excitation' must be a non-empty string")

    if not isinstance(data["source"], str) or not data["source"].strip():
        errors.append("'source' must be a non-empty string")

    return errors


def validate_sample(file_path) -> bool:
    """Validate one JSON file, printing findings. Returns True if valid."""
    fp = Path(file_path)
    if not fp.exists():
        print(f"[FAIL] File not found: {file_path}")
        return False

    try:
        with fp.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"[FAIL] Invalid JSON in {file_path}: {e}")
        return False

    errors = validate_sample_dict(data)
    if errors:
        for err in errors:
            print(f"[FAIL] {fp.name}: {err}")
        return False

    print(f"[OK] {fp.name}: {data['material']} ({len(data['vibration'])} samples)")
    return True


def validate_tree(data_dir) -> tuple[int, int]:
    """Validate every .json under `data_dir`. Returns (valid_count, invalid_count)."""
    root = Path(data_dir)
    valid = invalid = 0
    for json_file in sorted(root.rglob("*.json")):
        if validate_sample(json_file):
            valid += 1
        else:
            invalid += 1
    return valid, invalid
