import json
from scripts.validate_data import validate_sample


def test_validate_good(tmp_path):
    fp = tmp_path / "valid.json"
    data = {
        "material": "wood",
        "vibration": [0.0, 0.1, -0.05, 0.02],
        "sample_rate_hz": 1000,
        "excitation": "manual_tap",
        "source": "simulation",
    }
    fp.write_text(json.dumps(data))
    assert validate_sample(str(fp)) is True


def test_validate_missing_fields(tmp_path):
    fp = tmp_path / "invalid_missing.json"
    data = {
        "material": "wood",
        "vibration": [0.0, 0.1],
        # missing sample_rate_hz
        "excitation": "manual_tap",
        "source": "simulation",
    }
    fp.write_text(json.dumps(data))
    assert validate_sample(str(fp)) is False


def test_validate_non_numeric_vibration(tmp_path):
    fp = tmp_path / "invalid_vibration.json"
    data = {
        "material": "wood",
        "vibration": ["a", "b"],
        "sample_rate_hz": 1000,
        "excitation": "manual_tap",
        "source": "simulation",
    }
    fp.write_text(json.dumps(data))
    assert validate_sample(str(fp)) is False