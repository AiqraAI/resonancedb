import json
import numpy as np
import pandas as pd
from scripts.phyphox_to_resonancedb import phyphox_to_json


def test_phyphox_conversion(tmp_path):
    sr = 100
    t = np.arange(0, 1.01, 1 / sr)
    z = 9.81 + 0.981 * np.sin(2 * np.pi * 5.0 * t)  # 0.1 g sinusoid on top of gravity

    df = pd.DataFrame({
        "Time (s)": t,
        "Linear Acceleration z (m/s^2)": z,
    })
    csv_path = tmp_path / "raw.csv"
    df.to_csv(csv_path, index=False)

    out_json = tmp_path / "out.json"
    phyphox_to_json(str(csv_path), str(out_json), material="wood")

    data = json.loads(out_json.read_text())
    assert data["material"] == "wood"
    assert data["sample_rate_hz"] == sr
    assert data["excitation"] == "manual_tap"
    assert data["source"] == "phone_sensor"
    assert data.get("device") == "smartphone_phyphox"
    assert isinstance(data["vibration"], list) and len(data["vibration"]) > 0