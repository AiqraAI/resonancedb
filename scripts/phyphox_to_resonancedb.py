# scripts/phyphox_to_resonancedb.py
#
# LEGACY: phone accelerometer capture via phyphox is superseded by the
# microphone-based capture path (see ROADMAP.md Phase 2) — phone browsers
# cap motion sensors at ~60 Hz while the mic records at 44.1+ kHz.
# Kept for contributors who specifically want contact-vibration data.
#
# KNOWN ISSUE: phyphox "Linear Acceleration" already has gravity removed;
# the `- 1.0` subtraction below is wrong and will be fixed or removed when
# this path is revisited. Do not rely on this converter for real datasets.
import pandas as pd
import json
import numpy as np
import sys
from pathlib import Path

def phyphox_to_json(csv_file, output_json, material, notes=""):
    # Load CSV
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"❌ Could not read CSV: {e}")
        return

    # Debug: Show columns
    print("📋 Available columns:")
    print(df.columns.tolist())

    # Define expected column name
    z_col = "Linear Acceleration z (m/s^2)"
    time_col = "Time (s)"

    if z_col not in df.columns:
        print(f"❌ Column '{z_col}' not found!")
        print("💡 Your file may use different naming. Check above.")
        return

    if time_col not in df.columns:
        print(f"❌ Column '{time_col}' not found!")
        return

    # Extract data
    time = df[time_col].values
    acc_z_ms2 = df[z_col].values
    acc_z_g = acc_z_ms2 / 9.81  # Convert to g
    vibration = acc_z_g - 1.0   # Remove gravity

    # Trim around the main event (tap)
    # Find the largest spike in absolute value
    center_idx = np.argmax(np.abs(vibration))
    window = 500  # ~0.5 sec before/after
    start = max(0, center_idx - window)
    end = min(len(vibration), center_idx + window)
    trimmed = vibration[start:end]

    # Estimate sample rate
    if len(time) > 1:
        sample_rate = int(1 / np.mean(np.diff(time[start:end])))
    else:
        sample_rate = 100  # fallback

    # Create JSON
    data = {
        "material": material,
        "vibration": trimmed.tolist(),
        "sample_rate_hz": sample_rate,
        "excitation": "manual_tap",
        "source": "phone_sensor",
        "device": "smartphone_phyphox",
        "notes": notes or f"Converted from Phyphox: Linear Acceleration z (m/s^2)"
    }

    # Save
    with open(output_json, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"✅ Saved to {output_json}")
    print(f"📊 Samples: {len(trimmed)}, Sample Rate: {sample_rate} Hz")
    print(f"📍 Trimmed around index {center_idx} (peak vibration)")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python phyphox_to_resonancedb.py <input.csv> <material> <output.json>")
        print("Example: python phyphox_to_resonancedb.py raw_data.csv wood data/phone_wood.json")
        sys.exit(1)

    csv_file = sys.argv[1]
    material = sys.argv[2]
    output_json = sys.argv[3]

    phyphox_to_json(csv_file, output_json, material)