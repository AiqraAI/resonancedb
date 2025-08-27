# scripts/validate_data.py
import json
import os
from pathlib import Path

def validate_sample(file_path):
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return False

    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in {file_path}: {e}")
        return False

    # Required fields
    required = ['material', 'vibration', 'sample_rate_hz', 'excitation', 'source']
    missing = [f for f in required if f not in data]
    if missing:
        print(f"❌ Missing fields: {missing}")
        return False

    valid = True  # Track if all checks pass

    # 1. material: must be non-empty string
    if not isinstance(data['material'], str) or not data['material'].strip():
        print("❌ 'material' must be a non-empty string")
        valid = False

    # 2. vibration: must be non-empty list of numbers
    if not isinstance(data['vibration'], (list, tuple)):
        print("❌ 'vibration' must be a list of numbers")
        valid = False
    else:
        if len(data['vibration']) == 0:
            print("❌ 'vibration' array is empty")
            valid = False
        else:
            # Optional: check if all items are numbers
            if not all(isinstance(x, (int, float)) for x in data['vibration']):
                print("❌ 'vibration' must contain only numbers")
                valid = False

    # 3. sample_rate_hz: must be positive number
    if not isinstance(data['sample_rate_hz'], (int, float)):
        print("❌ 'sample_rate_hz' must be a number")
        valid = False
    elif data['sample_rate_hz'] <= 0:
        print("❌ 'sample_rate_hz' must be greater than 0")
        valid = False

    # 4. excitation: non-empty string
    if not isinstance(data['excitation'], str) or not data['excitation'].strip():
        print("❌ 'excitation' must be a non-empty string")
        valid = False

    # 5. source: non-empty string
    if not isinstance(data['source'], str) or not data['source'].strip():
        print("❌ 'source' must be a non-empty string")
        valid = False

    # Optional: device, temperature, etc. can be added later

    if valid:
        print(f"✅ Validated: {data['material']} ({len(data['vibration'])} samples)")
    else:
        print(f"❌ Failed validation: {file_path.name}")

    return valid


# --- PATH LOGIC ---
if __name__ == "__main__":
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / "data"

    print(f"🔍 Looking for data in: {data_dir}")

    if not data_dir.exists():
        print(f"📁 ERROR: '{data_dir}' does not exist!")
        print("💡 Run 'mkdir data' in your project root, then try again.")
        exit(1)

    valid = True
    has_json = False

    for json_file in data_dir.rglob("*.json"):
        has_json = True
        if not validate_sample(json_file):
            valid = False

    if not has_json:
        print(f"🟡 No JSON files found in {data_dir}")
        print("💡 Create a file like 'data/test_glass.json' to get started.")

    print("\n" + ("✅ ALL FILES VALID" if valid and has_json else "❌ SOME FILES INVALID"))