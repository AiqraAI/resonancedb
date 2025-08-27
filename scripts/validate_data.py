import json
import numpy as np

def validate_sample(file_path):
    with open(file_path) as f:
        data = json.load(f)

    required = ['material', 'vibration', 'sample_rate_hz', 'excitation']
    for field in required:
        if field not in data:
            print(f"❌ Missing: {field}")
            return False

    if not isinstance(data['vibration'], list):
        print("❌ Vibration must be a list of numbers")
        return False

    print("✅ Validated:", data['material'])
    return True