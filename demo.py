import joblib
import json
import numpy as np
from pathlib import Path
from python.features import compute_feature_vector, compute_features

def predict_material(file_path):
    # Load the trained model
    try:
        model = joblib.load('models/material_model.pkl')
    except FileNotFoundError:
        print("❌ Model not found! Run 'python models/train_classifier.py' first.")
        return

    # Load the vibration data
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Data file not found: {file_path}")
        return

    # Extract required data
    signal = np.array(data['vibration'])
    sample_rate = data['sample_rate_hz']

    # --- 🔧 Feature Extraction (shared) ---
    feat_dict = compute_features(signal, sample_rate)
    peak_freq = feat_dict["peak_freq"]
    decay_rate = feat_dict["decay_rate"]
    energy = feat_dict["energy"]

    # Print what we extracted
    print(f"📊 Features: {peak_freq:.1f} Hz, decay={decay_rate:.3f}, energy={energy:.5f}")

    # --- 🧠 Make Prediction ---
    features = np.array([[peak_freq, decay_rate, energy]])  # Must be 2D array
    prediction = model.predict(features)[0]

    print(f"🔮 This is {prediction.upper()}!")
    return prediction

# --- 🚀 Run the demo ---
if __name__ == "__main__":
    predict_material("data/test_wood.json")