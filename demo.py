import joblib
import json
import numpy as np
from pathlib import Path

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

    # --- 🔧 Feature Extraction (MUST be included!) ---
    # 1. Dominant frequency
    fft_vals = np.abs(np.fft.fft(signal))
    freqs = np.fft.fftfreq(len(signal), 1/sample_rate)
    half = len(fft_vals) // 2
    peak_freq = freqs[np.argmax(fft_vals[:half])]

    # 2. Decay rate (damping)
    envelope = np.abs(signal)
    log_env = np.log(envelope + 1e-8)  # Avoid log(0)
    decay_rate = -np.polyfit(np.arange(len(log_env)), log_env, 1)[0]

    # 3. Energy
    energy = np.sum(signal ** 2)

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