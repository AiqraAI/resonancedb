# models/predict.py
import joblib
import json
import numpy as np
from pathlib import Path

def predict_material(file_path):
    # Load model
    model = joblib.load('models/material_model.pkl')
    
    # Load data
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    signal = np.array(data['vibration'])
    sample_rate = data['sample_rate_hz']
    
    # Extract features (same as training)
    fft_vals = np.abs(np.fft.fft(signal))
    freqs = np.fft.fftfreq(len(signal), 1/sample_rate)
    peak_freq = freqs[np.argmax(fft_vals[:len(fft_vals)//2])]
    
    envelope = np.abs(signal)
    decay_rate = -np.polyfit(np.arange(len(envelope)), np.log(envelope + 1e-8), 1)[0]
    energy = np.sum(signal**2)
    
    # Predict
    features = np.array([[peak_freq, decay_rate, energy]])
    prediction = model.predict(features)[0]
    
    print(f"🔮 This is {prediction.upper()}!")
    return prediction

# Test it
if __name__ == "__main__":
    predict_material("data/test_glass.json")