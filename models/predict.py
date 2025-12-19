# models/predict.py
import joblib
import json
import numpy as np
from pathlib import Path
from python.features import compute_feature_vector

def predict_material(file_path):
    # Load model
    model = joblib.load('models/material_model.pkl')
    
    # Load data
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    signal = np.array(data['vibration'])
    sample_rate = data['sample_rate_hz']

    # Extract features via shared module
    vec = compute_feature_vector(signal, sample_rate)
    features = np.array([vec])  # shape (1, 3)
    prediction = model.predict(features)[0]
    
    print(f"🔮 This is {prediction.upper()}!")
    return prediction

# Test it
if __name__ == "__main__":
    predict_material("data/test_glass.json")