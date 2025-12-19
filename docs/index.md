# ResonanceDB Docs

Your guide to installing, contributing data, training models, and exploring the dataset.

## Getting Started
- Create a virtual environment and install dependencies:
  - `python -m venv .venv`
  - `.venv\Scripts\Activate.ps1` (PowerShell on Windows)
  - `pip install -r requirements.txt`

## Data Schema
- Read the schema in `docs/DATA_FORMAT.md`.
- Required: `material`, `vibration[]` (g), `sample_rate_hz`, `excitation`, `source`.
- Optional: `temperature_c`, `thickness_mm`, `load_g`, `mounting`, `device`, `notes`.

## Record, Convert, and Validate
1. Record with Phyphox or device of choice.
2. Convert CSV → JSON using:
   - `python scripts/phyphox_to_resonancedb.py "scripts/Raw Data.csv" wood data/phone_wood.json`
3. Validate all JSON files:
   - `python scripts/validate_data.py`

## Train
- Ensure you have at least a couple of samples (preferably multiple materials).
- Train a baseline classifier:
  - `python models/train_classifier.py`
- Output: `models/material_model.pkl` (RandomForest)

- CLI training:
  - `python scripts/resdb.py train --data data --out models/material_model.pkl`
  - Preprocess flags: `--target-length`, `--resample-rate-hz`, `--detrend/--no-detrend`, `--window/--no-window`
  - Extra features: `--extra all`, `--top-k-peaks 3`

## Predict
- Quick demo:
  - `python demo.py` (uses `data/test_wood.json` by default)
- Or run the predictor on a file:
  - `python models/predict.py` (edit the `__main__` path or import and call)

- CLI prediction:
  - `python scripts/resdb.py predict data/test_wood.json --model models/material_model.pkl`
  - Uses saved model configuration automatically unless overridden by flags.

## Simulation
- Generate synthetic taps:
  - `python python/simulate_tap.py`
- Files saved to `data/simulated/*.npz` with metadata for frequency and damping.

## Features (Shared)
- Unified feature extraction lives in `python/features.py` and is used by training, prediction, and the demo.
- Current vector: `[peak_freq, decay_rate, energy]` with optional preprocessing (detrend + Hann window).

- Preprocessing: detrend, Hann window, optional length normalization and resampling.

## CLI — Inspect, Evaluate, Tune
- Inspect model metadata:
  - `python scripts/resdb.py inspect --model models/material_model.pkl`
- Evaluate on dataset:
  - `python scripts/resdb.py evaluate --model models/material_model.pkl --data data --save-dir models --save-confusion-matrix`
  - Saves `eval_report.json` and `confusion_matrix.png`.
- Tune hyperparameters (k-fold CV):
  - `python scripts/resdb.py tune --data data --out models/material_model.pkl --cv 2 --n-estimators 10,50 --max-depth None,5 --max-features sqrt --extra all --top-k-peaks 3`

## Export (ONNX)
- Export trained model to ONNX for portability:
  - `python scripts/resdb.py export --model models/material_model.pkl --out models/material_model.onnx --verify`
- Requires `skl2onnx`, `onnx`, and `onnxruntime` (already in `requirements.txt`).
- TFLite: scikit‑learn `RandomForest` cannot be exported directly. Use ONNX or retrain with TensorFlow/Keras for TFLite.

## Notebooks
- Analyze simulated taps: `examples/01-analyze-tap-simulation.ipynb.ipynb`
- Dataset overview: `examples/02-dataset-overview.ipynb` (scans `data/`, computes features, and plots histograms).
- If you need Jupyter: `pip install jupyter`

## Contribution & Governance
- Follow `CONTRIBUTING.md` to add data and improvements.
- Prefer file naming like: `material_source_device_YYYYMMDD_session.json`.
- Always run `scripts/validate_data.py` before submitting.

## Tips
- Phone data may have lower `sample_rate_hz` (~100 Hz); it still works but features are noisier. Consider multiple taps per session.
- For FFT, the signal should contain the tap event; trimming around the largest spike improves consistency.