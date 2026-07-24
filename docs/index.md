# ResonanceDB Docs

Your guide to installing, contributing data, training models, and exploring
the dataset.

## Getting Started

```bash
git clone https://github.com/AiqraAI/resonancedb.git
cd resonancedb
pip install -e ".[dev]"
```

This installs the `resonancedb` package and the `resdb` command.

## Data Schema

- Read the schema in [DATA_FORMAT.md](DATA_FORMAT.md).
- Required: `material`, `vibration[]`, `sample_rate_hz`, `excitation`, `source`.
- Optional: `temperature_c`, `thickness_mm`, `load_g`, `mounting`, `device`, `notes`.
- Validation rules live in `resonancedb.schema` — the single source of truth
  for what a valid sample is.

## Validate

```bash
resdb validate --data data
```

Exits non-zero if any file fails, so it can gate CI on data pull requests.

## Simulate

```bash
resdb simulate --out-dir data/simulated
```

Writes one JSON sample per material (glass, wood, metal, plastic) in the
standard format. The default 4 kHz simulation rate keeps every simulated
resonance below Nyquist. Simulated data is for pipeline testing — don't
submit it to the dataset.

## Train

```bash
resdb train --data data --extra all --target-length 1024
```

- Preprocess flags: `--target-length`, `--resample-rate-hz`,
  `--detrend/--no-detrend`, `--window/--no-window`
- Extra features: `--extra all` (or a comma-separated subset), `--top-k-peaks 3`
- Output: `models/material_model.pkl` — a package containing the model AND
  the feature configuration it was trained with, so prediction always
  reproduces the same pipeline.

## Predict

```bash
resdb predict data/simulated/glass.json
```

Uses the configuration embedded in the saved model automatically; CLI flags
override it (with a shape-mismatch fallback to the saved config).

## Evaluate, Tune, Inspect, Export

```bash
resdb evaluate --model models/material_model.pkl --data data
resdb tune --data data --cv 2 --n-estimators 10,50 --max-depth None,5 --extra all
resdb inspect --model models/material_model.pkl
resdb export --model models/material_model.pkl --verify
```

- `evaluate` writes `eval_report.json` (and `confusion_matrix.png` when
  matplotlib is installed: `pip install "resonancedb[plots]"`).
- `export` produces ONNX and requires `pip install "resonancedb[export]"`.
  TFLite is not supported for scikit-learn models — use ONNX.

## Feature Vector

Feature extraction lives in `resonancedb/features.py` and is shared by
training, evaluation, and prediction.

- Base: `[peak_freq, decay_rate, energy]`
- Extras (`extra=True`): `spectral_centroid`, `spectral_bandwidth`, `zcr`,
  top-k FFT peak frequencies, autocorrelation lag
- Preprocessing (`resonancedb/preprocess.py`): mean detrend, Hann window,
  optional length normalization and resampling. When a signal is resampled,
  all frequency-dependent features use the effective post-resample rate.

## Notebooks

- Analyze simulated taps: `examples/01-analyze-tap-simulation.ipynb`
- Dataset overview: `examples/02-dataset-overview.ipynb`
- If you need Jupyter: `pip install jupyter`

## Contribution & Governance

- Follow [../CONTRIBUTING.md](../CONTRIBUTING.md) to add data and improvements.
- Data lives under `data/<material>/`; prefer file naming like
  `source_device_YYYYMMDD_session.json`.
- Always run `resdb validate --data data` before submitting.

## Phone capture (legacy)

The phyphox CSV converter (`scripts/phyphox_to_resonancedb.py`, requires
`pip install "resonancedb[phone]"`) is kept as a legacy path for
contact-vibration data. The primary capture path going forward is
microphone-based — see [../ROADMAP.md](../ROADMAP.md) Phase 2.

## Tips

- The signal should contain the tap event; trimming around the largest spike
  improves consistency.
- Record several taps per object/session — variation is data.
