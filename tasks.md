# Project Tasks and Roadmap

This task list tracks implementation to address identified gaps and risks.

## Phase 0 — Foundation (Quick Wins)
- [x] Add `requirements.txt` with core dependencies.
- [x] Create shared feature extraction module `python/features.py`.
- [x] Refactor `models/train_classifier.py` to use shared features.
- [x] Refactor `models/predict.py` and `demo.py` to use shared features.
- [x] Update `docs/index.md` with Getting Started and workflow summary.
- [x] Fill `examples/02-dataset-overview.ipynb` to scan/plot dataset stats.

Acceptance:
- `pip install -r requirements.txt` succeeds.
- A single feature function is used across train/predict/demo.
- Docs explain install, training, prediction, and dataset basics.

## Phase 1 — Reliability & CLI
- [x] Add tests for feature parity and validation utilities.
- [x] Add tests for Phyphox CSV conversion.
- [x] Provide a simple CLI (`resdb`) for simulate/train/predict/validate.
- [x] Integrate CI to run `pytest` on push and pull requests.

Acceptance:
- `pytest` passes locally.
- CLI offers helpful usage and errors.
- CI runs on pushes and pull requests and reports test results.

## Phase 2 — Signal & Data Quality
- [x] Add preprocessing (detrend, Hann window) and length normalization.
- [x] Optional resampling to standard `sample_rate_hz` with warnings for low-rate data.
- [x] Expand features (spectral descriptors, peaks, autocorrelation).
- [x] Expose extra feature flags in CLI for predict/train.
 - [x] Add preprocess flag group to CLI (target_length, resample_rate_hz, toggles).
 - [x] Persist feature configuration inside saved model metadata for consistent prediction.

## Phase 3 — Evaluation & Ops
- [x] Add `resdb evaluate` command to report accuracy, precision/recall, and confusion matrix.
 - [x] Implement k-fold cross-validation and simple hyperparameter search.
- [x] Add `resdb inspect` to print saved model metadata (features, preprocess, input dim).
- [x] Save evaluation report (JSON) and optional confusion matrix image to `models/`.
- [ ] Expand docs with guidance on dataset quality and evaluation workflows.

Acceptance:
- Configurable preprocessing pipeline with sane defaults.
- Stable feature vector shapes independent of raw sample length.

## Phase 3 — Model Robustness & Portability
- [ ] Add evaluation script (CV accuracy, confusion matrix).
- [x] Export trained model to `onnx` and add usage examples.
- [ ] Export trained model to `tflite` (TensorFlow-based) and add usage examples.

Acceptance:
- Metrics saved and printed; exports load successfully.

## Phase 4 — Data Governance & Community
- [ ] Define file naming conventions and update `CONTRIBUTING.md`.
- [ ] Add `scripts/README.md` with usage examples.

Acceptance:
- New submissions follow conventions and pass validation.

## Phase 5 — Hardware Docs & Coherence
- [ ] Align README firmware references or scaffold `firmware/` minimally.

Acceptance:
- README points to existing resources; setup is reproducible.