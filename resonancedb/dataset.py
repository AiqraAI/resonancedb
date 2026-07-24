"""Dataset loading: JSON sample files -> feature matrix + labels."""

import json
from pathlib import Path

import numpy as np

from .features import compute_feature_vector
from .schema import validate_sample_dict


def load_data(
    data_dir,
    *,
    extra=False,
    top_k_peaks: int = 3,
    detrend: bool | None = None,
    window: str | None = "hann",
    target_length: int | None = None,
    resample_rate_hz: float | None = None,
    verbose: bool = True,
):
    """Load every valid .json sample under `data_dir` and extract features.

    `window` is passed through as-is: None means NO window; the default is
    the standard Hann window.

    Returns (X, y) as numpy arrays; both empty if nothing valid was found.
    """
    features = []
    labels = []
    data_path = Path(data_dir)

    def log(msg):
        if verbose:
            print(msg)

    log(f"Looking for data in: {data_path.absolute()}")

    if not data_path.exists():
        log(f"[FAIL] Folder does not exist: {data_path}")
        return np.array([]), np.array([])

    json_files = sorted(data_path.rglob("*.json"))
    log(f"Found {len(json_files)} JSON file(s)")

    for file_path in json_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            log(f"[SKIP] {file_path.name}: {e}")
            continue

        # Training only needs the core fields; tolerate missing provenance
        # metadata but reject structurally invalid samples.
        check = dict(data)
        check.setdefault("excitation", "unknown")
        check.setdefault("source", "unknown")
        errors = validate_sample_dict(check)
        if errors:
            log(f"[SKIP] {file_path.name}: {'; '.join(errors)}")
            continue

        signal = np.array(data["vibration"])
        sample_rate = data["sample_rate_hz"]
        detrend_flag = True if detrend is None else bool(detrend)

        try:
            vec = compute_feature_vector(
                signal,
                sample_rate,
                detrend=detrend_flag,
                window=window,
                target_length=target_length,
                resample_rate_hz=resample_rate_hz,
                extra=extra,
                top_k_peaks=top_k_peaks,
            )
        except Exception as e:
            log(f"[SKIP] {file_path.name}: feature extraction failed: {e}")
            continue

        features.append(vec.tolist())
        labels.append(data["material"])
        log(f"[OK] {file_path.name}: {data['material']} | "
            f"{vec[0]:.1f} Hz | decay={vec[1]:.3f} | energy={vec[2]:.3f}")

    if not features:
        log("[FAIL] No valid data loaded.")
        return np.array([]), np.array([])

    log(f"Loaded {len(features)} sample(s): {sorted(set(labels))}")
    return np.array(features), np.array(labels)
