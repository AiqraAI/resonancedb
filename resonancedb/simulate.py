"""Synthetic tap generation for testing and demos.

Simulated samples are written in the same JSON format as real submissions
(see docs/DATA_FORMAT.md) so the whole pipeline (validate, train, predict)
works on them unchanged.
"""

import json
from pathlib import Path

import numpy as np

# Nominal resonant frequency (Hz) and damping factor per material.
# NOTE: with the default 4 kHz simulation rate every frequency below sits
# safely under Nyquist (2 kHz). Do not lower the sample rate below twice the
# highest frequency here or the "signal" will alias into garbage.
MATERIALS = {
    "glass": {"frequency": 800, "damping": 0.4},
    "wood": {"frequency": 300, "damping": 1.5},
    "metal": {"frequency": 1000, "damping": 0.3},
    "plastic": {"frequency": 500, "damping": 1.0},
}

DEFAULT_SAMPLE_RATE = 4000


def simulate_tap(frequency: float, damping: float, duration: float = 2.0,
                 sample_rate: int = 1000) -> np.ndarray:
    """Return a decaying sinusoid approximating a tap response."""
    t = np.linspace(0, duration, int(sample_rate * duration))
    return np.exp(-damping * t) * np.sin(2 * np.pi * frequency * t)


def generate_dataset(out_dir, sample_rate: int = DEFAULT_SAMPLE_RATE,
                     duration: float = 2.0, materials: dict | None = None) -> list[Path]:
    """Generate one JSON sample per material into `out_dir`.

    Returns the list of files written.
    """
    mats = materials or MATERIALS
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    written = []
    for name, props in mats.items():
        if props["frequency"] * 2 > sample_rate:
            raise ValueError(
                f"{name}: frequency {props['frequency']} Hz needs a sample rate "
                f">= {props['frequency'] * 2} Hz (got {sample_rate})"
            )
        signal = simulate_tap(props["frequency"], props["damping"],
                              duration=duration, sample_rate=sample_rate)
        payload = {
            "material": name,
            "vibration": signal.tolist(),
            "sample_rate_hz": sample_rate,
            "excitation": "simulated",
            "source": "simulation",
            "notes": f"synthetic tap: f0={props['frequency']} Hz, damping={props['damping']}",
        }
        path = out / f"{name}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f)
        written.append(path)
    return written
