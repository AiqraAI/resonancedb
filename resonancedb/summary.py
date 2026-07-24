"""Per-session dataset summary: did these recordings come out well?

`resdb ingest` reports how many taps it found, but not whether they are any
good. This module answers the questions you actually have after a recording
trip: did every session get enough taps, is anything clipped or too quiet to
be useful, are the taps within a session consistent with each other, and is
any material still stuck on a single session (which makes it impossible to
benchmark).
"""

import json
from pathlib import Path

import numpy as np

from .features import compute_feature_vector

# A tap that reaches full scale is clipped: the waveform is flat-topped and
# its spectrum is polluted with harmonics that belong to the recorder, not
# the material.
CLIPPING_LEVEL = 0.99
# Below this the tap is barely above a phone's noise floor.
QUIET_LEVEL = 0.01
# Fewer than this and a session carries little weight in training.
MIN_TAPS = 10


def summarize_dataset(data_dir, *, highpass_hz: float | None = 150.0,
                      include_simulated: bool = False) -> dict:
    """Summarize every session under `data_dir`.

    Returns {"sessions": [...], "totals": {...}, "warnings": [...]}.
    """
    root = Path(data_dir)
    by_session: dict = {}

    for path in sorted(root.rglob("*.json")):
        try:
            with path.open("r", encoding="utf-8") as f:
                d = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        if not include_simulated and d.get("source") == "simulation":
            continue
        if not isinstance(d.get("vibration"), list) or not d["vibration"]:
            continue

        key = (d.get("material", "?"), d.get("session") or path.stem,
               d.get("device", "?"))
        by_session.setdefault(key, []).append(d)

    sessions = []
    for (material, session, device), samples in sorted(by_session.items()):
        peaks, energies, amps, durations = [], [], [], []
        strikers = set()
        rates = set()
        for d in samples:
            sig = np.asarray(d["vibration"], dtype=float)
            sr = d["sample_rate_hz"]
            rates.add(sr)
            strikers.add(d.get("striker") or "unrecorded")
            durations.append(len(sig) / sr if sr else 0.0)
            amps.append(float(np.max(np.abs(sig))) if len(sig) else 0.0)
            try:
                vec = compute_feature_vector(sig, sr, highpass_hz=highpass_hz)
                peaks.append(float(vec[0]))
                energies.append(float(vec[2]))
            except Exception:
                continue

        peaks_arr = np.array(peaks) if peaks else np.array([0.0])
        flags = []
        max_amp = max(amps) if amps else 0.0
        if max_amp >= CLIPPING_LEVEL:
            flags.append("clipped")
        if max_amp < QUIET_LEVEL:
            flags.append("very quiet")
        if len(samples) < MIN_TAPS:
            flags.append(f"only {len(samples)} taps")
        if "unrecorded" in strikers:
            flags.append("striker unrecorded")
        if len(rates) > 1:
            flags.append("mixed sample rates")

        sessions.append({
            "material": material,
            "session": session,
            "device": device,
            "striker": "/".join(sorted(strikers)),
            "n_taps": len(samples),
            "sample_rate_hz": sorted(rates)[0] if rates else None,
            "duration_s": float(np.median(durations)) if durations else 0.0,
            "peak_freq_median": float(np.median(peaks_arr)),
            "peak_freq_q1": float(np.percentile(peaks_arr, 25)),
            "peak_freq_q3": float(np.percentile(peaks_arr, 75)),
            "energy_median": float(np.median(energies)) if energies else 0.0,
            "max_amplitude": max_amp,
            "flags": flags,
        })

    # Materials recorded in only one session cannot be benchmarked: hold that
    # session out and there is nothing left to learn the material from.
    sessions_per_material: dict = {}
    for s in sessions:
        sessions_per_material.setdefault(s["material"], []).append(s["session"])
    single = sorted(m for m, v in sessions_per_material.items() if len(v) == 1)

    warnings = []
    if single:
        warnings.append(
            "only one session each, so they cannot be benchmarked: "
            + ", ".join(single)
        )
    devices = {s["device"] for s in sessions}
    if len(devices) == 1 and sessions:
        warnings.append(
            f"all data comes from one device ({sorted(devices)[0]}), so "
            "cross-device generalization is still untested"
        )

    return {
        "sessions": sessions,
        "totals": {
            "n_sessions": len(sessions),
            "n_taps": sum(s["n_taps"] for s in sessions),
            "materials": sorted(sessions_per_material),
            "devices": sorted(devices),
        },
        "warnings": warnings,
    }


def format_summary(report: dict) -> str:
    """Render a summary report as an aligned text table."""
    sessions = report["sessions"]
    if not sessions:
        return "No samples found."

    lines = []
    header = (f"{'material':10s} {'session':16s} {'device':9s} {'striker':10s} "
              f"{'taps':>4s} {'peak Hz (q1-q3)':>22s} {'level':>6s}  notes")
    lines.append(header)
    lines.append("-" * len(header))

    for s in sessions:
        peak = (f"{s['peak_freq_median']:.0f} "
                f"({s['peak_freq_q1']:.0f}-{s['peak_freq_q3']:.0f})")
        lines.append(
            f"{s['material'][:10]:10s} {s['session'][:16]:16s} "
            f"{s['device'][:9]:9s} {s['striker'][:10]:10s} "
            f"{s['n_taps']:4d} {peak:>22s} {s['max_amplitude']:6.2f}  "
            f"{', '.join(s['flags'])}"
        )

    t = report["totals"]
    lines.append("")
    lines.append(f"{t['n_sessions']} session(s), {t['n_taps']} taps, "
                 f"materials: {', '.join(t['materials'])}")
    for w in report["warnings"]:
        lines.append(f"  note: {w}")
    return "\n".join(lines)
