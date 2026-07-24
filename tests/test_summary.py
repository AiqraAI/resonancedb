import json

import numpy as np

from resonancedb.summary import format_summary, summarize_dataset

SR = 44100


def write_session(root, material, session, n_taps, *, device="phone01",
                  striker="finger", freq=300.0, amplitude=0.2, source="microphone"):
    d = root / material
    d.mkdir(parents=True, exist_ok=True)
    t = np.arange(int(SR * 0.5)) / SR
    for i in range(n_taps):
        # Clip to full scale the way a real recorder does, so an over-loud
        # amplitude produces genuinely clipped samples.
        sig = np.clip(amplitude * np.exp(-20 * t) * np.sin(2 * np.pi * freq * t), -1.0, 1.0)
        sample = {
            "material": material,
            "vibration": np.round(sig, 6).tolist(),
            "sample_rate_hz": SR,
            "excitation": "manual_tap",
            "source": source,
            "device": device,
            "session": session,
        }
        if striker:
            sample["striker"] = striker
        (d / f"{material}_{device}_{session}_tap{i + 1:02d}.json").write_text(
            json.dumps(sample), encoding="utf-8"
        )


def test_summary_groups_by_session(tmp_path):
    write_session(tmp_path, "wood", "table01", 12, freq=300.0)
    write_session(tmp_path, "wood", "table02", 15, freq=280.0)
    write_session(tmp_path, "glass", "bottle01", 11, freq=3000.0)

    report = summarize_dataset(tmp_path)
    assert report["totals"]["n_sessions"] == 3
    assert report["totals"]["n_taps"] == 38
    assert report["totals"]["materials"] == ["glass", "wood"]

    by_session = {s["session"]: s for s in report["sessions"]}
    assert by_session["table01"]["n_taps"] == 12
    # Reported frequency should land near the simulated resonance
    assert abs(by_session["bottle01"]["peak_freq_median"] - 3000.0) < 60


def test_summary_flags_quality_problems(tmp_path):
    write_session(tmp_path, "wood", "loud01", 12, amplitude=1.5)      # clipped
    write_session(tmp_path, "wood", "faint01", 12, amplitude=0.001)   # too quiet
    write_session(tmp_path, "wood", "short01", 3)                     # too few taps
    write_session(tmp_path, "wood", "nostrike", 12, striker=None)     # no striker

    flags = {s["session"]: s["flags"] for s in summarize_dataset(tmp_path)["sessions"]}
    assert "clipped" in flags["loud01"]
    assert "very quiet" in flags["faint01"]
    assert any("only 3 taps" in f for f in flags["short01"])
    assert "striker unrecorded" in flags["nostrike"]


def test_summary_warns_about_single_session_material(tmp_path):
    write_session(tmp_path, "wood", "table01", 12)
    write_session(tmp_path, "wood", "table02", 12)
    write_session(tmp_path, "glass", "bottle01", 12)  # only one glass session

    warnings = " ".join(summarize_dataset(tmp_path)["warnings"])
    assert "glass" in warnings
    assert "wood" not in warnings.split("cannot be benchmarked")[0].split(":")[-1]


def test_summary_warns_about_single_device(tmp_path):
    write_session(tmp_path, "wood", "table01", 12, device="phone01")
    write_session(tmp_path, "concrete", "wall01", 12, device="phone01")

    warnings = " ".join(summarize_dataset(tmp_path)["warnings"])
    assert "one device" in warnings

    write_session(tmp_path, "wood", "table02", 12, device="phone02")
    warnings = " ".join(summarize_dataset(tmp_path)["warnings"])
    assert "one device" not in warnings


def test_summary_excludes_simulated_by_default(tmp_path):
    write_session(tmp_path, "wood", "real01", 12)
    write_session(tmp_path, "metal", "sim01", 12, source="simulation")

    assert summarize_dataset(tmp_path)["totals"]["n_sessions"] == 1
    assert summarize_dataset(tmp_path, include_simulated=True)["totals"]["n_sessions"] == 2


def test_format_summary_handles_empty(tmp_path):
    assert format_summary(summarize_dataset(tmp_path)) == "No samples found."
