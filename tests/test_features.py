import numpy as np
from python.features import compute_feature_vector
from python.simulate_tap import simulate_tap


def test_peak_frequency_matches_simulated():
    sr = 1000
    freq = 300.0
    damping = 1.0
    sig = simulate_tap(freq, damping, duration=2.0, sample_rate=sr)
    vec = compute_feature_vector(sig, sr)
    peak_freq = vec[0]
    assert abs(peak_freq - freq) < 10.0


def test_energy_nonnegative():
    sr = 1000
    sig = np.random.randn(sr)
    vec = compute_feature_vector(sig, sr)
    energy = vec[2]
    assert energy >= 0.0


def test_deterministic_same_input():
    sr = 1000
    t = np.arange(sr) / sr
    sig = np.sin(2 * np.pi * 50.0 * t)
    v1 = compute_feature_vector(sig, sr)
    v2 = compute_feature_vector(sig, sr)
    assert np.allclose(v1, v2)


def test_detrend_removes_dc_offset():
    sr = 1000
    t = np.arange(sr) / sr
    base = np.sin(2 * np.pi * 80.0 * t)
    sig_with_offset = base + 0.75
    v_offset = compute_feature_vector(sig_with_offset, sr)
    v_base = compute_feature_vector(base, sr)
    # Peak frequency and decay rate should be effectively identical after detrend
    assert abs(v_offset[0] - v_base[0]) < 1.0
    assert abs(v_offset[1] - v_base[1]) < 1e-3