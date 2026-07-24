import numpy as np
import pytest

from resonancedb.features import compute_feature_vector
from resonancedb.preprocess import PreprocessConfig, highpass, run_pipeline

SR = 44100


def rumble_plus_tap(sr=SR, rumble_hz=50.0, tap_hz=800.0,
                    rumble_amp=1.0, tap_amp=0.15, duration=0.5):
    """A quiet tap buried under loud low-frequency rumble.

    This is what real phone recordings look like: sub-100 Hz handling noise
    and mains hum carry most of the energy, while the material's resonance
    is smaller but much higher in frequency.
    """
    t = np.arange(int(sr * duration)) / sr
    rumble = rumble_amp * np.sin(2 * np.pi * rumble_hz * t)
    tap = tap_amp * np.exp(-25 * t) * np.sin(2 * np.pi * tap_hz * t)
    return rumble + tap


def test_highpass_removes_low_frequency_energy():
    x = rumble_plus_tap()
    filtered = highpass(x, SR, 150.0)

    def band_power(sig, lo, hi):
        freqs = np.fft.rfftfreq(len(sig), 1 / SR)
        mag = np.abs(np.fft.rfft(sig))
        m = (freqs >= lo) & (freqs < hi)
        return float((mag[m] ** 2).sum())

    # Rumble is crushed, the tap band survives
    assert band_power(filtered, 0, 100) < band_power(x, 0, 100) / 1000
    assert band_power(filtered, 700, 900) > band_power(x, 700, 900) * 0.5


def test_highpass_rescues_peak_frequency():
    """Without filtering, peak_freq reports the rumble, not the material."""
    x = rumble_plus_tap(tap_hz=800.0)

    unfiltered = compute_feature_vector(x, SR)
    assert unfiltered[0] < 100  # locked onto the 50 Hz rumble

    filtered = compute_feature_vector(x, SR, highpass_hz=150.0)
    assert abs(filtered[0] - 800.0) < 20  # finds the real resonance


def test_highpass_out_of_range_warns_and_passes_through():
    x = rumble_plus_tap()
    with pytest.warns(RuntimeWarning, match="out of range"):
        out = highpass(x, SR, SR)  # cutoff at the sample rate is nonsense
    assert np.array_equal(out, x)


def test_highpass_disabled_by_default():
    """Existing behavior is unchanged unless highpass_hz is set."""
    x = rumble_plus_tap()
    default = run_pipeline(x, SR, PreprocessConfig())
    explicit_off = run_pipeline(x, SR, PreprocessConfig(highpass_hz=None))
    assert np.allclose(default, explicit_off)


def test_highpass_survives_short_signals():
    """Very short signals are returned untouched rather than crashing filtfilt."""
    short = np.array([0.1, -0.2, 0.3, -0.1, 0.05])
    out = highpass(short, SR, 150.0)
    assert np.array_equal(out, short)
