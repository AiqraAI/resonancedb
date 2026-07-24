import numpy as np
import pytest
from scipy.io import wavfile

from resonancedb.audio import detect_taps, extract_taps, load_wav, wav_to_samples
from resonancedb.schema import validate_sample_dict

SR = 48000
TAP_TIMES_S = [0.5, 1.5, 2.5]


def synth_recording(sr=SR, duration_s=3.0, tap_times=TAP_TIMES_S, noise=0.002):
    """A quiet room with three clear taps (decaying 1 kHz sinusoids)."""
    rng = np.random.default_rng(0)
    signal = rng.normal(0, noise, int(sr * duration_s))
    t_tap = np.arange(int(sr * 0.2)) / sr
    tap = 0.8 * np.exp(-30 * t_tap) * np.sin(2 * np.pi * 1000 * t_tap)
    for t0 in tap_times:
        start = int(t0 * sr)
        signal[start:start + len(tap)] += tap
    return signal


def write_wav(path, signal, sr=SR, stereo=False):
    pcm = (np.clip(signal, -1, 1) * 32767).astype(np.int16)
    if stereo:
        pcm = np.column_stack([pcm, pcm])
    wavfile.write(str(path), sr, pcm)


def test_load_wav_mono_normalized(tmp_path):
    signal = synth_recording()
    path = tmp_path / "rec.wav"
    write_wav(path, signal, stereo=True)

    loaded, sr = load_wav(path)
    assert sr == SR
    assert loaded.ndim == 1
    assert np.max(np.abs(loaded)) <= 1.0
    assert np.max(np.abs(loaded)) > 0.5  # the taps survived


def test_detect_taps_finds_all_three():
    signal = synth_recording()
    onsets = detect_taps(signal, SR)
    assert len(onsets) == 3
    for onset, expected_s in zip(onsets, TAP_TIMES_S):
        assert abs(onset / SR - expected_s) < 0.01  # within 10 ms


def test_detect_taps_silence_returns_nothing():
    rng = np.random.default_rng(1)
    silence = rng.normal(0, 0.001, SR)
    assert detect_taps(silence, SR) == []


def test_extract_taps_segment_length():
    signal = synth_recording()
    segments = extract_taps(signal, SR, duration_s=0.4)
    assert len(segments) == 3
    for seg in segments:
        assert len(seg) == int(0.4 * SR)


def test_wav_to_samples_schema_valid(tmp_path):
    path = tmp_path / "oak_table_pixel7_01.wav"
    write_wav(path, synth_recording())

    samples = wav_to_samples(path, "oak_wood", device="pixel7", session="s01")
    assert len(samples) == 3
    for sample in samples:
        assert validate_sample_dict(sample) == []
        assert sample["material"] == "oak_wood"
        assert sample["device"] == "pixel7"
        assert sample["session"] == "s01"
        assert sample["sample_rate_hz"] == SR


def test_wav_to_samples_default_session_is_filename(tmp_path):
    path = tmp_path / "kitchen_glass_take2.wav"
    write_wav(path, synth_recording())
    samples = wav_to_samples(path, "glass", device="iphone")
    assert samples[0]["session"] == "kitchen_glass_take2"
