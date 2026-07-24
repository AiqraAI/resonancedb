"""Audio capture support: WAV loading, tap onset detection, and trimming.

Microphone recordings are the primary capture path (see ROADMAP.md): every
phone records at 44.1+ kHz, which covers the kHz-range resonances of glass
and metal that low-rate accelerometers cannot see.

Typical flow:

    signal, sr = load_wav("kitchen_table.wav")
    segments = extract_taps(signal, sr)          # one array per detected tap
    # -> resdb ingest wraps each segment in a schema-valid JSON sample
"""

from pathlib import Path

import numpy as np


def load_audio(path) -> tuple[np.ndarray, int]:
    """Load any common audio/video file as a mono float signal in [-1, 1].

    WAV files are read directly. Other formats (mp4, m4a, mp3, ogg, phone
    video recordings, ...) are decoded through a bundled ffmpeg, which
    requires the optional dependency: pip install "resonancedb[media]".
    """
    p = Path(path)
    if p.suffix.lower() == ".wav":
        return load_wav(p)
    return _load_via_ffmpeg(p)


def _load_via_ffmpeg(path: Path) -> tuple[np.ndarray, int]:
    """Decode a non-WAV file to a temporary WAV using bundled ffmpeg."""
    import subprocess
    import tempfile

    try:
        import imageio_ffmpeg
    except ImportError:
        raise RuntimeError(
            f"{path.name} is not a WAV file. Decoding other formats needs "
            "the media extra: pip install \"resonancedb[media]\""
        )

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    try:
        result = subprocess.run(
            [ffmpeg, "-y", "-i", str(path), "-vn",
             "-acodec", "pcm_s16le", tmp.name],
            capture_output=True,
        )
        if result.returncode != 0:
            tail = result.stderr.decode(errors="replace").strip().splitlines()[-1:]
            raise RuntimeError(
                f"ffmpeg could not decode {path.name}: {' '.join(tail)}"
            )
        return load_wav(tmp.name)
    finally:
        Path(tmp.name).unlink(missing_ok=True)


def load_wav(path) -> tuple[np.ndarray, int]:
    """Load a WAV file as a mono float signal in [-1, 1].

    Returns (signal, sample_rate_hz). Multi-channel audio is averaged to
    mono. Integer PCM is scaled to [-1, 1]; float PCM is passed through.
    """
    from scipy.io import wavfile

    sample_rate, data = wavfile.read(str(path))

    # Capture the on-disk dtype BEFORE any processing: averaging channels
    # promotes integer PCM to float64, which would skip the scaling below.
    raw_dtype = data.dtype

    if data.ndim > 1:
        data = data.mean(axis=1)

    signal = data.astype(np.float64)
    if np.issubdtype(raw_dtype, np.integer):
        signal /= float(np.iinfo(raw_dtype).max)

    return signal, int(sample_rate)


def detect_taps(
    signal: np.ndarray,
    sample_rate_hz: float,
    *,
    threshold_ratio: float = 0.25,
    min_separation_s: float = 0.25,
) -> list[int]:
    """Find tap onsets in a recording. Returns sample indices, in order.

    Method: a short-window energy envelope is compared against a threshold
    that adapts to both the loudest event and the noise floor, so it works
    for recordings that are mostly silence as well as noisier rooms. Events
    closer together than `min_separation_s` are treated as one tap (a tap
    plus its ring, not two taps).
    """
    x = np.abs(np.asarray(signal, dtype=float))
    if len(x) == 0:
        return []

    # ~2 ms smoothing window: long enough to bridge single-sample spikes,
    # short enough to keep the onset sharp.
    win = max(1, int(0.002 * sample_rate_hz))
    env = np.convolve(x, np.ones(win) / win, mode="same")

    peak = float(env.max())
    if peak <= 0.0:
        return []
    noise_floor = float(np.median(env))

    # A real tap must rise well above the noise floor AND be a meaningful
    # fraction of the loudest event in the recording.
    threshold = max(noise_floor * 6.0, peak * threshold_ratio)

    min_sep = max(1, int(min_separation_s * sample_rate_hz))
    onsets: list[int] = []
    i = 0
    n = len(env)
    while i < n:
        if env[i] >= threshold:
            onsets.append(i)
            i += min_sep
        else:
            i += 1
    return onsets


def trim_tap(
    signal: np.ndarray,
    sample_rate_hz: float,
    onset_idx: int,
    *,
    pre_s: float = 0.005,
    duration_s: float = 0.5,
) -> np.ndarray:
    """Cut one tap out of a recording.

    The segment starts `pre_s` before the detected onset (so the attack is
    never clipped) and lasts `duration_s` in total.
    """
    start = max(0, onset_idx - int(pre_s * sample_rate_hz))
    end = min(len(signal), start + int(duration_s * sample_rate_hz))
    return np.asarray(signal[start:end], dtype=float)


def extract_taps(
    signal: np.ndarray,
    sample_rate_hz: float,
    *,
    threshold_ratio: float = 0.25,
    min_separation_s: float = 0.25,
    pre_s: float = 0.005,
    duration_s: float = 0.5,
) -> list[np.ndarray]:
    """Detect and trim every tap in a recording. Returns a list of segments."""
    onsets = detect_taps(
        signal, sample_rate_hz,
        threshold_ratio=threshold_ratio,
        min_separation_s=min_separation_s,
    )
    return [
        trim_tap(signal, sample_rate_hz, onset, pre_s=pre_s, duration_s=duration_s)
        for onset in onsets
    ]


def wav_to_samples(
    wav_path,
    material: str,
    *,
    device: str = "unknown",
    session: str | None = None,
    excitation: str = "manual_tap",
    source: str = "microphone",
    striker: str | None = None,
    notes: str | None = None,
    threshold_ratio: float = 0.25,
    min_separation_s: float = 0.25,
    duration_s: float = 0.5,
) -> list[dict]:
    """Convert one recording (WAV, or any ffmpeg-decodable format) into a
    list of schema-valid sample dicts, one per detected tap.
    """
    signal, sample_rate = load_audio(wav_path)
    segments = extract_taps(
        signal, sample_rate,
        threshold_ratio=threshold_ratio,
        min_separation_s=min_separation_s,
        duration_s=duration_s,
    )

    stem = Path(wav_path).stem
    samples = []
    for i, seg in enumerate(segments, start=1):
        sample = {
            "material": material,
            # 6 decimals is well below the noise floor of any microphone
            # (16-bit PCM resolves ~3e-5) and roughly halves the JSON size.
            "vibration": np.round(seg, 6).tolist(),
            "sample_rate_hz": sample_rate,
            "excitation": excitation,
            "source": source,
            "device": device,
            "session": session or stem,
            "notes": notes or f"tap {i}/{len(segments)} from {Path(wav_path).name}",
        }
        # What struck the object changes the excitation a lot (a metal key
        # excites far higher frequencies than a fingertip), so it has to be
        # recorded or it becomes a hidden variable in every comparison.
        if striker:
            sample["striker"] = striker
        samples.append(sample)
    return samples
