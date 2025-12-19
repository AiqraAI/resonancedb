import numpy as np
from .preprocess import PreprocessConfig, run_pipeline


def _apply_window(signal: np.ndarray, window: str | bool | None) -> np.ndarray:
    if window is None or window is False:
        return signal
    if not isinstance(window, str):
        return signal  # Skip if not a string
    if window.lower() == "hann":
        w = np.hanning(len(signal))
        return signal * w
    # Fallback: no window
    return signal


def preprocess(signal: np.ndarray, sample_rate_hz: float, *, detrend: bool = True, window: str | None = "hann",
               target_length: int | None = None, resample_rate_hz: float | None = None) -> np.ndarray:
    """Preprocess with optional mean removal, windowing, length normalization, and resampling."""
    cfg = PreprocessConfig(detrend=detrend, window=window,
                           target_length=target_length, resample_rate_hz=resample_rate_hz)
    return run_pipeline(signal, sample_rate_hz, cfg)


def compute_feature_vector(signal: np.ndarray, sample_rate_hz: float, *, detrend: bool = True, window: str | None = "hann",
                           target_length: int | None = None, resample_rate_hz: float | None = None,
                           config: PreprocessConfig | None = None,
                           extra: bool | list[str] = False,
                           top_k_peaks: int = 3) -> np.ndarray:
    """Compute the core feature vector [peak_freq, decay_rate, energy],
    optionally appending additional descriptors when `extra` is True or a list of names.

    Extra names supported: 'spectral_centroid', 'spectral_bandwidth', 'zcr', 'top_peaks', 'ac_lag_s'.
    """
    if config is None:
        x = preprocess(signal, sample_rate_hz, detrend=detrend, window=window,
                       target_length=target_length, resample_rate_hz=resample_rate_hz)
    else:
        x = run_pipeline(signal, sample_rate_hz, config)

    # FFT-based dominant frequency (first half of spectrum)
    fft_vals = np.abs(np.fft.fft(x))
    freqs = np.fft.fftfreq(len(x), 1 / sample_rate_hz)
    half = len(fft_vals) // 2
    peak_idx = np.argmax(fft_vals[:half])
    peak_freq = float(freqs[peak_idx])

    # Envelope-based decay rate (damping proxy)
    envelope = np.abs(x)
    log_env = np.log(envelope + 1e-8)
    decay_rate = float(-np.polyfit(np.arange(len(log_env)), log_env, 1)[0])

    # Signal energy
    energy = float(np.sum(x ** 2))

    base = [peak_freq, decay_rate, energy]

    # Optional extras
    if extra:
        extras_order = (
            'spectral_centroid',
            'spectral_bandwidth',
            'zcr',
            'top_peaks',
            'ac_lag_s',
        )
        if isinstance(extra, list):
            requested = set(extra)
        else:
            requested = set(extras_order)

        # Compute spectrum only once
        mag = np.abs(fft_vals[:half])
        pos_freqs = freqs[:half]
        # Avoid DC bias in peak selection: mask very small frequencies
        non_dc = np.where(np.abs(pos_freqs) > 1e-9)[0]
        mag_ndc = mag[non_dc] if len(non_dc) else mag
        freq_ndc = pos_freqs[non_dc] if len(non_dc) else pos_freqs

        if 'spectral_centroid' in requested:
            msum = np.sum(mag_ndc)
            centroid = float(np.sum(freq_ndc * mag_ndc) / msum) if msum > 0 else 0.0
            base.append(centroid)

        if 'spectral_bandwidth' in requested:
            # Use centroid computed above if available, else recompute local
            msum = np.sum(mag_ndc)
            if msum > 0:
                centroid_local = np.sum(freq_ndc * mag_ndc) / msum
                bandwidth = float(np.sqrt(np.sum(mag_ndc * (freq_ndc - centroid_local) ** 2) / msum))
            else:
                bandwidth = 0.0
            base.append(bandwidth)

        if 'zcr' in requested:
            signs = np.sign(x)
            signs[signs == 0] = 1  # treat zeros as no crossing
            zc = np.sum(signs[1:] != signs[:-1])
            zcr = float(zc / len(x) * sample_rate_hz)
            base.append(zcr)

        if 'top_peaks' in requested:
            k = max(1, int(top_k_peaks))
            if len(mag_ndc) == 0:
                peaks = [0.0] * k
            else:
                # Get indices of k largest magnitudes
                if k >= len(mag_ndc):
                    idxs = np.argsort(mag_ndc)[::-1]
                else:
                    idxs = np.argpartition(mag_ndc, -k)[-k:]
                    idxs = idxs[np.argsort(mag_ndc[idxs])[::-1]]
                peaks = [float(freq_ndc[i]) for i in idxs]
                # If fewer than k peaks, pad with zeros
                if len(peaks) < k:
                    peaks += [0.0] * (k - len(peaks))
            base.extend(peaks)

        if 'ac_lag_s' in requested:
            ac = np.correlate(x, x, mode='full')
            ac = ac[len(x)-1:]  # lags >= 0
            if len(ac) > 1:
                lag_idx = int(np.argmax(ac[1:]) + 1)
                lag_s = float(lag_idx / sample_rate_hz)
            else:
                lag_s = 0.0
            base.append(lag_s)

    return np.array(base, dtype=float)


def compute_features(signal: np.ndarray, sample_rate_hz: float, **kwargs) -> dict:
    """Return a features dict alongside the vector for introspection/logging.
    Includes extra descriptors when requested via `extra`/`top_k_peaks`.
    """
    vec = compute_feature_vector(signal, sample_rate_hz, **kwargs)
    result = {
        "peak_freq": float(vec[0]),
        "decay_rate": float(vec[1]),
        "energy": float(vec[2]),
    }

    extra = kwargs.get('extra', False)
    top_k = int(kwargs.get('top_k_peaks', 3))
    if extra:
        # Determine offsets based on requested extras; recompute indexes matching compute_feature_vector order
        # Recompute auxiliary to know ordering (cheap operations compared to computing spectrum again)
        requested = set(extra) if isinstance(extra, list) else {
            'spectral_centroid', 'spectral_bandwidth', 'zcr', 'top_peaks', 'ac_lag_s'
        }
        idx = 3
        if 'spectral_centroid' in requested:
            result['spectral_centroid'] = float(vec[idx]); idx += 1
        if 'spectral_bandwidth' in requested:
            result['spectral_bandwidth'] = float(vec[idx]); idx += 1
        if 'zcr' in requested:
            result['zcr'] = float(vec[idx]); idx += 1
        if 'top_peaks' in requested:
            for i in range(top_k):
                result[f'peak_freq_{i+1}'] = float(vec[idx + i])
            idx += top_k
        if 'ac_lag_s' in requested:
            result['ac_lag_s'] = float(vec[idx]); idx += 1

    return result