from dataclasses import dataclass
import warnings
import numpy as np
from scipy.signal import resample as sp_resample


@dataclass
class PreprocessConfig:
    detrend: bool = True
    window: str | None = "hann"
    target_length: int | None = None
    resample_rate_hz: float | None = None
    length_align: str = "center"  # 'center', 'left', 'right'


def apply_window(x: np.ndarray, window: str | bool | None) -> np.ndarray:
    if window is None or window is False:
        return x
    if not isinstance(window, str):
        return x  # Skip if not a string
    wname = window.lower()
    if wname == "hann":
        w = np.hanning(len(x))
        return x * w
    # Unknown window -> no change
    return x


def detrend_mean(x: np.ndarray) -> np.ndarray:
    return x - np.mean(x)


def resample_signal(x: np.ndarray, sample_rate_hz: float, target_rate_hz: float) -> np.ndarray:
    if target_rate_hz <= 0:
        return x
    # Compute target length proportionally
    target_len = max(1, int(round(len(x) * (target_rate_hz / sample_rate_hz))))
    if target_len == len(x):
        return x
    return sp_resample(x, target_len)


def normalize_length(x: np.ndarray, target_length: int, align: str = "center") -> np.ndarray:
    if target_length is None or target_length <= 0:
        return x
    n = len(x)
    if n == target_length:
        return x
    if n > target_length:
        # Crop
        if align == "left":
            return x[:target_length]
        elif align == "right":
            return x[n - target_length:]
        else:
            start = (n - target_length) // 2
            return x[start:start + target_length]
    else:
        # Pad with zeros
        pad = target_length - n
        if align == "left":
            return np.concatenate([x, np.zeros(pad, dtype=x.dtype)])
        elif align == "right":
            return np.concatenate([np.zeros(pad, dtype=x.dtype), x])
        else:
            left = pad // 2
            right = pad - left
            return np.concatenate([np.zeros(left, dtype=x.dtype), x, np.zeros(right, dtype=x.dtype)])


def run_pipeline(signal: np.ndarray, sample_rate_hz: float, config: PreprocessConfig) -> np.ndarray:
    x = np.asarray(signal, dtype=float)

    if config.detrend:
        x = detrend_mean(x)

    x = apply_window(x, config.window)

    if config.resample_rate_hz is not None:
        if sample_rate_hz < config.resample_rate_hz:
            warnings.warn(
                f"Upsampling from {sample_rate_hz} Hz to {config.resample_rate_hz} Hz; original rate may limit fidelity.",
                RuntimeWarning,
            )
        x = resample_signal(x, sample_rate_hz, config.resample_rate_hz)
        # After resampling, we conceptually adopt the new rate for downstream steps
        sample_rate_hz = config.resample_rate_hz

    if config.target_length is not None:
        x = normalize_length(x, config.target_length, align=config.length_align)

    return x