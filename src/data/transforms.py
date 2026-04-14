"""Differential Entropy (DE) feature extraction for EEG signals.

Used by graph-based models (DGCNN, RS-STGCN) that operate on frequency-domain
features rather than raw time-series.

For a Gaussian-distributed signal segment with variance σ²:
    DE = 0.5 * log(2πe * σ²)

We compute DE in 5 standard frequency bands:
    δ (1–4 Hz), θ (4–8 Hz), α (8–14 Hz), β (14–30 Hz), γ (30–40 Hz)

Input:  (C, T) raw EEG — 128 channels, 500 samples at 250 Hz
Output: (C, 5) DE features — 128 channels × 5 frequency bands
"""
import numpy as np
import torch
from scipy.signal import butter, sosfilt


# Frequency bands (Hz) — matches standard EEG neuroscience conventions
FREQ_BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 14.0),
    "beta":  (14.0, 30.0),
    "gamma": (30.0, 40.0),
}

BAND_NAMES = list(FREQ_BANDS.keys())
N_BANDS = len(BAND_NAMES)


def _design_bandpass_filters(
    sfreq: float = 250.0,
    order: int = 5,
) -> list:
    """Pre-compute SOS filter coefficients for each frequency band."""
    filters = []
    nyq = sfreq / 2.0
    for low, high in FREQ_BANDS.values():
        sos = butter(order, [low / nyq, high / nyq], btype="band", output="sos")
        filters.append(sos)
    return filters


class DEFeatureTransform:
    """Callable transform: raw EEG → Differential Entropy features.

    Args:
        sfreq:  Sampling frequency in Hz.
        order:  Butterworth filter order.

    Usage::

        transform = DEFeatureTransform(sfreq=250)
        de = transform(eeg_tensor)   # (128, 500) → (128, 5)
    """

    def __init__(self, sfreq: float = 250.0, order: int = 5) -> None:
        self.sfreq = sfreq
        self._filters = _design_bandpass_filters(sfreq, order)

    def __call__(self, eeg: torch.Tensor) -> torch.Tensor:
        """Compute DE features from raw EEG.

        Args:
            eeg: FloatTensor of shape (C, T).

        Returns:
            FloatTensor of shape (C, N_BANDS).
        """
        x = eeg.numpy()  # (C, T)
        C, T = x.shape
        de = np.empty((C, N_BANDS), dtype=np.float32)

        for b, sos in enumerate(self._filters):
            # Band-pass filter each channel
            filtered = sosfilt(sos, x, axis=-1)          # (C, T)
            # Variance of the filtered signal per channel
            var = np.var(filtered, axis=-1)               # (C,)
            # Clip tiny variances to avoid log(0)
            var = np.clip(var, 1e-12, None)
            # DE = 0.5 * log(2πe * σ²)
            de[:, b] = 0.5 * np.log(2.0 * np.pi * np.e * var)

        return torch.from_numpy(de)

    def __repr__(self) -> str:
        return f"DEFeatureTransform(sfreq={self.sfreq}, bands={BAND_NAMES})"
