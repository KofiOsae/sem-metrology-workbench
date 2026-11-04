from __future__ import annotations

import numpy as np


def low_pass_three_point(values: np.ndarray) -> np.ndarray:
    """Symmetric [1, 2, 1]/4 low-pass filter with reflected endpoints."""
    values = np.asarray(values, dtype=float)
    if values.size < 3:
        return values.copy()
    return np.convolve(np.pad(values, 1, mode="reflect"), np.array([0.25, 0.5, 0.25]), mode="valid")


def spatial_descriptors(values: np.ndarray, spacing: float) -> tuple[float, float]:
    """Return 1/e autocorrelation length and strongest non-DC wavelength.

    Both outputs use the same lateral unit as ``spacing``. NaN is returned when
    the trace is too short or has no measurable variance.
    """
    trace = np.asarray(values, dtype=float)
    trace = trace[np.isfinite(trace)]
    if trace.size < 8:
        return np.nan, np.nan
    trace = trace - trace.mean()
    variance = float(np.dot(trace, trace))
    if variance <= 1e-18:
        return np.nan, np.nan
    ac = np.correlate(trace, trace, mode="full")[trace.size - 1:]
    ac /= np.maximum(1, np.arange(trace.size, 0, -1))
    ac /= ac[0]
    crossings = np.flatnonzero(ac <= np.exp(-1))
    correlation = float(crossings[0] * spacing) if crossings.size else np.nan
    spectrum = np.abs(np.fft.rfft(trace)) ** 2
    frequencies = np.fft.rfftfreq(trace.size, d=spacing)
    spectrum[0] = 0
    index = int(np.argmax(spectrum))
    wavelength = float(1 / frequencies[index]) if index > 0 and frequencies[index] > 0 else np.nan
    return correlation, wavelength
