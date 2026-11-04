from __future__ import annotations

from collections import defaultdict

import numpy as np

from .models import Measurement


def measurement_statistics(measurements: list[Measurement]) -> list[dict]:
    """Create an always-present statistical report grouped by metric and unit."""
    groups = defaultdict(list)
    for measurement in measurements:
        if np.isfinite(measurement.value):
            groups[(measurement.name, measurement.unit)].append(float(measurement.value))
    rows = []
    for (name, unit), raw in groups.items():
        values = np.asarray(raw, dtype=float)
        mean = float(values.mean())
        std = float(values.std(ddof=1)) if len(values) > 1 else np.nan
        standard_error=float(std/np.sqrt(len(values))) if np.isfinite(std) else np.nan
        rows.append({
            "Metric": name,
            "Unit": unit,
            "N": len(values),
            "Mean": mean,
            "Std": std,
            "3σ": 3 * std if np.isfinite(std) else np.nan,
            "Std error": standard_error,
            "Mean 95% CI ±": 1.96*standard_error if np.isfinite(standard_error) else np.nan,
            "Median": float(np.median(values)),
            "Min": float(values.min()),
            "Max": float(values.max()),
            "Range": float(values.max() - values.min()),
            "CV (%)": float(std / abs(mean) * 100) if np.isfinite(std) and mean else np.nan,
            "Status": ("REVIEW — combined isolated/contact visible segments"
                       if name.startswith("Combined visible segments — ") else
                       ("REVIEW — visible watershed segments; hidden boundaries not reconstructed"
                       if name.startswith("Contact candidate — ") else
                       ("OK" if len(values) > 1 else "N=1; dispersion unavailable"))),
        })
    return rows
