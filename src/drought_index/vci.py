"""Vegetation Condition Index (VCI) from an NDVI time series.

VCI compares current NDVI with the historical minimum and maximum for the same
period of the year (Kogan, 1995):

    VCI = 100 * (NDVI - NDVI_min) / (NDVI_max - NDVI_min)

Kenya's National Drought Management Authority (NDMA) uses a 3-month average VCI
to classify drought, with the classes in ``DROUGHT_CLASSES``.
"""

from __future__ import annotations

import warnings

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

# (upper bound of VCI, label), following NDMA drought phases.
DROUGHT_CLASSES: tuple[tuple[float, str], ...] = (
    (10.0, "extreme drought"),
    (20.0, "severe drought"),
    (35.0, "moderate drought"),
    (50.0, "normal"),
    (float("inf"), "above normal"),
)
MIN_RANGE = 1e-6  # NDVI range below this is treated as undefined (no variability)


def vci_from_climatology(
    ndvi: FloatArray, ndvi_min: FloatArray, ndvi_max: FloatArray
) -> FloatArray:
    """Compute VCI for an NDVI value or array against its period climatology.

    Args:
        ndvi: Current NDVI (scalar or array).
        ndvi_min: Historical minimum NDVI for the same period, same shape.
        ndvi_max: Historical maximum NDVI for the same period, same shape.

    Returns:
        VCI clipped to 0–100; NaN where inputs are NaN or the range is ~0.
    """
    ndvi = np.asarray(ndvi, dtype=float)
    lo = np.asarray(ndvi_min, dtype=float)
    hi = np.asarray(ndvi_max, dtype=float)
    rng = hi - lo
    with np.errstate(invalid="ignore", divide="ignore"):
        vci = 100.0 * (ndvi - lo) / np.where(rng > MIN_RANGE, rng, np.nan)
    return np.clip(vci, 0.0, 100.0)


def vci_series(ndvi: FloatArray, periods_per_year: int) -> FloatArray:
    """Compute VCI for a regular multi-year NDVI series.

    The climatology for each period of the year (e.g. each month) is taken from all
    years in the series, ignoring NaN (cloud) values. Works on 1-D series (time)
    or stacks shaped (time, rows, cols). Complexity is O(n) in the number of values.

    Args:
        ndvi: Array with time as the first axis; length must be a multiple of
            ``periods_per_year``.
        periods_per_year: 12 for monthly, 36 for dekadal, 23 for 16-day composites.

    Returns:
        VCI array of the same shape as ``ndvi``.

    Raises:
        ValueError: If the series length is not a whole number of years.
    """
    ndvi = np.asarray(ndvi, dtype=float)
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive.")
    if ndvi.shape[0] == 0 or ndvi.shape[0] % periods_per_year:
        raise ValueError("Series length must be a non-zero multiple of periods_per_year.")
    years = ndvi.shape[0] // periods_per_year
    by_period = ndvi.reshape((years, periods_per_year, *ndvi.shape[1:]))
    with warnings.catch_warnings():
        # All-NaN periods (persistent cloud) legitimately yield NaN climatology.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        lo = np.nanmin(by_period, axis=0)
        hi = np.nanmax(by_period, axis=0)
    vci = vci_from_climatology(by_period, lo, hi)
    return vci.reshape(ndvi.shape)


def rolling_mean(values: FloatArray, window: int = 3) -> FloatArray:
    """Trailing mean over ``window`` periods along time (NaN until the window is full)."""
    values = np.asarray(values, dtype=float)
    if window < 1:
        raise ValueError("window must be at least 1.")
    out = np.full(values.shape, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for t in range(window - 1, values.shape[0]):
            out[t] = np.nanmean(values[t - window + 1 : t + 1], axis=0)
    return out


def classify(vci: float) -> str:
    """Return the NDMA drought class for a VCI value ('no data' for NaN)."""
    if np.isnan(vci):
        return "no data"
    for upper, label in DROUGHT_CLASSES:
        if vci < upper:
            return label
    return DROUGHT_CLASSES[-1][1]
