"""Demonstration on a synthetic 10-year monthly NDVI series for three wards.

Run: PYTHONPATH=src python -m drought_index.cli
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from drought_index.insurance import season_payouts
from drought_index.vci import classify, vci_series

SEED = 7
YEARS, MONTHS = 10, 12
SHORT_RAINS = [9, 10, 11]  # Oct, Nov, Dec (0-based month index)
HISTORICAL_DROUGHT_DROP = 0.25


def synthetic_ndvi(severity: float, rng: np.random.Generator) -> np.ndarray:
    """Seasonal NDVI with noise, a historical drought in year 4, and a final year
    dried by ``severity`` (0 = normal, 1 = as bad as the historical drought)."""
    t = np.arange(YEARS * MONTHS)
    ndvi = 0.45 + 0.15 * np.sin(2 * np.pi * (t % MONTHS) / MONTHS) + rng.normal(0, 0.02, t.size)
    ndvi[3 * MONTHS : 4 * MONTHS] -= HISTORICAL_DROUGHT_DROP
    ndvi[-MONTHS:] -= severity * HISTORICAL_DROUGHT_DROP
    return ndvi


def main() -> None:
    """Print VCI drought classes and payouts for the last short-rains season."""
    rng = np.random.default_rng(SEED)
    rows = []
    for ward, severity in (("Ward A", 0.0), ("Ward B", 0.8), ("Ward C", 0.95)):
        vci = vci_series(synthetic_ndvi(severity, rng), MONTHS)
        for m in SHORT_RAINS:
            rows.append({"unit": ward, "vci": vci[-MONTHS + m]})
    table = pd.DataFrame(rows)
    payouts = season_payouts(table, {"Ward A": 1e6, "Ward B": 1e6, "Ward C": 1e6})
    payouts["drought_class"] = payouts["seasonal_vci"].map(classify)
    print(payouts.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))


if __name__ == "__main__":
    main()
