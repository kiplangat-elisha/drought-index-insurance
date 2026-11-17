"""Index-insurance payout rule driven by seasonal mean VCI.

Payout rises linearly from 0% at the trigger to 100% at the exit level:

    payout_share = clip((trigger - VCI) / (trigger - exit), 0, 1)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DEFAULT_TRIGGER_VCI = 35.0  # start of moderate drought (NDMA)
DEFAULT_EXIT_VCI = 10.0  # extreme drought: full payout


@dataclass(frozen=True)
class PayoutRule:
    """Linear payout between a trigger and an exit VCI."""

    trigger: float = DEFAULT_TRIGGER_VCI
    exit: float = DEFAULT_EXIT_VCI

    def __post_init__(self) -> None:
        if not self.trigger > self.exit:
            raise ValueError("trigger must be greater than exit.")

    def share(self, seasonal_vci: float) -> float:
        """Share of the sum insured paid for a seasonal mean VCI (0 for NaN)."""
        if np.isnan(seasonal_vci):
            return 0.0
        return float(np.clip((self.trigger - seasonal_vci) / (self.trigger - self.exit), 0, 1))


def season_payouts(
    vci_table: pd.DataFrame, sum_insured: dict[str, float], rule: PayoutRule | None = None
) -> pd.DataFrame:
    """Compute payouts per unit area (e.g. ward) for one season.

    Args:
        vci_table: Columns ``unit`` and ``vci`` with one row per period of the season.
        sum_insured: Sum insured (KES) per unit.
        rule: Payout rule; defaults to the NDMA-based trigger and exit.

    Returns:
        One row per unit with seasonal VCI, payout share and payout amount.
    """
    rule = rule or PayoutRule()
    seasonal = vci_table.groupby("unit", as_index=False)["vci"].mean()
    seasonal = seasonal.rename(columns={"vci": "seasonal_vci"})
    seasonal["payout_share"] = seasonal["seasonal_vci"].map(rule.share)
    seasonal["sum_insured_kes"] = seasonal["unit"].map(sum_insured).fillna(0.0)
    seasonal["payout_kes"] = (seasonal["payout_share"] * seasonal["sum_insured_kes"]).round(0)
    return seasonal.sort_values("payout_share", ascending=False).reset_index(drop=True)
