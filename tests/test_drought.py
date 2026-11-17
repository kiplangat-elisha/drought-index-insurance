import numpy as np
import pandas as pd
import pytest

from drought_index.insurance import PayoutRule, season_payouts
from drought_index.vci import classify, rolling_mean, vci_from_climatology, vci_series


def test_vci_basic_and_clip():
    assert vci_from_climatology(0.5, 0.0, 1.0) == pytest.approx(50.0)
    assert vci_from_climatology(1.5, 0.0, 1.0) == pytest.approx(100.0)
    assert vci_from_climatology(-1.0, 0.0, 1.0) == pytest.approx(0.0)


def test_zero_range_is_nan():
    assert np.isnan(vci_from_climatology(0.3, 0.3, 0.3))


def test_series_min_and_max_years():
    s = np.array([0.2, 0.4, 0.6, 0.8])  # 4 years x 1 period
    v = vci_series(s, periods_per_year=1)
    assert v[0] == pytest.approx(0.0)
    assert v[-1] == pytest.approx(100.0)


def test_series_stack_shape_and_nan():
    stack = np.random.default_rng(0).random((24, 3, 3))
    stack[5, 1, 1] = np.nan
    v = vci_series(stack, 12)
    assert v.shape == stack.shape
    assert np.isnan(v[5, 1, 1])


def test_series_bad_length():
    with pytest.raises(ValueError):
        vci_series(np.ones(13), 12)


def test_rolling_mean():
    r = rolling_mean(np.array([1.0, 2.0, 3.0, 4.0]), 3)
    assert np.isnan(r[0]) and r[2] == pytest.approx(2.0) and r[3] == pytest.approx(3.0)


@pytest.mark.parametrize("v,label", [(5, "extreme drought"), (15, "severe drought"),
                                     (30, "moderate drought"), (40, "normal"),
                                     (80, "above normal"), (float("nan"), "no data")])
def test_classes(v, label):
    assert classify(v) == label


def test_payout_rule():
    r = PayoutRule(35, 10)
    assert r.share(40) == 0.0
    assert r.share(22.5) == pytest.approx(0.5)
    assert r.share(5) == 1.0
    assert r.share(float("nan")) == 0.0
    with pytest.raises(ValueError):
        PayoutRule(10, 35)


def test_season_payouts():
    t = pd.DataFrame({"unit": ["A", "A", "B", "B"], "vci": [20, 25, 60, 50]})
    out = season_payouts(t, {"A": 100_000, "B": 100_000}).set_index("unit")
    assert out.loc["A", "payout_kes"] == pytest.approx(50_000)
    assert out.loc["B", "payout_kes"] == 0
