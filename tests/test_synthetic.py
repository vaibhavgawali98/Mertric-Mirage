import numpy as np
import pandas as pd
import pytest

from data.synthetic import GROUND_TRUTH, make_ab_test, make_all, make_heavy_tail, make_magnitude_only
from data.synthetic import make_mix_shift, make_null_case, make_simpson


def _rates(df):
    seg = df.groupby(["period", "segment"])["converted"].agg(["sum", "count"])
    agg = df.groupby("period")["converted"].mean()
    return seg, agg


def _decompose(df):
    """Independent midpoint-weight decomposition (used only to check the data)."""
    seg = df.groupby(["period", "segment"])["converted"].agg(["mean", "count"])
    total = df.groupby("period").size()
    mix = rate = 0.0
    for s in df["segment"].unique():
        w0, w1 = seg.loc[("Q1", s), "count"] / total["Q1"], seg.loc[("Q2", s), "count"] / total["Q2"]
        m0, m1 = seg.loc[("Q1", s), "mean"], seg.loc[("Q2", s), "mean"]
        mix += (w1 - w0) * (m0 + m1) / 2
        rate += (w0 + w1) / 2 * (m1 - m0)
    return mix, rate


def test_simpson_exact_counts_and_rates():
    df = make_simpson()
    seg, agg = _rates(df)
    gt = GROUND_TRUTH["simpson"]
    assert agg["Q1"] == pytest.approx(gt["aggregate_rate"]["Q1"])
    assert agg["Q2"] == pytest.approx(gt["aggregate_rate"]["Q2"])
    for (period, segment), expected in gt["segment_rate"].items():
        s = seg.loc[(period, segment)]
        assert s["sum"] / s["count"] == pytest.approx(expected)


def test_simpson_is_a_genuine_reversal():
    df = make_simpson()
    seg, agg = _rates(df)
    assert agg["Q2"] > agg["Q1"]
    for s in ("Returning", "New"):
        rate = lambda p: seg.loc[(p, s), "sum"] / seg.loc[(p, s), "count"]
        assert rate("Q2") < rate("Q1")


@pytest.mark.parametrize("name,maker", [
    ("simpson", make_simpson), ("mix_shift", make_mix_shift), ("magnitude", make_magnitude_only),
])
def test_decomposition_matches_ground_truth(name, maker):
    df = maker()
    gt = GROUND_TRUTH[name]
    mix, rate = _decompose(df)
    assert mix == pytest.approx(gt["mix_effect"], abs=1e-9)
    assert rate == pytest.approx(gt["rate_effect"], abs=1e-9)
    assert mix + rate == pytest.approx(gt["total_change"], abs=1e-9)


def test_null_case_has_no_planted_difference():
    df = make_null_case()
    overall = df["converted"].mean()
    assert 0.13 < overall < 0.17
    assert set(df["segment"]) == {"A", "B"}


def test_heavy_tail_mean_moves_but_median_does_not():
    df = make_heavy_tail()
    q1 = df.loc[df["period"] == "Q1", "order_value"]
    q2 = df.loc[df["period"] == "Q2", "order_value"]
    assert q2.mean() - q1.mean() > 30
    assert abs(q2.median() - q1.median()) < 5
    assert (q2 == GROUND_TRUTH["heavy_tail"]["outlier_value"]).sum() == 5


def test_ab_test_has_exact_counts_and_balanced_device():
    df = make_ab_test()
    gt = GROUND_TRUTH["ab_test"]
    by_arm = df.groupby("arm")["converted"].agg(["sum", "count"])
    assert by_arm.loc["control", "sum"] == gt["control_conversions"]
    assert by_arm.loc["treatment", "sum"] == gt["treatment_conversions"]
    assert (by_arm["count"] == gt["n_per_arm"]).all()
    mobile_share = df.groupby("arm")["device"].apply(lambda s: (s == "mobile").mean())
    assert ((mobile_share > 0.57) & (mobile_share < 0.63)).all()


def test_every_table_is_labelled_simulated_and_reproducible():
    a, b = make_all(seed=7), make_all(seed=7)
    for name in a:
        assert a[name]["is_simulated"].all()
        pd.testing.assert_frame_equal(a[name], b[name])