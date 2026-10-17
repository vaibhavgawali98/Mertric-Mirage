import numpy as np
import pandas as pd
import pytest

from core.contracts import EvidenceLevel
from core.integrity import (
    IntegrityError, change_point_candidate, check_rate_columns, compare_periods_robust,
    distribution_summary, duplicate_finding, duplicate_summary, grain_check, iqr_outliers,
    key_conflicts, leakage_warnings, mad_outliers, missing_share_finding,
    missingness_association, missingness_by_column, missingness_by_segment, missingness_by_time,
)
from data.synthetic import make_heavy_tail


def test_missingness_by_column():
    df = pd.DataFrame({"a": [1, None, None], "b": [1, 2, 3]})
    out = missingness_by_column(df).set_index("column")
    assert out.loc["a", "n_missing"] == 2 and out.loc["a", "pct_missing"] == pytest.approx(2 / 3)
    assert out.loc["b", "n_missing"] == 0
    with pytest.raises(IntegrityError):
        missingness_by_column(pd.DataFrame())


def test_missingness_by_time():
    df = pd.DataFrame({
        "d": ["2010-01-01"] * 4 + ["2010-02-01"] * 4,
        "v": [1, 2, 3, None, None, None, None, 4],
    })
    out = missingness_by_time(df, "d", "v").set_index("period")
    assert out.loc["2010-01", "pct_missing"] == pytest.approx(0.25)
    assert out.loc["2010-02", "pct_missing"] == pytest.approx(0.75)


def test_missingness_association_detects_and_rejects():
    seg = ["A"] * 100 + ["B"] * 100
    skewed = pd.DataFrame({"seg": seg, "v": [None] * 40 + [1] * 60 + [1] * 100})
    even = pd.DataFrame({"seg": seg, "v": [None] * 10 + [1] * 90 + [None] * 10 + [1] * 90})
    a, b = missingness_association(skewed, "v", "seg"), missingness_association(even, "v", "seg")
    assert a["testable"] and a["p_value"] < 0.001 and a["cramers_v"] > 0.3
    assert b["testable"] and b["p_value"] == pytest.approx(1.0) and b["cramers_v"] == pytest.approx(0.0)
    never = missingness_association(pd.DataFrame({"seg": seg, "v": 1}), "v", "seg")
    assert not never["testable"] and never["reason"]
    assert not missingness_by_segment(skewed, "v", "seg", min_n=1000)["reliable"].any()


def test_duplicate_summary_and_finding():
    df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
    s = duplicate_summary(df)
    assert s["n_duplicate_rows"] == 1 and s["n_duplicated_groups"] == 1
    assert duplicate_summary(df, subset=["a"])["n_duplicate_rows"] == 1
    f = duplicate_finding(s, data_source="synthetic")
    assert f.evidence_level == EvidenceLevel.DESCRIPTIVE and f.alternatives
    assert duplicate_finding(duplicate_summary(pd.DataFrame({"a": [1, 2]}))) is None


def test_key_conflicts():
    df = pd.DataFrame({"inv": ["A", "A", "B", "B"], "date": ["d1", "d2", "d1", "d1"], "cust": [1, 1, 2, 3]})
    out = key_conflicts(df, "inv", ["date", "cust"]).set_index("attribute")
    assert out.loc["date", "n_keys_conflicting"] == 1
    assert out.loc["cust", "pct_conflicting"] == pytest.approx(0.5)


def test_iqr_outliers_flag_only_the_extreme_value():
    x = list(range(1, 21)) + [1000]
    res = iqr_outliers(x)
    assert res.n_outliers == 1 and res.mask.iloc[-1]
    assert res.share_of_total == pytest.approx(1000 / 1210)


def test_mad_outliers_flag_only_the_extreme_value():
    res = mad_outliers(list(range(1, 21)) + [1000])
    assert res.n_outliers == 1 and res.mask.iloc[-1]


def test_degenerate_spread_flags_nothing():
    for fn in (iqr_outliers, mad_outliers):
        res = fn([5.0] * 30)
        assert res.degenerate and res.n_outliers == 0
    with pytest.raises(IntegrityError):
        iqr_outliers([None, None])


def test_distribution_summary_mean_median_gap():
    s = distribution_summary(list(range(1, 21)) + [1000])
    assert s["mean"] > 3 * s["median"] and s["mean_median_gap_pct"] > 2
    assert s["top_k"] == 1 and s["top_k_share_of_total"] == pytest.approx(1000 / 1210)


def test_extreme_values_are_flagged_and_control_is_not():
    df = make_heavy_tail()
    res = compare_periods_robust(df, "order_value", "period", "Q1", "Q2")
    assert res.flagged and res.rel_change_mean > 0.5 and abs(res.rel_change_trimmed) < 0.1
    assert res.top5_share_comparison > 0.3
    assert res.to_finding(data_source="synthetic").is_simulated
    control = compare_periods_robust(df[df["order_value"] != 20000.0], "order_value", "period", "Q1", "Q2")
    assert not control.flagged and control.to_finding() is None


def test_change_point_finds_a_step_and_rejects_a_flat_series():
    noise = np.tile([0, 1, -1], 4)
    step = pd.Series(np.r_[100 + noise, 200 + noise], index=[f"m{i:02d}" for i in range(24)])
    res = change_point_candidate(step)
    assert res.index == 12 and res.label == "m12" and res.significant
    assert res.mean_after - res.mean_before == pytest.approx(100.0)
    assert res.to_finding("monthly AOV") is not None
    flat = pd.Series(100 + np.tile([0, 1, -1], 8))
    assert change_point_candidate(flat).p_value > 0.05
    with pytest.raises(IntegrityError):
        change_point_candidate(pd.Series([1.0, 2.0, 3.0]))
    with pytest.raises(IntegrityError):
        change_point_candidate(pd.Series([1.0, np.nan] * 6))


def test_rate_column_checks():
    df = pd.DataFrame({"n": [1, 5, 2, None], "d": [2, 4, 0, 3]})
    out = check_rate_columns(df, "n", "d")
    assert out["n_zero_denominator"] == 1 and out["n_numerator_exceeds_denominator"] == 2
    assert out["n_missing"] == 1 and out["warnings"]


def test_grain_check_flags_repeated_order_level_values():
    df = pd.DataFrame({
        "order": np.repeat(np.arange(30), 2),
        "order_total": np.repeat(np.arange(30) * 10.0, 2),
        "qty": np.tile([1, 2], 30),
    })
    out = grain_check(df, "order", ["order_total", "qty"]).set_index("column")
    assert out.loc["order_total", "suspected_repeated_value"]
    assert not out.loc["qty", "suspected_repeated_value"]


def test_leakage_warnings():
    rng = np.random.default_rng(0)
    y = rng.normal(size=200)
    df = pd.DataFrame({"y": y, "f1": 2 * y, "f2": rng.normal(size=200), "id": np.arange(200)})
    warnings = leakage_warnings(df, "y", ["f1", "f2", "id"])
    assert any("f1" in w for w in warnings) and any("'id'" in w for w in warnings)
    assert not any("'f2'" in w for w in warnings)
    assert missing_share_finding("Customer ID", 0, 100) is None
    assert missing_share_finding("Customer ID", 22, 100).evidence["share"] == pytest.approx(0.22)

def test_leakage_warnings_handles_text_and_float_features():
    rng = np.random.default_rng(1)
    n = 200
    df = pd.DataFrame({
        "y": rng.normal(size=n),
        "text_id": [f"INV{i}" for i in range(n)],
        "category": rng.choice(["a", "b", "c"], n),
        "noise": rng.normal(size=n),
    })
    warns = leakage_warnings(df, "y", ["text_id", "category", "noise"])
    assert any("'text_id'" in w for w in warns)
    assert not any("'category'" in w for w in warns)
    assert not any("'noise'" in w for w in warns)