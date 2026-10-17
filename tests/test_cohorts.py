import numpy as np
import pandas as pd
import pytest

from core.cohorts import (
    CohortError, cohort_retention, compare_cohort_retention, concentration_finding,
    customer_table, funnel_available, funnel_conversion, prepare_customer_orders,
    repeat_purchase_gaps, retention_comparison_finding, revenue_concentration,
)


def orders_from(rows):
    df = pd.DataFrame(rows, columns=["customer_id", "order_date", "order_value"])
    df["order_date"] = pd.to_datetime(df["order_date"], format="mixed")
    return df


BASE = [
    ("A", "2010-01-10", 10), ("A", "2010-02-10", 10), ("A", "2010-03-10", 10),
    ("B", "2010-01-15", 10),
    ("C", "2010-02-05", 10), ("C", "2010-04-30", 10),
    ("D", "2010-02-20", 10),
    ("E", "2010-03-15", 10),
]


def test_retention_matrix_matches_hand_calculation():
    res = cohort_retention(orders_from(BASE), min_cohort_size=1)
    assert res.counts.shape == (3, 4)
    assert res.counts.loc["2010-01"].tolist() == [2, 1, 1, 0]
    assert res.counts.loc["2010-02"].iloc[:3].tolist() == [2, 0, 1]
    assert np.isnan(res.counts.loc["2010-02", 3])
    assert res.rates.loc["2010-01"].tolist() == pytest.approx([1.0, 0.5, 0.5, 0.0])
    assert res.rates.loc["2010-03"].iloc[:2].tolist() == pytest.approx([1.0, 0.0])
    assert np.isnan(res.rates.loc["2010-03", 2])
    assert res.dropped_incomplete_period is None
    assert res.cohort_sizes.tolist() == [2, 2, 1]


def test_incomplete_final_period_is_dropped_not_treated_as_zero_retention():
    rows = BASE + [("F", "2010-05-03", 10)]
    res = cohort_retention(orders_from(rows), min_cohort_size=1)
    assert res.dropped_incomplete_period == "2010-05"
    assert res.n_orders_dropped_incomplete == 1
    assert res.counts.shape == (3, 4)
    kept = cohort_retention(orders_from(rows), min_cohort_size=1, drop_incomplete_period=False)
    assert "2010-05" in kept.counts.index


def test_orders_without_customer_are_excluded_and_reported():
    rows = BASE + [(None, "2010-02-11", 99)]
    df, info = prepare_customer_orders(orders_from(rows))
    assert info["orders_without_customer"] == 1 and info["value_without_customer"] == 99
    assert len(df) == len(BASE)


def test_missing_required_columns_disable_the_analysis():
    with pytest.raises(CohortError, match="customer_id"):
        cohort_retention(pd.DataFrame({"order_date": ["2010-01-01"], "order_value": [1.0]}))
    with pytest.raises(CohortError):
        cohort_retention(pd.DataFrame())
    with pytest.raises(CohortError):
        cohort_retention(orders_from(BASE), freq="W")


def test_unobservable_offset_cannot_be_compared():
    res = cohort_retention(orders_from(BASE), min_cohort_size=1)
    with pytest.raises(CohortError):
        compare_cohort_retention(res, "2010-01", "2010-02", 3)
    t = compare_cohort_retention(res, "2010-01", "2010-02", 2)
    assert t.rate_control == pytest.approx(0.5) and t.rate_treatment == pytest.approx(0.5)
    f = retention_comparison_finding(t, "2010-01", "2010-02", 2, data_source="synthetic")
    assert f.is_simulated and f.alternatives


def test_repeat_purchase_gaps():
    rows = [
        (1, "2010-01-01", 5), (1, "2010-01-11", 5), (1, "2010-02-10", 5),
        (2, "2010-01-01", 5),
        (3, "2010-01-01 09:00", 5), (3, "2010-01-01 15:00", 5), (3, "2010-01-06", 5),
    ]
    summary, gaps = repeat_purchase_gaps(orders_from(rows))
    assert summary["n_customers"] == 3 and summary["n_repeat_customers"] == 2
    assert sorted(gaps.tolist()) == [5, 10, 30] and summary["median_days"] == 10
    summary2, gaps2 = repeat_purchase_gaps(orders_from(rows), distinct_days=False)
    assert summary2["n_gaps"] == 4 and summary2["median_days"] == pytest.approx(7.5)


def test_customer_table_tiers():
    rows = [(i, "2010-01-01", 10 - i) for i in range(10)]     # revenue 10, 9, ..., 1
    t = customer_table(orders_from(rows))
    assert t["value_tier"].value_counts().to_dict() == {"bottom_50pct": 5, "middle_30pct": 3, "top_20pct": 2}
    assert t.loc[0, "value_tier"] == "top_20pct" and t.loc[9, "value_tier"] == "bottom_50pct"


def test_concentration_known_values():
    rows = [(i, "2010-01-01", 1) for i in range(9)] + [(9, "2010-01-01", 91)]
    res = revenue_concentration(orders_from(rows), n_boot=50)
    assert res.top_shares["top_10pct"] == pytest.approx(0.91)
    assert res.top_shares["top_20pct"] == pytest.approx(0.92)
    assert res.largest_customer_share == pytest.approx(0.91)
    assert res.gini == pytest.approx(0.81)
    lo, hi = res.top_share_ci["top_10pct"]
    assert 0 < lo <= hi <= 1
    assert res.lorenz["revenue_share"].iloc[-1] == pytest.approx(1.0)
    assert concentration_finding(res).evidence["gini"] == pytest.approx(0.81)


def test_gini_is_zero_for_equal_revenue_and_rejects_bad_input():
    equal = revenue_concentration(orders_from([(i, "2010-01-01", 5) for i in range(8)]), n_boot=10)
    assert equal.gini == pytest.approx(0.0)
    with pytest.raises(CohortError):
        revenue_concentration(orders_from([(1, "2010-01-01", 5)]))
    with pytest.raises(CohortError):
        revenue_concentration(orders_from(BASE), top_fracs=(0.0,))


def test_funnel_counts_and_segments():
    ev = pd.DataFrame({
        "user": [1, 1, 1, 2, 2, 3, 4, 4],
        "stage": ["view", "cart", "buy", "view", "cart", "view", "cart", "buy"],
        "device": ["m", "m", "m", "d", "d", "d", "m", "m"],
    })
    out = funnel_conversion(ev, "user", "stage", ["view", "cart", "buy"])
    assert out["users"].tolist() == [3, 2, 1]      # user 4 never viewed, so never enters the funnel
    assert out["step_conversion"].iloc[1] == pytest.approx(2 / 3)
    assert out["overall_conversion"].iloc[2] == pytest.approx(1 / 3)
    seg = funnel_conversion(ev, "user", "stage", ["view", "cart", "buy"], segment_col="device")
    assert set(seg["segment"]) == {"m", "d"}


def test_funnel_disabled_without_stage_data():
    retail_like = pd.DataFrame({"Invoice": ["1"], "Quantity": [1]})
    ok, reason = funnel_available(retail_like, "Customer ID", "stage", ["a", "b"])
    assert not ok and "disabled" in reason
    with pytest.raises(CohortError):
        funnel_conversion(retail_like, "Customer ID", "stage", ["a", "b"])