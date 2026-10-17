import pandas as pd
import pytest

from core.metrics import (
    MetricError, add_period, drop_incomplete_final_period, matched_window, period_series,
)


def _frame(dates, freq="M", values=None):
    df = pd.DataFrame({"order_date": pd.to_datetime(dates)})
    df["order_value"] = 1.0 if values is None else values
    return add_period(df, "order_date", freq)


def test_incomplete_final_month_is_dropped():
    df = _frame(["2010-01-05", "2010-02-10", "2010-03-03"])
    out, label, n = drop_incomplete_final_period(df, "order_date", "period", "M")
    assert label == "2010-03" and n == 1 and len(out) == 2


def test_complete_final_period_is_kept():
    df = _frame(["2010-01-05", "2010-03-31"])
    out, label, n = drop_incomplete_final_period(df, "order_date", "period", "M")
    assert label is None and n == 0 and len(out) == 2
    q = _frame(["2010-10-01", "2010-12-31"], "Q")
    assert drop_incomplete_final_period(q, "order_date", "period", "Q")[1] is None


def test_period_series_means_and_drops_the_partial_period():
    df = _frame(
        ["2010-01-05", "2010-01-20", "2010-02-10", "2010-02-28", "2010-03-03"],
        values=[10, 20, 30, 50, 999],
    )
    series, dropped = period_series(df)
    assert dropped == "2010-03"
    assert series.to_dict() == {"2010-01": 15.0, "2010-02": 40.0}
    series2, dropped2 = period_series(df, drop_incomplete=False)
    assert dropped2 is None and series2["2010-03"] == 999
    with pytest.raises(MetricError):
        period_series(df, how="mode")


def test_matched_window_cuts_the_longer_period_to_the_shorter_span():
    df = _frame(
        ["2010-10-01", "2010-11-15", "2010-11-16", "2010-12-20", "2011-10-01", "2011-11-15"], "Q"
    )
    out, info = matched_window(df, "order_date", "period", "2010Q4", "2011Q4", "Q")
    assert info["limit_days"] == 45
    assert info["baseline_removed"] == 2 and info["comparison_removed"] == 0
    assert len(out) == 4
    assert out["order_date"].max() == pd.Timestamp("2011-11-15")


def test_matched_window_validation():
    df = _frame(["2010-10-01", "2011-10-01"], "Q")
    with pytest.raises(MetricError):
        matched_window(df, "order_date", "period", "2010Q4", "2010Q4", "Q")
    with pytest.raises(MetricError):
        matched_window(df, "order_date", "period", "2010Q4", "2012Q4", "Q")
    with pytest.raises(MetricError):
        matched_window(df, "nope", "period", "2010Q4", "2011Q4", "Q")