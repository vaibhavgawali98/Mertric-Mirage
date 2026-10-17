"""Metric definitions for retail-style transaction data (single source of truth).

All functions expect a frame produced by core.data_loader.classify_retail_rows
(it needs row_class, line_revenue and has_customer). Definitions are written
down in docs/METRICS.md. Zero denominators return NaN, never 0 or inf.
"""
from __future__ import annotations

import pandas as pd

from core.schema import ONLINE_RETAIL_II, DatasetSchema, Role

import numpy as np

from typing import Any


class MetricError(ValueError):
    """Raised when a metric cannot be computed from the given frame."""


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator is None or pd.isna(denominator) or denominator == 0:
        return float("nan")
    return float(numerator) / float(denominator)


def _require_classified(df: pd.DataFrame) -> None:
    missing = [c for c in ("row_class", "line_revenue", "has_customer") if c not in df.columns]
    if missing:
        raise MetricError(
            f"Frame is not classified (missing {missing}). Run classify_retail_rows first."
        )


def _sales(df: pd.DataFrame) -> pd.DataFrame:
    _require_classified(df)
    return df[df["row_class"] == "sale"]


def gross_sales(df: pd.DataFrame) -> float:
    """Sum of line revenue over genuine sale rows (no returns, postage, fees)."""
    return float(_sales(df)["line_revenue"].sum())


def returns_value(df: pd.DataFrame) -> float:
    """Absolute value of returns, as a positive number."""
    _require_classified(df)
    return float(-df.loc[df["row_class"] == "return", "line_revenue"].sum())


def net_revenue(df: pd.DataFrame) -> float:
    return gross_sales(df) - returns_value(df)


def order_count(df: pd.DataFrame, schema: DatasetSchema = ONLINE_RETAIL_II) -> int:
    return int(_sales(df)[schema.mapping[Role.INVOICE_ID]].nunique())


def active_customers(df: pd.DataFrame, schema: DatasetSchema = ONLINE_RETAIL_II) -> int:
    s = _sales(df)
    return int(s.loc[s["has_customer"], schema.mapping[Role.CUSTOMER_ID]].nunique())


def average_order_value(df: pd.DataFrame, schema: DatasetSchema = ONLINE_RETAIL_II) -> float:
    return safe_divide(gross_sales(df), order_count(df, schema))


def revenue_per_customer(df: pd.DataFrame, schema: DatasetSchema = ONLINE_RETAIL_II) -> float:
    """Revenue from rows WITH a customer ID, divided by distinct customers."""
    s = _sales(df)
    with_id = float(s.loc[s["has_customer"], "line_revenue"].sum())
    return safe_divide(with_id, active_customers(df, schema))


def return_rate_value(df: pd.DataFrame) -> float:
    return safe_divide(returns_value(df), gross_sales(df))


def repeat_rate(df: pd.DataFrame, schema: DatasetSchema = ONLINE_RETAIL_II) -> float:
    """Customers with >= 2 distinct orders / customers with >= 1 order.
    Right-censored: recent customers have had less time to come back."""
    s = _sales(df)
    s = s[s["has_customer"]]
    orders_per_customer = s.groupby(schema.mapping[Role.CUSTOMER_ID])[
        schema.mapping[Role.INVOICE_ID]
    ].nunique()
    return safe_divide(int((orders_per_customer >= 2).sum()), len(orders_per_customer))


def metric_summary(df: pd.DataFrame, schema: DatasetSchema = ONLINE_RETAIL_II) -> dict[str, float]:
    return {
        "gross_sales": gross_sales(df),
        "returns_value": returns_value(df),
        "net_revenue": net_revenue(df),
        "orders": order_count(df, schema),
        "active_customers": active_customers(df, schema),
        "average_order_value": average_order_value(df, schema),
        "revenue_per_customer": revenue_per_customer(df, schema),
        "return_rate_value": return_rate_value(df),
        "repeat_rate": repeat_rate(df, schema),
    }


def add_period(
    df: pd.DataFrame, date_col: str, freq: str = "M", out_col: str = "period"
) -> pd.DataFrame:
    """Return a copy with a text period label ('2010-11' for M, '2010Q4' for Q, '2010' for Y)."""
    if date_col not in df.columns:
        raise MetricError(f"Column '{date_col}' not found.")
    if freq not in ("M", "Q", "Y"):
        raise MetricError("freq must be 'M', 'Q' or 'Y'.")
    ts = pd.to_datetime(df[date_col], errors="coerce")
    if ts.notna().sum() == 0:
        raise MetricError(f"No valid dates in '{date_col}'.")
    period = ts.dt.to_period(freq)
    out = df.copy()
    out[out_col] = period.astype(str).where(period.notna())
    return out


def order_level_frame(
    df: pd.DataFrame, schema: DatasetSchema = ONLINE_RETAIL_II, freq: str = "M"
) -> pd.DataFrame:
    """One row per genuine sale invoice: the grain needed for order-value metrics.

    Columns: invoice_id, order_value, n_lines, order_date, customer_id,
    <segment columns>, period.
    """
    s = _sales(df)
    m = schema.mapping
    agg = {
        "order_value": ("line_revenue", "sum"),
        "n_lines": ("line_revenue", "size"),
        "order_date": (m[Role.TIMESTAMP], "min"),
        "customer_id": (m[Role.CUSTOMER_ID], "first"),
    }
    for col in schema.segment_columns:
        if col not in s.columns:
            raise MetricError(f"Segment column '{col}' not found.")
        agg[col] = (col, "first")
    orders = s.groupby(m[Role.INVOICE_ID]).agg(**agg)
    orders.index.name = "invoice_id"
    return add_period(orders.reset_index(), "order_date", freq)

# --------------------------------------------------------------- period helpers
def _period_obj(label: Any, freq: str) -> pd.Period:
    if freq not in ("M", "Q", "Y"):
        raise MetricError("freq must be 'M', 'Q' or 'Y'.")
    try:
        return pd.Period(str(label), freq=freq)
    except (ValueError, TypeError) as exc:
        raise MetricError(f"Cannot read {label!r} as a '{freq}' period label.") from exc


def drop_incomplete_final_period(
    df: pd.DataFrame, date_col: str, period_col: str, freq: str
) -> tuple[pd.DataFrame, str | None, int]:
    """Drop the last period's rows when the data stops before that period ends.

    Returns (frame, dropped_label or None, rows_dropped). A half-observed final
    month would otherwise look like a collapse in totals and a distorted mean.
    """
    if df is None or len(df) == 0:
        raise MetricError("The dataset is empty.")
    for col in (date_col, period_col):
        if col not in df.columns:
            raise MetricError(f"Column '{col}' not found.")
    ts = pd.to_datetime(df[date_col], errors="coerce")
    if ts.notna().sum() == 0:
        raise MetricError(f"No valid dates in '{date_col}'.")
    labels = df[period_col].dropna()
    if labels.empty:
        raise MetricError(f"No period labels in '{period_col}'.")
    last_label = labels.max()  # zero-padded labels sort chronologically
    period = _period_obj(last_label, freq)
    if ts.max().normalize() >= period.end_time.normalize():
        return df, None, 0
    mask = (df[period_col] == last_label).to_numpy()
    return df.loc[~mask], str(last_label), int(mask.sum())


def period_series(
    orders: pd.DataFrame,
    value_col: str = "order_value",
    period_col: str = "period",
    date_col: str = "order_date",
    freq: str = "M",
    how: str = "mean",
    drop_incomplete: bool = True,
) -> tuple[pd.Series, str | None]:
    """A KPI per period as an ordered Series (e.g. input for a change-point check).
    Returns (series, label of the incomplete period that was dropped, if any)."""
    if how not in ("mean", "sum", "median"):
        raise MetricError("how must be 'mean', 'sum' or 'median'.")
    if value_col not in orders.columns:
        raise MetricError(f"Column '{value_col}' not found.")
    work, dropped = orders, None
    if drop_incomplete:
        work, dropped, _ = drop_incomplete_final_period(orders, date_col, period_col, freq)
    series = work.groupby(period_col)[value_col].agg(how).sort_index()
    if series.empty:
        raise MetricError("No data left to build the series.")
    return series, dropped


def matched_window(
    df: pd.DataFrame, date_col: str, period_col: str, baseline: Any, comparison: Any, freq: str
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Make two periods comparable when one is only partly observed.

    Both periods are cut to the same number of elapsed days since their own start,
    using the shorter observed span. Example: 2011Q4 observed only to 9 Dec is
    compared with 2010Q4 up to 9 Dec. Weekday mix and holiday timing are NOT
    controlled. Rows outside the two periods are returned untouched.
    """
    if baseline == comparison:
        raise MetricError("Baseline and comparison periods must differ.")
    for col in (date_col, period_col):
        if col not in df.columns:
            raise MetricError(f"Column '{col}' not found.")
    ts = pd.to_datetime(df[date_col], errors="coerce")
    starts: dict[Any, pd.Timestamp] = {}
    spans: dict[Any, int] = {}
    for label in (baseline, comparison):
        in_period = (df[period_col] == label).to_numpy()
        if not in_period.any():
            raise MetricError(f"Period {label!r} has no rows.")
        start = _period_obj(label, freq).start_time.normalize()
        starts[label] = start
        spans[label] = int((ts[in_period].max().normalize() - start).days)
    limit = min(spans.values())

    keep = np.ones(len(df), dtype=bool)
    removed: dict[Any, int] = {}
    for label in (baseline, comparison):
        in_period = (df[period_col] == label).to_numpy()
        elapsed = (ts - starts[label]).dt.days
        too_late = in_period & (elapsed > limit).to_numpy()
        keep &= ~too_late
        removed[label] = int(too_late.sum())
    info = {
        "limit_days": limit,
        "span_baseline": spans[baseline],
        "span_comparison": spans[comparison],
        "baseline_removed": removed[baseline],
        "comparison_removed": removed[comparison],
    }
    return df.loc[keep], info