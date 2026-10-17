"""Engine C: hidden cohort and behavioural analysis.

Works on the ORDER-LEVEL frame (one row per sale invoice) with columns
customer_id, order_date, order_value (see core.metrics.order_level_frame).
Orders without a customer ID are excluded and the exclusion is reported.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint

from core.contracts import EvidenceLevel, Finding, FindingCategory
from core.experiments import ProportionTestResult, two_proportion_test


class CohortError(ValueError):
    """Raised when cohort analysis is impossible or inputs are invalid."""


REQUIRED_COLUMNS = ("customer_id", "order_date", "order_value")
FREQS = ("M", "Q", "Y")


# ----------------------------------------------------------------- periods
def _check_freq(freq: str) -> None:
    if freq not in FREQS:
        raise CohortError("freq must be 'M', 'Q' or 'Y'.")


def _period_index(ts: pd.Series, freq: str) -> pd.Series:
    _check_freq(freq)
    if freq == "M":
        return (ts.dt.year * 12 + ts.dt.month - 1).astype(int)
    if freq == "Q":
        return (ts.dt.year * 4 + ts.dt.quarter - 1).astype(int)
    return ts.dt.year.astype(int)


def _label(idx: int, freq: str) -> str:
    idx = int(idx)
    if freq == "M":
        return f"{idx // 12}-{idx % 12 + 1:02d}"
    if freq == "Q":
        return f"{idx // 4}Q{idx % 4 + 1}"
    return str(idx)


def _period_end(idx: int, freq: str) -> pd.Timestamp:
    idx = int(idx)
    if freq == "Y":
        return pd.Timestamp(idx, 12, 31)
    year, month = (idx // 4, (idx % 4) * 3 + 3) if freq == "Q" else (idx // 12, idx % 12 + 1)
    return pd.Timestamp(year, month, 1) + pd.offsets.MonthEnd(0)


# --------------------------------------------------------------- preparing
def prepare_customer_orders(orders: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate required columns, drop orders with no customer ID or no valid date, and
    report what was excluded."""
    if orders is None or len(orders) == 0:
        raise CohortError("The order table is empty.")
    missing = [c for c in REQUIRED_COLUMNS if c not in orders.columns]
    if missing:
        raise CohortError(
            f"Cohort analysis needs {', '.join(REQUIRED_COLUMNS)}; missing: {', '.join(missing)}. "
            "Without customer identifiers and order dates this analysis is disabled."
        )
    has_customer = orders["customer_id"].notna()
    info = {
        "orders_total": int(len(orders)),
        "orders_without_customer": int((~has_customer).sum()),
        "value_total": float(orders["order_value"].sum()),
        "value_without_customer": float(orders.loc[~has_customer, "order_value"].sum()),
    }
    out = orders.loc[has_customer].copy()
    if out.empty:
        raise CohortError("No orders have a customer ID.")
    out["order_date"] = pd.to_datetime(out["order_date"], errors="coerce")
    bad_date = out["order_date"].isna()
    info["orders_bad_date"] = int(bad_date.sum())
    out = out.loc[~bad_date]
    if out.empty:
        raise CohortError("No orders have a valid date.")
    return out, info


# --------------------------------------------------------------- retention
@dataclass
class RetentionResult:
    freq: str
    counts: pd.DataFrame        # rows: cohort label; columns: periods since first order
    rates: pd.DataFrame         # counts / cohort size; NaN where not yet observable
    cohort_sizes: pd.Series
    reliable: pd.Series         # cohort size >= min_cohort_size
    dropped_incomplete_period: str | None
    n_orders_dropped_incomplete: int
    exclusions: dict[str, Any]
    warnings: list[str] = field(default_factory=list)


def cohort_retention(orders: pd.DataFrame, freq: str = "M", min_cohort_size: int = 30,
                     drop_incomplete_period: bool = True) -> RetentionResult:
    """Share of each first-order cohort that orders again k periods later.

    Cells that cannot be observed yet are NaN, never 0. An incomplete final period
    is dropped by default so it cannot masquerade as low retention.
    """
    _check_freq(freq)
    df, info = prepare_customer_orders(orders)
    df = df.copy()
    df["_idx"] = _period_index(df["order_date"], freq)

    dropped_label, n_dropped = None, 0
    if drop_incomplete_period:
        last = int(df["_idx"].max())
        if df["order_date"].max().normalize() < _period_end(last, freq):
            mask = df["_idx"] == last
            n_dropped, dropped_label = int(mask.sum()), _label(last, freq)
            df = df.loc[~mask]
            if df.empty:
                raise CohortError("Nothing is left after dropping the incomplete final period.")

    first = df.groupby("customer_id")["_idx"].min().rename("cohort_idx")
    df = df.join(first, on="customer_id")
    df["offset"] = df["_idx"] - df["cohort_idx"]

    last_idx, cmin, cmax = int(df["_idx"].max()), int(first.min()), int(first.max())
    max_off = last_idx - cmin
    active = (df.groupby(["cohort_idx", "offset"])["customer_id"].nunique().unstack("offset")
              .reindex(index=range(cmin, cmax + 1), columns=range(0, max_off + 1)).fillna(0.0))
    unobservable = np.arange(max_off + 1)[None, :] > (last_idx - np.arange(cmin, cmax + 1))[:, None]
    counts = active.where(~unobservable)
    sizes = active[0].astype(float)
    rates = counts.div(sizes.where(sizes > 0), axis=0)

    labels = [_label(i, freq) for i in range(cmin, cmax + 1)]
    for frame in (counts, rates):
        frame.index = labels
        frame.columns.name = "periods_since_first_order"
    sizes.index = labels
    reliable = sizes >= min_cohort_size

    warnings = [
        f"The first cohort ({labels[0]}) contains every customer already active when the data begins; "
        "'first seen' is not 'first purchase'.",
        "Later cohorts have fewer observable periods (right-censoring); blank cells mean 'not yet observable', not zero.",
    ]
    if dropped_label:
        warnings.append(f"The final period {dropped_label} is incomplete; its {n_dropped:,} orders were dropped.")
    if (~reliable).any():
        warnings.append(f"{int((~reliable).sum())} cohort(s) have fewer than {min_cohort_size} customers.")
    if info["orders_without_customer"]:
        warnings.append(
            f"{info['orders_without_customer']:,} orders ({info['value_without_customer'] / info['value_total']:.1%} "
            "of order value) have no customer ID and are excluded.")
    return RetentionResult(freq, counts, rates, sizes, reliable, dropped_label, n_dropped, info, warnings)


def compare_cohort_retention(result: RetentionResult, cohort_a: str, cohort_b: str, offset: int,
                             alpha: float = 0.05) -> ProportionTestResult:
    """Compare retention at the same offset for two cohorts (a = baseline, b = comparison).
    Different calendar windows mean seasonality is not controlled."""
    for c in (cohort_a, cohort_b):
        if c not in result.counts.index:
            raise CohortError(f"Cohort {c!r} not found.")
    if offset not in result.counts.columns:
        raise CohortError(f"Offset {offset} is outside the observed range.")
    xa, xb = result.counts.loc[cohort_a, offset], result.counts.loc[cohort_b, offset]
    if pd.isna(xa) or pd.isna(xb):
        raise CohortError("That offset is not yet observable for one of the cohorts.")
    na, nb = result.cohort_sizes[cohort_a], result.cohort_sizes[cohort_b]
    if na <= 0 or nb <= 0:
        raise CohortError("A selected cohort is empty.")
    return two_proportion_test(int(round(xa)), int(round(na)), int(round(xb)), int(round(nb)), alpha)


def retention_comparison_finding(test: ProportionTestResult, cohort_a: str, cohort_b: str, offset: int,
                                 data_source: str = "upload") -> Finding:
    return Finding(
        title=f"Retention at +{offset} periods: {cohort_b} cohort {test.rate_treatment:.1%} vs {cohort_a} cohort {test.rate_control:.1%}",
        category=FindingCategory.COHORT_BEHAVIOR,
        evidence_level=EvidenceLevel.STATISTICAL_LEAD if test.significant else EvidenceLevel.DESCRIPTIVE,
        metric="cohort retention rate", comparison=f"{cohort_b} vs {cohort_a} at offset {offset}",
        evidence=asdict_safe(test), method=f"Two-proportion comparison ({test.test_used}); Newcombe interval.",
        n=test.n_control + test.n_treatment,
        alternatives=["The cohorts start in different seasons and customer mixes.",
                      "Acquisition source or customer type may differ between cohorts.",
                      "The earliest cohort contains customers who were already active (left-censoring)."],
        next_step="Check what differed in acquisition between the cohorts, then compare like-for-like seasons.",
        parameters={"cohort_a": cohort_a, "cohort_b": cohort_b, "offset": offset, "alpha": test.alpha},
        data_source=data_source, affected=f"{cohort_a}, {cohort_b}",
        uncertainty={"ci": [test.ci_low, test.ci_high], "alpha": test.alpha},
        limitations=["A single pre-chosen comparison; scanning many cohort pairs inflates false positives."],
    )


def asdict_safe(obj: Any) -> dict[str, Any]:
    from dataclasses import asdict
    return asdict(obj)


# ------------------------------------------------------- repeat purchasing
def repeat_purchase_gaps(orders: pd.DataFrame, distinct_days: bool = True) -> tuple[dict[str, float], pd.Series]:
    """Days between consecutive purchases per customer.

    With distinct_days=True several invoices on the same calendar day count as one
    purchase occasion, so 'repeat' here can differ from metrics.repeat_rate (invoice-based).
    """
    df, _ = prepare_customer_orders(orders)
    d = df[["customer_id", "order_date"]].copy()
    d["day"] = d["order_date"].dt.normalize()
    if distinct_days:
        d = d.drop_duplicates(["customer_id", "day"])
    d = d.sort_values(["customer_id", "day"])
    gaps = d.groupby("customer_id")["day"].diff().dt.days.dropna()
    per_customer = d.groupby("customer_id").size()
    n_customers = int(len(per_customer))
    n_repeat = int((per_customer >= 2).sum())
    summary = {
        "n_customers": n_customers, "n_repeat_customers": n_repeat,
        "repeat_share": n_repeat / n_customers, "n_gaps": int(len(gaps)),
        "median_days": float(gaps.median()) if len(gaps) else float("nan"),
        "p25_days": float(gaps.quantile(0.25)) if len(gaps) else float("nan"),
        "p75_days": float(gaps.quantile(0.75)) if len(gaps) else float("nan"),
        "mean_days": float(gaps.mean()) if len(gaps) else float("nan"),
    }
    return summary, gaps


# -------------------------------------------------- customers and value tiers
def customer_table(orders: pd.DataFrame, as_of: pd.Timestamp | str | None = None) -> pd.DataFrame:
    """One row per customer. value_tier is by revenue rank (top 20% / next 30% / bottom 50%).

    CAUTION: tiers are defined by the same revenue they would be used to explain, so
    segmenting a revenue metric by tier is circular.
    """
    df, _ = prepare_customer_orders(orders)
    as_of_ts = df["order_date"].max().normalize() if as_of is None else pd.Timestamp(as_of).normalize()
    g = df.groupby("customer_id").agg(
        first_order=("order_date", "min"), last_order=("order_date", "max"),
        n_orders=("order_value", "size"), revenue=("order_value", "sum"))
    g["recency_days"] = (as_of_ts - g["last_order"].dt.normalize()).dt.days
    pct = g["revenue"].rank(method="first", ascending=False) / len(g)
    g["value_tier"] = np.select([pct <= 0.2, pct <= 0.5], ["top_20pct", "middle_30pct"], "bottom_50pct")
    return g


# ----------------------------------------------------- revenue concentration
@dataclass
class ConcentrationResult:
    n_customers: int
    total_revenue: float
    top_shares: dict[str, float]
    top_share_ci: dict[str, tuple[float, float]]
    gini: float
    largest_customer_share: float
    n_nonpositive: int
    lorenz: pd.DataFrame
    n_boot: int
    alpha: float


def revenue_concentration(orders: pd.DataFrame, top_fracs: tuple[float, ...] = (0.01, 0.05, 0.10, 0.20),
                          n_boot: int = 500, alpha: float = 0.05, seed: int = 0) -> ConcentrationResult:
    """Share of identified-customer revenue held by the top customers, Gini, and a
    bootstrap interval over customers."""
    if any(not 0 < f <= 1 for f in top_fracs):
        raise CohortError("top_fracs must lie in (0, 1].")
    df, _ = prepare_customer_orders(orders)
    rev = df.groupby("customer_id")["order_value"].sum().to_numpy(float)
    rev = rev[np.isfinite(rev)]
    n = len(rev)
    if n < 2 or rev.sum() <= 0:
        raise CohortError("Need at least two customers and positive total revenue.")

    def shares(desc: np.ndarray) -> dict[str, float]:
        total = desc.sum()
        return {f"top_{int(round(f * 100))}pct":
                float(desc[:max(1, math.ceil(round(f * n, 9)))].sum() / total) for f in top_fracs}

    desc = np.sort(rev)[::-1]
    observed = shares(desc)
    rng = np.random.default_rng(seed)
    boots = {k: np.empty(n_boot) for k in observed}
    for i in range(n_boot):
        sample = np.sort(rng.choice(rev, size=n, replace=True))[::-1]
        for k, v in shares(sample).items():
            boots[k][i] = v
    ci = {k: tuple(float(x) for x in np.percentile(v, [100 * alpha / 2, 100 * (1 - alpha / 2)]))
          for k, v in boots.items()}

    asc = np.sort(rev)
    cum = np.cumsum(asc)
    gini = float((n + 1 - 2 * cum.sum() / cum[-1]) / n)
    lorenz = pd.DataFrame({"customer_share": np.r_[0.0, np.arange(1, n + 1) / n],
                           "revenue_share": np.r_[0.0, cum / cum[-1]]})
    return ConcentrationResult(n, float(rev.sum()), observed, ci, gini, float(desc[0] / rev.sum()),
                               int((rev <= 0).sum()), lorenz, n_boot, alpha)


def concentration_finding(res: ConcentrationResult, data_source: str = "upload") -> Finding:
    top10 = res.top_shares.get("top_10pct", next(iter(res.top_shares.values())))
    return Finding(
        title=f"Top 10% of customers hold {top10:.0%} of identified-customer revenue (Gini {res.gini:.2f})",
        category=FindingCategory.COHORT_BEHAVIOR,
        evidence_level=EvidenceLevel.DESCRIPTIVE,
        metric="revenue concentration", comparison="top customers vs all identified customers",
        evidence={"top_shares": res.top_shares, "top_share_ci": res.top_share_ci, "gini": res.gini,
                  "largest_customer_share": res.largest_customer_share, "n_customers": res.n_customers},
        method="Sort customers by total order value; cumulative share; Gini; bootstrap over customers.",
        n=res.n_customers,
        alternatives=["Heavy concentration is normal where a wholesale segment exists.",
                      "One customer ID may merge several buyers (a reseller) or split one buyer.",
                      "Orders without a customer ID are excluded, so these are shares of identified revenue only."],
        next_step="Check whether the top customers' purchasing is stable over time before treating them as a dependency risk.",
        parameters={"n_boot": res.n_boot, "alpha": res.alpha}, data_source=data_source,
        uncertainty={"bootstrap_ci": res.top_share_ci, "alpha": res.alpha},
        limitations=["Bootstrap over customers ignores their changing behaviour over time."],
    )


# ------------------------------------------------------------------ funnels
def funnel_available(df: pd.DataFrame, user_col: str, stage_col: str, stages: list[str]) -> tuple[bool, str]:
    if df is None or len(df) == 0:
        return False, "Dataset is empty."
    missing = [c for c in (user_col, stage_col) if c not in df.columns]
    if missing:
        return False, (f"No {', '.join(missing)} column. A funnel needs one row per user event with a stage "
                       "label, so this analysis is disabled for this dataset.")
    if len(set(stages)) < 2:
        return False, "Need at least two distinct stages."
    absent = [s for s in stages if s not in set(df[stage_col].dropna().unique())]
    if absent:
        return False, f"Stages not found in '{stage_col}': {absent}."
    return True, ""


def funnel_conversion(events: pd.DataFrame, user_col: str, stage_col: str, stages: list[str],
                      segment_col: str | None = None, alpha: float = 0.05) -> pd.DataFrame:
    """Strict funnel: a user counts at a stage only if they reached every earlier stage
    (event ORDER in time is not checked). Step intervals are Wilson intervals."""
    ok, reason = funnel_available(events, user_col, stage_col, stages)
    if not ok:
        raise CohortError(reason)
    if segment_col is not None and segment_col not in events.columns:
        raise CohortError(f"Segment column '{segment_col}' not found.")
    stages = list(stages)
    cols = [user_col, stage_col] + ([segment_col] if segment_col else [])
    ev = events[cols].dropna(subset=[user_col, stage_col])
    ev = ev[ev[stage_col].isin(stages)]
    presence = (ev.assign(_v=1).pivot_table(index=user_col, columns=stage_col, values="_v",
                                            aggfunc="max", fill_value=0)
                .reindex(columns=stages, fill_value=0))
    reached = presence.cumprod(axis=1)
    if segment_col:
        seg = ev.groupby(user_col)[segment_col].first().reindex(reached.index).fillna("(missing)")
        groups = list(reached.groupby(seg))
    else:
        groups = [("(all)", reached)]
    rows = []
    for name, sub in groups:
        counts = sub.sum()
        for i, stage in enumerate(stages):
            users = int(counts[stage])
            prev = int(counts[stages[i - 1]]) if i > 0 else None
            if prev and prev > 0:
                step = users / prev
                lo, hi = proportion_confint(users, prev, alpha=alpha, method="wilson")
            else:
                step, lo, hi = (float("nan"),) * 3
            first = int(counts[stages[0]])
            rows.append({"segment": name, "stage": stage, "users": users, "step_conversion": step,
                         "step_ci_low": lo, "step_ci_high": hi,
                         "overall_conversion": users / first if first > 0 else float("nan")})
    return pd.DataFrame(rows)