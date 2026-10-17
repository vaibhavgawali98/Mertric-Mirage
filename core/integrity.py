"""Engine B: metric integrity and anomaly investigation.

Vocabulary: a *statistical anomaly* is something these functions can measure
(a gap, a share, a rate). A *confirmed data error* needs a human or a check
against the source system. Nothing here claims the latter.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from core.contracts import EvidenceLevel, Finding, FindingCategory


class IntegrityError(ValueError):
    """Raised for invalid inputs to an integrity check."""


def _nonempty(df: pd.DataFrame) -> None:
    if df is None or len(df) == 0 or len(df.columns) == 0:
        raise IntegrityError("The dataset is empty.")


def _require(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise IntegrityError(f"Columns not found: {missing}")


def _rel_change(old: float, new: float) -> float:
    return float("nan") if (old is None or pd.isna(old) or old == 0) else (new - old) / abs(old)


def _clean_numeric(values) -> pd.Series:
    x = pd.to_numeric(pd.Series(values), errors="coerce").dropna().astype("float64")
    x = x[np.isfinite(x)]
    if x.empty:
        raise IntegrityError("No usable numeric values.")
    return x


# ------------------------------------------------------------- missingness
def missingness_by_column(df: pd.DataFrame) -> pd.DataFrame:
    _nonempty(df)
    out = pd.DataFrame({"column": df.columns, "n_missing": df.isna().sum().to_numpy()})
    out["pct_missing"] = out["n_missing"] / len(df)
    return out.sort_values(["n_missing", "column"], ascending=[False, True]).reset_index(drop=True)


def missingness_by_time(df: pd.DataFrame, date_col: str, column: str, freq: str = "M") -> pd.DataFrame:
    _nonempty(df)
    _require(df, [date_col, column])
    ts = pd.to_datetime(df[date_col], errors="coerce")
    valid = ts.notna()
    if not valid.any():
        raise IntegrityError(f"No valid dates in '{date_col}'.")
    period = ts[valid].dt.to_period(freq).astype(str)
    miss = df.loc[valid, column].isna()
    out = miss.groupby(period).agg(n_rows="size", n_missing="sum").reset_index()
    out.columns = ["period", "n_rows", "n_missing"]
    out["pct_missing"] = out["n_missing"] / out["n_rows"]
    return out


def missingness_by_segment(df: pd.DataFrame, column: str, segment_col: str, min_n: int = 30) -> pd.DataFrame:
    _nonempty(df)
    _require(df, [column, segment_col])
    seg = df[segment_col].astype("object").where(df[segment_col].notna(), "(missing)").astype(str)
    out = df[column].isna().groupby(seg).agg(n_rows="size", n_missing="sum").reset_index()
    out.columns = ["segment", "n_rows", "n_missing"]
    out["pct_missing"] = out["n_missing"] / out["n_rows"]
    out["reliable"] = out["n_rows"] >= min_n
    return out.sort_values("pct_missing", ascending=False).reset_index(drop=True)


def missingness_association(df: pd.DataFrame, column: str, segment_col: str, min_n: int = 30) -> dict[str, Any]:
    """Is missingness in `column` related to `segment_col`? Chi-square plus Cramer's V.
    With big data the p-value is tiny for trivial gaps; read Cramer's V for size."""
    table = missingness_by_segment(df, column, segment_col, min_n)
    table = table[table["reliable"]]
    result = {"testable": False, "chi2": float("nan"), "p_value": float("nan"),
              "cramers_v": float("nan"), "n_segments": int(len(table)), "reason": ""}
    if len(table) < 2:
        result["reason"] = "Fewer than two segments have enough rows."
        return result
    total_missing = int(table["n_missing"].sum())
    if total_missing == 0 or total_missing == int(table["n_rows"].sum()):
        result["reason"] = "The column is never (or always) missing in the testable segments."
        return result
    counts = np.column_stack([table["n_missing"], table["n_rows"] - table["n_missing"]])
    chi2, p, _, _ = stats.chi2_contingency(counts)
    result.update(testable=True, chi2=float(chi2), p_value=float(p),
                  cramers_v=float(math.sqrt(chi2 / counts.sum())))
    return result


# ------------------------------------------------------ duplicates and keys
def duplicate_summary(df: pd.DataFrame, subset: list[str] | None = None) -> dict[str, Any]:
    """Rows repeating an earlier row on `subset` (all columns if None)."""
    _nonempty(df)
    if subset is not None:
        _require(df, subset)
    dup = df.duplicated(subset=subset, keep="first")
    n_dup = int(dup.sum())
    return {
        "n_rows": int(len(df)),
        "n_duplicate_rows": n_dup,
        "pct_duplicate_rows": n_dup / len(df),
        "n_duplicated_groups": int(len(df.loc[dup].drop_duplicates(subset=subset))) if n_dup else 0,
        "subset": list(subset) if subset is not None else list(df.columns),
    }


def key_conflicts(df: pd.DataFrame, key_col: str, attr_cols: list[str]) -> pd.DataFrame:
    """For each attribute, how many keys carry more than one distinct value
    (e.g. an invoice with two customers). Suspicious, not proof of an error."""
    _nonempty(df)
    _require(df, [key_col, *attr_cols])
    n_keys = int(df[key_col].nunique())
    rows = []
    for col in attr_cols:
        conflicting = int((df.groupby(key_col)[col].nunique(dropna=True) > 1).sum())
        rows.append({"attribute": col, "n_keys": n_keys, "n_keys_conflicting": conflicting,
                     "pct_conflicting": conflicting / n_keys if n_keys else float("nan")})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- outliers
@dataclass
class OutlierSummary:
    method: str
    n: int
    n_outliers: int
    share_of_rows: float
    lower: float | None
    upper: float | None
    share_of_total: float          # outliers' share of the summed value
    degenerate: bool               # spread is zero, so fences are meaningless
    mask: pd.Series = field(repr=False, default=None)  # aligned to non-missing values


def _summarise_outliers(x: pd.Series, mask: pd.Series, method: str, lower, upper, degenerate) -> OutlierSummary:
    total = float(x.sum())
    return OutlierSummary(
        method=method, n=int(len(x)), n_outliers=int(mask.sum()),
        share_of_rows=float(mask.mean()), lower=lower, upper=upper,
        share_of_total=float(x[mask].sum() / total) if total != 0 else float("nan"),
        degenerate=degenerate, mask=mask,
    )


def iqr_outliers(values, k: float = 1.5) -> OutlierSummary:
    x = _clean_numeric(values)
    q1, q3 = x.quantile([0.25, 0.75])
    iqr = q3 - q1
    if iqr == 0:
        return _summarise_outliers(x, pd.Series(False, index=x.index), "iqr", None, None, True)
    lower, upper = float(q1 - k * iqr), float(q3 + k * iqr)
    return _summarise_outliers(x, (x < lower) | (x > upper), "iqr", lower, upper, False)


def mad_outliers(values, threshold: float = 3.5) -> OutlierSummary:
    """Modified z-score 0.6745 * (x - median) / MAD, flagged when |score| > threshold."""
    x = _clean_numeric(values)
    med = float(x.median())
    mad = float((x - med).abs().median())
    if mad == 0:
        return _summarise_outliers(x, pd.Series(False, index=x.index), "mad", None, None, True)
    score = 0.6745 * (x - med) / mad
    half_width = threshold * mad / 0.6745
    return _summarise_outliers(x, score.abs() > threshold, "mad", med - half_width, med + half_width, False)


def distribution_summary(values, top_pct: float = 0.01) -> dict[str, float]:
    """Mean-versus-median gap, skewness and how much the largest values carry."""
    x = _clean_numeric(values)
    mean, median = float(x.mean()), float(x.median())
    k = max(1, math.ceil(len(x) * top_pct))
    total = float(x.sum())
    return {
        "n": int(len(x)), "mean": mean, "median": median,
        "mean_median_gap_pct": _rel_change(median, mean),
        "skewness": float(stats.skew(x)) if len(x) > 2 else float("nan"),
        "top_k": int(k),
        "top_k_share_of_total": float(x.nlargest(k).sum() / total) if total != 0 else float("nan"),
    }


# ------------------------------------------- mean driven by extreme values?
@dataclass
class RobustComparison:
    value_col: str
    period_col: str
    baseline: Any
    comparison: Any
    n_baseline: int
    n_comparison: int
    mean_baseline: float
    mean_comparison: float
    median_baseline: float
    median_comparison: float
    trimmed_baseline: float
    trimmed_comparison: float
    rel_change_mean: float
    rel_change_median: float
    rel_change_trimmed: float
    top5_share_comparison: float
    trim: float
    rel_change_threshold: float
    flagged: bool
    reason: str
    warnings: list[str]

    def to_finding(self, data_source: str = "upload") -> Finding | None:
        if not self.flagged:
            return None
        return Finding(
            title=f"Change in mean {self.value_col} is carried by extreme values",
            category=FindingCategory.DISTRIBUTION_ANOMALY,
            evidence_level=EvidenceLevel.STATISTICAL_LEAD,
            metric=f"mean {self.value_col}",
            comparison=f"{self.comparison} vs {self.baseline}",
            evidence={k: getattr(self, k) for k in (
                "mean_baseline", "mean_comparison", "median_baseline", "median_comparison",
                "trimmed_baseline", "trimmed_comparison", "rel_change_mean",
                "rel_change_median", "rel_change_trimmed", "top5_share_comparison")},
            method=f"Mean vs median vs {self.trim:.0%}-trimmed mean per period.",
            n=self.n_baseline + self.n_comparison,
            alternatives=[
                "The large values may be genuine (bulk or wholesale orders) rather than errors.",
                "A single large customer or one-off event may explain them.",
                "A data-entry or unit error in a few rows is possible but not shown by this check.",
            ],
            next_step="Inspect the largest rows in the comparison period and decide whether they belong in this metric.",
            parameters={"value_col": self.value_col, "period_col": self.period_col,
                        "baseline": self.baseline, "comparison": self.comparison,
                        "trim": self.trim, "rel_change_threshold": self.rel_change_threshold},
            data_source=data_source, affected=str(self.comparison),
            limitations=["The thresholds are analyst-set parameters, not statistical constants."],
        )


def compare_periods_robust(
    df: pd.DataFrame, value_col: str, period_col: str, baseline: Any, comparison: Any,
    trim: float = 0.01, rel_change_threshold: float = 0.10,
) -> RobustComparison:
    """Flag when the mean moved by at least `rel_change_threshold` but most of that
    movement disappears once each period's extreme tails are trimmed."""
    _nonempty(df)
    _require(df, [value_col, period_col])
    if not 0 <= trim < 0.5:
        raise IntegrityError("trim must be in [0, 0.5).")
    a = _clean_numeric(df.loc[df[period_col] == baseline, value_col])
    b = _clean_numeric(df.loc[df[period_col] == comparison, value_col])
    ta, tb = float(stats.trim_mean(a, trim)), float(stats.trim_mean(b, trim))
    rel_mean = _rel_change(float(a.mean()), float(b.mean()))
    rel_trim = _rel_change(ta, tb)
    warnings = []
    if min(len(a), len(b)) < 100:
        warnings.append("Fewer than 100 rows in a period: a 1% trim removes nothing, so the check is weak.")
    flagged = bool(
        np.isfinite(rel_mean) and np.isfinite(rel_trim)
        and abs(rel_mean) >= rel_change_threshold and abs(rel_trim) <= 0.5 * abs(rel_mean)
    )
    total_b = float(b.sum())
    reason = (
        f"Mean changed {rel_mean:+.1%} but the trimmed mean changed {rel_trim:+.1%}."
        if np.isfinite(rel_mean) and np.isfinite(rel_trim) else "Relative change undefined (zero baseline)."
    )
    return RobustComparison(
        value_col, period_col, baseline, comparison, len(a), len(b),
        float(a.mean()), float(b.mean()), float(a.median()), float(b.median()), ta, tb,
        rel_mean, _rel_change(float(a.median()), float(b.median())), rel_trim,
        float(b.nlargest(5).sum() / total_b) if total_b != 0 else float("nan"),
        trim, rel_change_threshold, flagged, reason, warnings,
    )


# ------------------------------------------------------- change-point lead
def _best_split(x: np.ndarray, min_segment: int) -> tuple[float, int]:
    """Largest two-sample t-style statistic over all splits; returns (stat, left size)."""
    n = len(x)
    ks = np.arange(min_segment, n - min_segment + 1)
    csum, csq = np.cumsum(x), np.cumsum(x ** 2)
    left_sum, left_sq = csum[ks - 1], csq[ks - 1]
    right_sum, right_sq = csum[-1] - left_sum, csq[-1] - left_sq
    right_n = n - ks
    ssl = left_sq - left_sum ** 2 / ks
    ssr = right_sq - right_sum ** 2 / right_n
    pooled = np.maximum(ssl + ssr, 0) / (n - 2)
    diff = left_sum / ks - right_sum / right_n
    with np.errstate(all="ignore"):
        se = np.sqrt(pooled * (1 / ks + 1 / right_n))
        stat = np.where(se > 0, np.abs(diff) / se, np.where(np.abs(diff) > 0, np.inf, 0.0))
    i = int(np.argmax(stat))
    return float(stat[i]), int(ks[i])


@dataclass
class ChangePointResult:
    n: int
    index: int                 # position of the first point after the split
    label: Any
    statistic: float
    p_value: float
    mean_before: float
    mean_after: float
    n_perm: int
    alpha: float
    note: str = ("Permutation p-value assumes the points are exchangeable under 'no change'; "
                 "trend, seasonality and autocorrelation violate that, so treat this as a candidate.")

    @property
    def significant(self) -> bool:
        return self.p_value < self.alpha

    def to_finding(self, metric: str, data_source: str = "upload") -> Finding | None:
        if not self.significant:
            return None
        return Finding(
            title=f"Candidate level shift in {metric} starting at {self.label}",
            category=FindingCategory.DISTRIBUTION_ANOMALY,
            evidence_level=EvidenceLevel.STATISTICAL_LEAD,
            metric=metric, comparison=f"before vs from {self.label}",
            evidence={"mean_before": self.mean_before, "mean_after": self.mean_after,
                      "statistic": self.statistic, "p_value": self.p_value},
            method=f"Best single mean-shift split with a {self.n_perm}-permutation p-value.",
            n=self.n,
            alternatives=["Seasonality or a trend can look like a level shift.",
                          "A change in measurement, definition or data collection at that date.",
                          "A genuine business change (pricing, campaign, supply)."],
            next_step="Check what changed operationally at that date and whether prior years show the same pattern.",
            parameters={"n_perm": self.n_perm, "alpha": self.alpha}, data_source=data_source,
            affected=str(self.label), limitations=[self.note],
        )


def change_point_candidate(values: pd.Series, min_segment: int = 3, n_perm: int = 999,
                           seed: int = 0, alpha: float = 0.05) -> ChangePointResult:
    """Best single mean-shift split of an ordered series (e.g. a monthly KPI)."""
    if not isinstance(values, pd.Series):
        raise IntegrityError("values must be a pandas Series ordered in time.")
    x = pd.to_numeric(values, errors="coerce").to_numpy(dtype="float64", na_value=np.nan)
    if np.isnan(x).any():
        raise IntegrityError("The series contains missing values; fill or drop them first.")
    if len(x) < 2 * min_segment:
        raise IntegrityError(f"Need at least {2 * min_segment} points, got {len(x)}.")
    observed, k = _best_split(x, min_segment)
    rng = np.random.default_rng(seed)
    exceed = 1
    for _ in range(n_perm):
        if _best_split(rng.permutation(x), min_segment)[0] >= observed - 1e-12:
            exceed += 1
    return ChangePointResult(
        n=len(x), index=k, label=values.index[k], statistic=observed,
        p_value=exceed / (n_perm + 1), mean_before=float(x[:k].mean()),
        mean_after=float(x[k:].mean()), n_perm=n_perm, alpha=alpha,
    )


# ------------------------------------- denominators, grain, leakage warnings
def check_rate_columns(df: pd.DataFrame, numerator: str, denominator: str) -> dict[str, Any]:
    _nonempty(df)
    _require(df, [numerator, denominator])
    num = pd.to_numeric(df[numerator], errors="coerce")
    den = pd.to_numeric(df[denominator], errors="coerce")
    out = {
        "n_rows": int(len(df)),
        "n_missing": int((num.isna() | den.isna()).sum()),
        "n_zero_denominator": int((den == 0).sum()),
        "n_negative_denominator": int((den < 0).sum()),
        "n_numerator_exceeds_denominator": int((num > den).sum()),
        "warnings": [],
    }
    if out["n_zero_denominator"]:
        out["warnings"].append("Zero denominators: the rate is undefined for those rows.")
    if out["n_negative_denominator"]:
        out["warnings"].append("Negative denominators: a rate over them is not meaningful.")
    if out["n_numerator_exceeds_denominator"]:
        out["warnings"].append("Numerator exceeds denominator; this is invalid if the metric is a proportion.")
    return out


def grain_check(df: pd.DataFrame, key_col: str, value_cols: list[str],
                min_multi_row_keys: int = 20, constant_share_threshold: float = 0.9) -> pd.DataFrame:
    """Where a key spans several rows, is a value column constant within the key?
    If so it may be a key-level amount repeated on each row; summing it double counts."""
    _nonempty(df)
    _require(df, [key_col, *value_cols])
    sizes = df.groupby(key_col).size()
    multi = sizes[sizes > 1].index
    sub = df[df[key_col].isin(multi)]
    rows = []
    for col in value_cols:
        share = (float((sub.groupby(key_col)[col].nunique(dropna=False) == 1).mean())
                 if len(multi) else float("nan"))
        suspected = bool(len(multi) >= min_multi_row_keys and share >= constant_share_threshold)
        rows.append({
            "column": col, "n_keys": int(len(sizes)), "n_multi_row_keys": int(len(multi)),
            "share_constant_within_key": share, "suspected_repeated_value": suspected,
            "note": "Possible key-level amount repeated per row; do not sum it across rows." if suspected else "",
        })
    return pd.DataFrame(rows)


def leakage_warnings(df: pd.DataFrame, target_col: str, feature_cols: list[str],
                     corr_threshold: float = 0.95, id_unique_ratio: float = 0.95) -> list[str]:
    """Possible-leakage warnings for a predictive task. Warnings, not verdicts."""
    _nonempty(df)
    _require(df, [target_col, *feature_cols])
    warnings = []
    target_numeric = pd.api.types.is_numeric_dtype(df[target_col])
    for col in feature_cols:
        if col == target_col or df[col].equals(df[target_col]):
            warnings.append(f"'{col}' is identical to the target.")
            continue
        is_numeric = pd.api.types.is_numeric_dtype(df[col])
        is_float = pd.api.types.is_float_dtype(df[col])
        if target_numeric and is_numeric:
            pair = df[[col, target_col]].dropna()
            if len(pair) >= 10 and pair[col].nunique() > 1 and pair[target_col].nunique() > 1:
                r = float(pair[col].corr(pair[target_col]))
                if abs(r) >= corr_threshold:
                    warnings.append(f"'{col}' correlates |r|={abs(r):.3f} with the target; check it is known before the outcome.")
        # Continuous float columns are naturally all-unique, so only non-float columns can look like IDs.
        if not is_float and len(df) and df[col].nunique() / len(df) >= id_unique_ratio:
            warnings.append(f"'{col}' looks like an identifier (almost all values unique); it can let a model memorise rows.")
    return warnings

# -------------------------------------------------------------- finding cards
def duplicate_finding(summary: dict[str, Any], data_source: str = "upload") -> Finding | None:
    if summary["n_duplicate_rows"] == 0:
        return None
    return Finding(
        title=f"{summary['n_duplicate_rows']:,} rows repeat an earlier row ({summary['pct_duplicate_rows']:.2%})",
        category=FindingCategory.DATA_QUALITY,
        evidence_level=EvidenceLevel.DESCRIPTIVE,
        metric="row count", comparison="rows repeating an earlier row on the key columns",
        evidence=dict(summary), method="pandas duplicated(subset=columns, keep='first')",
        n=summary["n_rows"],
        alternatives=["Legitimate repeated scans or identical line items on one order.",
                      "A file that was loaded or appended twice.",
                      "Missing columns in the key that would make the rows distinct."],
        next_step="Compare duplicated rows against the source system before removing anything.",
        parameters={"subset": summary["subset"]}, data_source=data_source,
        limitations=["Duplicate detection depends on which columns form the key."],
    )


def missing_share_finding(column: str, n_missing: int, n_rows: int,
                          association: dict[str, Any] | None = None,
                          data_source: str = "upload") -> Finding | None:
    if n_rows <= 0 or n_missing == 0:
        return None
    evidence: dict[str, Any] = {"n_missing": n_missing, "n_rows": n_rows, "share": n_missing / n_rows}
    if association:
        evidence["association"] = association
    return Finding(
        title=f"'{column}' is missing in {n_missing / n_rows:.1%} of rows",
        category=FindingCategory.DATA_QUALITY,
        evidence_level=EvidenceLevel.DESCRIPTIVE,
        metric=f"missing share of {column}", comparison="missing vs present rows",
        evidence=evidence, method="count of missing values", n=n_rows,
        alternatives=["Missingness may be structural (e.g. guest checkouts) rather than a data fault.",
                      "Missingness may differ by period or segment, biasing any metric that excludes it."],
        next_step="Check how missingness varies by time and segment, and what share of the key metric the missing rows carry.",
        parameters={"column": column}, data_source=data_source, affected=column,
        limitations=["Counts show how much is missing, not why."],
    )