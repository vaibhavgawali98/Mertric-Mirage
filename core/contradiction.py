"""Engine A: aggregate-versus-segment contradiction detector.

The metric is M = sum(numerator) / sum(denominator) (a plain mean or rate when
there is no denominator). Writing it as a weighted average over segments,

    M = sum_k w_k * m_k,   w_k = den_k / den_total,   m_k = num_k / den_k,

the change between two periods splits EXACTLY (midpoint weights) into

    mix effect  = sum_k (w1_k - w0_k) * (m0_k + m1_k) / 2
    rate effect = sum_k (w0_k + w1_k) / 2 * (m1_k - m0_k)

plus an entry/exit term for segments present in only one period. The verdict
is derived from the signs of these pieces and of each segment's change, so a
difference in magnitude is never mislabelled as Simpson's paradox.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

from core.contracts import EvidenceLevel, Finding, FindingCategory

TOL = 1e-12
MISSING_LABEL = "(missing)"
REVERSAL_VERDICTS = {"simpson_reversal", "partial_reversal"}
NOTABLE_VERDICTS = REVERSAL_VERDICTS | {"mix_driven", "hidden_movement"}


class ContradictionError(ValueError):
    """Raised for invalid inputs to the contradiction analysis."""


@dataclass(frozen=True)
class MetricSpec:
    """A metric defined as sum(numerator) / sum(denominator).

    With denominator=None it is the plain mean of the numerator column
    (a conversion rate when that column is 0/1).
    """

    name: str
    numerator: str
    denominator: str | None = None


# ------------------------------------------------------------------ helpers
def _sign(x: float, tol: float = TOL) -> int:
    if x is None or np.isnan(x):
        return 0
    return 1 if x > tol else (-1 if x < -tol else 0)


def _inference(delta: np.ndarray, se: np.ndarray, alpha: float):
    """Normal-approximation CI and two-sided p-value for each delta."""
    z_crit = norm.ppf(1 - alpha / 2)
    with np.errstate(all="ignore"):
        z = delta / se
        p = np.where(
            se > 0,
            2 * norm.sf(np.abs(z)),
            np.where(np.abs(delta) > TOL, 0.0, 1.0),
        )
        p = np.where(np.isnan(se) | np.isnan(delta), np.nan, p)
        return delta - z_crit * se, delta + z_crit * se, p


def _cell_stats(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Per-group n, sums, ratio value and delta-method standard error.

    For R = sum(x)/sum(y), Var(R) ~ n/(n-1) * sum((x_i - R*y_i)^2) / sum(y)^2.
    With y = 1 this is exactly s / sqrt(n).
    """
    cell = frame.groupby(group_cols, sort=True, observed=True).agg(
        n=("_x", "size"), x_sum=("_x", "sum"), y_sum=("_y", "sum")
    )
    positive_y = cell["y_sum"].where(cell["y_sum"] > 0)
    cell["value"] = cell["x_sum"] / positive_y

    joined = frame.merge(cell[["value"]].reset_index(), on=group_cols, how="left")
    resid_sq = (joined["_x"] - joined["value"] * joined["_y"]) ** 2
    cell["sse"] = resid_sq.groupby([joined[c] for c in group_cols]).sum()

    n = cell["n"].astype(float)
    factor = np.where(n > 1, n / (n - 1), np.nan)
    cell["se"] = np.sqrt(factor * cell["sse"]) / positive_y
    return cell


def _validate_inputs(df, metric, period_col, baseline, comparison, segment_col,
                     min_segment_n, min_coverage, alpha, flat_rel_tol) -> None:
    if df is None or len(df) == 0:
        raise ContradictionError("The dataset is empty.")
    needed = [period_col, segment_col, metric.numerator]
    if metric.denominator:
        needed.append(metric.denominator)
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ContradictionError(f"Columns not found: {missing}")
    if period_col == segment_col:
        raise ContradictionError("The period column and the segment column must differ.")
    if baseline == comparison:
        raise ContradictionError("Baseline and comparison periods must differ.")
    for col in (metric.numerator, metric.denominator):
        if col is not None and not pd.api.types.is_numeric_dtype(df[col]):
            raise ContradictionError(f"Column '{col}' must be numeric.")
    if min_segment_n < 2:
        raise ContradictionError("min_segment_n must be at least 2.")
    if not 0 < alpha < 1:
        raise ContradictionError("alpha must be between 0 and 1.")
    if not 0 <= min_coverage <= 1:
        raise ContradictionError("min_coverage must be between 0 and 1.")
    if flat_rel_tol < 0:
        raise ContradictionError("flat_rel_tol cannot be negative.")


def _prepare_rows(df, metric, period_col, baseline, comparison, segment_col):
    """Keep the two periods, label missing segments, drop unusable rows (counted)."""
    scoped = df.loc[df[period_col].isin([baseline, comparison]).to_numpy()]
    if scoped.empty:
        raise ContradictionError(
            f"No rows found for {baseline!r} or {comparison!r} in column '{period_col}'."
        )
    x = pd.to_numeric(scoped[metric.numerator], errors="coerce").to_numpy(
        dtype="float64", na_value=np.nan
    )
    if metric.denominator:
        y = pd.to_numeric(scoped[metric.denominator], errors="coerce").to_numpy(
            dtype="float64", na_value=np.nan
        )
    else:
        y = np.ones(len(scoped))

    seg_raw = scoped[segment_col]
    seg_missing = seg_raw.isna().to_numpy()
    seg = seg_raw.astype("object").where(seg_raw.notna(), MISSING_LABEL).astype(str).to_numpy()

    valid = ~np.isnan(x) & ~np.isnan(y)
    if np.any(y[valid] < 0):
        raise ContradictionError("The denominator column contains negative values.")

    missing = {
        "rows_in_periods": int(len(scoped)),
        "rows_missing_numerator": int(np.isnan(x).sum()),
        "rows_missing_denominator": int(np.isnan(y).sum()),
        "rows_missing_segment": int(seg_missing[valid].sum()),
        "rows_used": int(valid.sum()),
    }
    frame = pd.DataFrame(
        {"_p": scoped[period_col].to_numpy(), "_s": seg, "_x": x, "_y": y}
    )[valid]
    for period in (baseline, comparison):
        if not (frame["_p"] == period).any():
            raise ContradictionError(f"Period {period!r} has no usable rows.")
    return frame, missing


def _classify(delta, base_value, mix, rate, dirs, sig, eligible, coverage,
              min_coverage, flat_rel_tol):
    """Return (verdict, n_opposing, n_opposing_significant)."""
    n_el = int(eligible.sum())
    if n_el < 2 or coverage < min_coverage:
        return "insufficient_data", 0, 0

    d, s = dirs[eligible], sig[eligible]
    agg_dir, mix_dir, rate_dir = _sign(delta), _sign(mix), _sign(rate)

    if abs(delta) <= max(flat_rel_tol * abs(base_value), TOL):
        return ("hidden_movement" if s.any() else "consistent"), 0, 0

    opposing = d == -agg_dir
    n_opp, n_opp_sig = int(opposing.sum()), int((opposing & s).sum())

    if mix_dir == agg_dir and rate_dir == -agg_dir:
        if n_opp == n_el:
            return "simpson_reversal", n_opp, n_opp_sig
        return "partial_reversal", n_opp, n_opp_sig
    if mix_dir == agg_dir and abs(mix) > abs(rate):
        return "mix_driven", n_opp, n_opp_sig
    if np.all((d == agg_dir) | (d == 0)):
        return "consistent", n_opp, n_opp_sig
    return "heterogeneous", n_opp, n_opp_sig


# ------------------------------------------------------------------ result
_ALTERNATIVES = {
    "simpson_reversal": [
        "Sampling noise: the segment-level changes may lie within random variation (see the confidence intervals).",
        "The segment label may itself change over time (who counts as 'new' or 'returning'), so segments are not comparable across periods.",
        "Seasonality or an external event may affect both the mix and the metric.",
        "The metric or the segmentation may have been measured differently in the two periods.",
    ],
    "partial_reversal": [
        "Sampling noise in the segments that moved against the overall direction.",
        "Seasonality or an external event shifting both the mix and the metric.",
        "A change in how segments are assigned between periods.",
    ],
    "mix_driven": [
        "The change in mix may be intentional (a campaign, a pricing change) rather than a data artefact.",
        "Seasonality may drive which segments are active in each period.",
        "Segment definitions may differ between periods.",
    ],
    "hidden_movement": [
        "With many segments, some will move by chance; the FDR correction limits but does not remove this.",
        "Offsetting movements may reflect a reassignment of customers between segments.",
        "Seasonality may affect segments differently.",
    ],
}
_NEXT_STEPS = {
    "simpson_reversal": "Check whether the segments are stable definitions, then look for the driver of the mix shift (acquisition channel, season, campaign).",
    "partial_reversal": "Inspect the segments that moved against the overall direction and what changed in the mix.",
    "mix_driven": "Find what changed the composition and whether like-for-like (fixed-mix) comparison tells a different story.",
    "hidden_movement": "Investigate the segments flagged as significant and check whether they are offsetting each other.",
}


def _explain(verdict, name, b, c, agg, mix, rate, entry, n_el, n_opp, n_opp_sig,
             coverage, min_coverage, flat_rel_tol) -> str:
    delta = agg["change"]
    text = (
        f"Overall {name} moved from {agg['value_baseline']:.4g} ({b}) to "
        f"{agg['value_comparison']:.4g} ({c}), a change of {delta:+.4g}. "
        "Writing the overall value as a weighted average of segment values, "
        "M = sum(w_k * m_k) with w_k the segment's share of the denominator, "
        f"the change splits exactly into a mix effect of {mix:+.4g} (shifts in w_k, valued at "
        f"average segment levels) and a rate effect of {rate:+.4g} (changes in m_k, weighted at "
        "average shares)"
    )
    if abs(entry) > TOL:
        text += f", plus {entry:+.4g} from segments present in only one period"
    text += ". "
    mix_share = mix / delta if abs(delta) > TOL else float("nan")

    if verdict == "insufficient_data":
        text += (
            f"Only {n_el} segment(s) have enough rows in both periods (coverage {coverage:.0%} of "
            f"weight, minimum {min_coverage:.0%}), so no aggregate-versus-segment claim is made."
        )
    elif verdict == "simpson_reversal":
        text += (
            f"All {n_el} evaluable segments moved opposite to the overall change, yet the overall "
            "value moved because composition shifted: the mix effect outweighs the opposing rate "
            "effect. This is the arithmetic structure of Simpson's paradox. "
        )
        text += (
            "None of the opposing segment changes is statistically distinguishable from zero "
            "(FDR-adjusted), so it is an arithmetic fact about these samples, not yet a "
            "demonstrated within-segment decline."
            if n_opp_sig == 0
            else f"{n_opp_sig} of {n_opp} opposing segment changes are statistically "
            "distinguishable from zero (FDR-adjusted)."
        )
    elif verdict == "partial_reversal":
        text += (
            f"The weighted-average within-segment change ({rate:+.4g}) points opposite to the "
            f"overall change, which comes from the shift in composition. {n_opp} of {n_el} "
            f"evaluable segments moved opposite to the overall change ({n_opp_sig} significantly)."
        )
    elif verdict == "mix_driven":
        text += (
            f"Composition accounts for {mix_share:.0%} of the change; within-segment movement is "
            "smaller. No reversal, but the headline change is mostly a change in who is in the data."
        )
    elif verdict == "hidden_movement":
        text += (
            f"The overall change is within {flat_rel_tol:.0%} of the baseline, but at least one "
            "segment changed significantly (FDR-adjusted): the aggregate hides movement."
        )
    elif verdict == "heterogeneous":
        text += (
            "Segments move in different directions and the average within-segment movement agrees "
            "with the overall direction. No aggregation reversal."
        )
    else:
        text += (
            "Overall and segment movements agree in direction and composition explains less than "
            "half of the change. No aggregation paradox detected."
        )
    return text


@dataclass
class ContradictionResult:
    metric_name: str
    config: dict[str, Any]
    data_source: str
    aggregate: dict[str, float]
    segments: pd.DataFrame
    mix_effect: float
    rate_effect: float
    entry_exit_effect: float
    verdict: str
    n_eligible: int
    n_opposing: int
    n_opposing_significant: int
    coverage: float
    rows_used: int
    missing: dict[str, int]
    warnings: list[str]
    explanation: str

    @property
    def is_notable(self) -> bool:
        return self.verdict in NOTABLE_VERDICTS

    @property
    def mix_share_of_change(self) -> float:
        change = self.aggregate["change"]
        return self.mix_effect / change if abs(change) > TOL else float("nan")

    def to_finding(self) -> Finding | None:
        """An investigation card for notable verdicts; None means 'nothing to report'."""
        if not self.is_notable:
            return None
        agg, cfg = self.aggregate, self.config
        agg_sig = bool(agg["p_value"] < cfg["alpha"])
        if self.verdict in REVERSAL_VERDICTS:
            lead = agg_sig and self.n_opposing_significant > 0
        elif self.verdict == "hidden_movement":
            lead = True
        else:
            lead = agg_sig

        seg, name = cfg["segment_col"], self.metric_name
        went_up = agg["change"] > 0
        word, opp = ("rose", "fell") if went_up else ("fell", "rose")
        titles = {
            "simpson_reversal": f"{name} {word} overall but {opp} in every evaluable {seg} segment",
            "partial_reversal": f"{name} {word} overall, but within-segment movement points the other way",
            "mix_driven": f"Change in {name} is mostly a shift in {seg} mix",
            "hidden_movement": f"{name} looks stable overall while some {seg} segments moved",
        }
        limitations = [
            "Standard errors use a normal (delta-method) approximation and assume independent rows; clustered rows (several orders per customer) make them too small.",
            "Segment p-values are FDR-adjusted across evaluable segments only, not across other segmentations that may have been tried.",
            "The two periods may differ in length or season; this analysis does not adjust for that.",
        ]
        if self.verdict in REVERSAL_VERDICTS and self.n_opposing_significant == 0:
            limitations.insert(0, "No opposing segment change is statistically distinguishable from zero.")

        return Finding(
            title=titles[self.verdict],
            category=FindingCategory.AGGREGATE_VS_SEGMENT,
            evidence_level=EvidenceLevel.STATISTICAL_LEAD if lead else EvidenceLevel.DESCRIPTIVE,
            metric=name,
            comparison=f"{cfg['comparison']} vs {cfg['baseline']} by {seg}",
            evidence={
                "aggregate": agg,
                "mix_effect": self.mix_effect,
                "rate_effect": self.rate_effect,
                "entry_exit_effect": self.entry_exit_effect,
                "n_evaluable_segments": self.n_eligible,
                "n_opposing": self.n_opposing,
                "n_opposing_significant": self.n_opposing_significant,
                "coverage": self.coverage,
                "top_segments": json.loads(self.segments.head(15).to_json(orient="records")),
            },
            method=(
                "Midpoint-weight mix/rate decomposition of a weighted-average metric; "
                "delta-method standard errors; Benjamini-Hochberg FDR across evaluable segments."
            ),
            n=self.rows_used,
            alternatives=_ALTERNATIVES[self.verdict],
            next_step=_NEXT_STEPS[self.verdict],
            parameters={**cfg, "data_source": self.data_source, "rows_used": self.rows_used},
            data_source=self.data_source,
            affected=seg,
            uncertainty={
                "aggregate_change_ci": [agg["ci_low"], agg["ci_high"]],
                "alpha": cfg["alpha"],
                "method": "normal approximation (delta method)",
            },
            limitations=limitations,
        )


# -------------------------------------------------------------------- main
def investigate_contradiction(
    df: pd.DataFrame,
    metric: MetricSpec,
    period_col: str,
    baseline: Any,
    comparison: Any,
    segment_col: str,
    *,
    min_segment_n: int = 30,
    min_coverage: float = 0.8,
    alpha: float = 0.05,
    flat_rel_tol: float = 0.01,
    filters: dict[str, Any] | None = None,
    data_source: str = "upload",
) -> ContradictionResult:
    """Compare `comparison` against `baseline` overall and within each segment."""
    _validate_inputs(df, metric, period_col, baseline, comparison, segment_col,
                     min_segment_n, min_coverage, alpha, flat_rel_tol)
    config = {
        "metric": metric.name, "numerator": metric.numerator, "denominator": metric.denominator,
        "period_col": period_col, "baseline": baseline, "comparison": comparison,
        "segment_col": segment_col, "min_segment_n": min_segment_n,
        "min_coverage": min_coverage, "alpha": alpha, "flat_rel_tol": flat_rel_tol,
        "filters": filters or {},
    }
    frame, missing = _prepare_rows(df, metric, period_col, baseline, comparison, segment_col)

    agg = _cell_stats(frame, ["_p"])
    cell = _cell_stats(frame, ["_p", "_s"])
    for period in (baseline, comparison):
        if not agg.loc[period, "y_sum"] > 0:
            raise ContradictionError(
                f"The denominator for period {period!r} is zero; the metric is undefined."
            )
    broken = cell[(cell["y_sum"] <= 0) & (cell["x_sum"].abs() > 0)]
    if len(broken):
        raise ContradictionError(
            "Some segments have a non-zero numerator but a zero denominator, so the ratio "
            "is undefined for them. Filter those rows or choose a different metric."
        )

    labels = sorted(frame["_s"].unique())
    part0 = cell.xs(baseline, level="_p").reindex(labels)
    part1 = cell.xs(comparison, level="_p").reindex(labels)

    def arr(part: pd.DataFrame, col: str) -> np.ndarray:
        return part[col].to_numpy(dtype="float64")

    n0, n1 = np.nan_to_num(arr(part0, "n")), np.nan_to_num(arr(part1, "n"))
    y0, y1 = np.nan_to_num(arr(part0, "y_sum")), np.nan_to_num(arr(part1, "y_sum"))
    m0, m1 = arr(part0, "value"), arr(part1, "value")
    se0, se1 = arr(part0, "se"), arr(part1, "se")

    w0, w1 = y0 / float(agg.loc[baseline, "y_sum"]), y1 / float(agg.loc[comparison, "y_sum"])
    both = (y0 > 0) & (y1 > 0)
    m0f, m1f = np.nan_to_num(m0), np.nan_to_num(m1)

    mix = np.where(both, (w1 - w0) * (m0f + m1f) / 2, 0.0)
    rate = np.where(both, (w0 + w1) / 2 * (m1f - m0f), 0.0)
    entry = np.where(~both, w1 * m1f - w0 * m0f, 0.0)

    delta_seg = np.where(both, m1f - m0f, np.nan)
    se_seg = np.where(both, np.sqrt(se0 ** 2 + se1 ** 2), np.nan)
    lo, hi, p = _inference(delta_seg, se_seg, alpha)

    value_b, value_c = float(agg.loc[baseline, "value"]), float(agg.loc[comparison, "value"])
    delta = value_c - value_b
    se_agg = float(np.sqrt(agg.loc[baseline, "se"] ** 2 + agg.loc[comparison, "se"] ** 2))
    a_lo, a_hi, a_p = _inference(np.array([delta]), np.array([se_agg]), alpha)

    if not np.isclose(mix.sum() + rate.sum() + entry.sum(), delta, rtol=1e-9, atol=1e-9):
        raise ContradictionError("Internal check failed: the decomposition does not sum to the change.")

    eligible = both & (n0 >= min_segment_n) & (n1 >= min_segment_n)
    q = np.full(len(labels), np.nan)
    if eligible.any():
        q[eligible] = multipletests(np.nan_to_num(p[eligible], nan=1.0), alpha=alpha,
                                    method="fdr_bh")[1]
    significant = eligible & (q < alpha)
    dirs = np.where(delta_seg > TOL, 1, np.where(delta_seg < -TOL, -1, 0))
    coverage = float(min(w0[eligible].sum(), w1[eligible].sum())) if eligible.any() else 0.0

    verdict, n_opp, n_opp_sig = _classify(
        delta, value_b, mix.sum(), rate.sum(), dirs, significant, eligible,
        coverage, min_coverage, flat_rel_tol,
    )

    aggregate = {
        "value_baseline": value_b, "value_comparison": value_c, "change": delta,
        "se_change": se_agg, "ci_low": float(a_lo[0]), "ci_high": float(a_hi[0]),
        "p_value": float(a_p[0]),
        "n_baseline": int(agg.loc[baseline, "n"]), "n_comparison": int(agg.loc[comparison, "n"]),
        "denominator_baseline": float(agg.loc[baseline, "y_sum"]),
        "denominator_comparison": float(agg.loc[comparison, "y_sum"]),
    }

    table = pd.DataFrame({
        "segment": labels,
        "n_baseline": n0.astype(int), "n_comparison": n1.astype(int),
        "weight_baseline": w0, "weight_comparison": w1,
        "value_baseline": m0, "value_comparison": m1,
        "change": delta_seg, "se_change": se_seg, "ci_low": lo, "ci_high": hi,
        "p_value": p, "q_value": q,
        "direction": np.where(~both, "n/a", np.select([dirs > 0, dirs < 0], ["up", "down"], "flat")),
        "evaluable": eligible, "significant": significant,
        "mix_effect": mix, "rate_effect": rate, "entry_exit_effect": entry,
    })
    table["total_contribution"] = table[["mix_effect", "rate_effect", "entry_exit_effect"]].sum(axis=1)
    table = table.loc[table["total_contribution"].abs().sort_values(ascending=False).index]
    table = table.reset_index(drop=True)

    warnings: list[str] = []
    small = ~eligible & both
    if small.any():
        warnings.append(
            f"{int(small.sum())} segment(s) holding {w1[small].sum():.1%} of comparison-period weight "
            f"have fewer than {min_segment_n} rows in a period and are not evaluated."
        )
    if (~both).any():
        warnings.append(f"{int((~both).sum())} segment(s) appear in only one period (entry/exit term).")
    if missing["rows_missing_segment"]:
        warnings.append(f"{missing['rows_missing_segment']:,} rows have no segment value; they form the '{MISSING_LABEL}' segment.")
    dropped = missing["rows_in_periods"] - missing["rows_used"]
    if dropped:
        warnings.append(f"{dropped:,} rows were dropped for a missing numerator or denominator.")
    if abs(delta) > TOL and abs(entry.sum()) > 0.1 * abs(delta):
        warnings.append("Entry/exit effects exceed 10% of the change; segment definitions may not be stable.")
    if eligible.sum() > 1:
        warnings.append("Segment p-values are FDR-adjusted across evaluable segments only.")

    explanation = _explain(verdict, metric.name, baseline, comparison, aggregate,
                           float(mix.sum()), float(rate.sum()), float(entry.sum()),
                           int(eligible.sum()), n_opp, n_opp_sig, coverage, min_coverage,
                           flat_rel_tol)
    return ContradictionResult(
        metric_name=metric.name, config=config, data_source=data_source, aggregate=aggregate,
        segments=table, mix_effect=float(mix.sum()), rate_effect=float(rate.sum()),
        entry_exit_effect=float(entry.sum()), verdict=verdict, n_eligible=int(eligible.sum()),
        n_opposing=n_opp, n_opposing_significant=n_opp_sig, coverage=coverage,
        rows_used=missing["rows_used"], missing=missing, warnings=warnings,
        explanation=explanation,
    )