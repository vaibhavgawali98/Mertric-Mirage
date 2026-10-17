"""Engine D: business experiment and causal-reasoning lab.

Three evidence tiers are kept apart on purpose:
  descriptive comparison -> adjusted association -> causal estimate.
Only the last needs randomization (declared by the analyst, never verified here).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.power import NormalIndPower, TTestIndPower
from statsmodels.stats.proportion import proportion_confint

from core.contracts import EvidenceLevel, Finding, FindingCategory


class ExperimentError(ValueError):
    """Raised for invalid inputs to an experiment analysis."""


class EvidenceTier(str, Enum):
    DESCRIPTIVE = "descriptive_comparison"
    ADJUSTED = "adjusted_association"
    CAUSAL = "causal_estimate"


CAUSAL_ASSUMPTIONS = [
    "Units were randomly assigned, so assignment is independent of potential outcomes.",
    "No interference: one unit's assignment does not change another unit's outcome.",
    "Consistency: the treatment applied is the treatment defined (no silent changes, adequate compliance).",
    "Missing outcomes and attrition are unrelated to arm.",
    "The outcome and subgroup were chosen before looking at results.",
]


def _numeric(values) -> np.ndarray:
    x = pd.to_numeric(pd.Series(values), errors="coerce").dropna().astype("float64")
    return x[np.isfinite(x)].to_numpy()


def _require_cols(df: pd.DataFrame, cols: list[str]) -> None:
    if df is None or len(df) == 0:
        raise ExperimentError("The dataset is empty.")
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ExperimentError(f"Columns not found: {missing}")


# ------------------------------------------------------------- proportions
@dataclass
class ProportionTestResult:
    x_control: int
    n_control: int
    x_treatment: int
    n_treatment: int
    rate_control: float
    rate_treatment: float
    risk_difference: float      # treatment - control
    ci_low: float               # Newcombe hybrid-score interval
    ci_high: float
    relative_lift: float
    cohens_h: float
    p_value: float
    test_used: str
    alpha: float

    @property
    def significant(self) -> bool:
        return self.p_value < self.alpha


def _check_counts(name: str, x, n) -> None:
    if n != int(n) or x != int(x) or n <= 0 or x < 0 or x > n:
        raise ExperimentError(f"{name}: need integer 0 <= successes <= trials and trials > 0 (got {x}/{n}).")


def two_proportion_test(x_control, n_control, x_treatment, n_treatment, alpha: float = 0.05) -> ProportionTestResult:
    """Compare two proportions. The test is chosen from the data: Fisher's exact
    test when any expected cell count is below 5, otherwise the pooled z-test."""
    if not 0 < alpha < 1:
        raise ExperimentError("alpha must be between 0 and 1.")
    _check_counts("control", x_control, n_control)
    _check_counts("treatment", x_treatment, n_treatment)
    x1, n1, x2, n2 = int(x_control), int(n_control), int(x_treatment), int(n_treatment)
    p1, p2 = x1 / n1, x2 / n2
    d = p2 - p1

    l1, u1 = proportion_confint(x1, n1, alpha=alpha, method="wilson")
    l2, u2 = proportion_confint(x2, n2, alpha=alpha, method="wilson")
    lo = d - math.sqrt((p2 - l2) ** 2 + (u1 - p1) ** 2)    # Newcombe (1998), method 10
    hi = d + math.sqrt((u2 - p2) ** 2 + (p1 - l1) ** 2)

    total_x, total_n = x1 + x2, n1 + n2
    table = [[x1, n1 - x1], [x2, n2 - x2]]
    if total_x in (0, total_n):
        p_value, test = 1.0, "none (no variation in outcome)"
    else:
        expected = np.outer([n1, n2], [total_x, total_n - total_x]) / total_n
        if expected.min() < 5:
            p_value, test = float(stats.fisher_exact(table)[1]), "fisher_exact"
        else:
            pbar = total_x / total_n
            se = math.sqrt(pbar * (1 - pbar) * (1 / n1 + 1 / n2))
            p_value, test = float(2 * stats.norm.sf(abs(d / se))), "two_proportion_z"

    h = 2 * math.asin(math.sqrt(p2)) - 2 * math.asin(math.sqrt(p1))
    return ProportionTestResult(
        x1, n1, x2, n2, p1, p2, d, float(lo), float(hi),
        d / p1 if p1 > 0 else float("nan"), h, p_value, test, alpha,
    )


# ------------------------------------------------------------------- means
@dataclass
class MeanTestResult:
    n_control: int
    n_treatment: int
    mean_control: float
    mean_treatment: float
    mean_difference: float      # treatment - control
    ci_low: float               # Welch t interval
    ci_high: float
    boot_ci_low: float          # percentile bootstrap interval
    boot_ci_high: float
    median_difference: float
    hedges_g: float
    relative_lift: float
    p_value: float              # Welch t-test (about the MEAN)
    p_value_mannwhitney: float  # sensitivity check (about stochastic ordering, not the mean)
    alpha: float
    n_boot: int
    warnings: list[str] = field(default_factory=list)

    @property
    def significant(self) -> bool:
        return self.p_value < self.alpha


def two_mean_test(control, treatment, alpha: float = 0.05, n_boot: int = 2000, seed: int = 0) -> MeanTestResult:
    if not 0 < alpha < 1:
        raise ExperimentError("alpha must be between 0 and 1.")
    a, b = _numeric(control), _numeric(treatment)
    if len(a) < 2 or len(b) < 2:
        raise ExperimentError("Each group needs at least two non-missing numeric values.")
    na, nb = len(a), len(b)
    diff = float(b.mean() - a.mean())
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se2 = va / na + vb / nb
    if se2 == 0:
        lo = hi = diff
        p = 1.0 if diff == 0 else 0.0
    else:
        se = math.sqrt(se2)
        df = se2 ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
        tcrit = stats.t.ppf(1 - alpha / 2, df)
        lo, hi = diff - tcrit * se, diff + tcrit * se
        p = float(2 * stats.t.sf(abs(diff / se), df))
    try:
        p_mw = float(stats.mannwhitneyu(b, a, alternative="two-sided").pvalue)
    except ValueError:
        p_mw = float("nan")

    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        boots[i] = rng.choice(b, nb).mean() - rng.choice(a, na).mean()
    blo, bhi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    pooled = math.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    g = diff / pooled * (1 - 3 / (4 * (na + nb) - 9)) if pooled > 0 else float("nan")
    warnings = []
    if min(na, nb) < 30:
        warnings.append("Fewer than 30 values in a group: normal-theory intervals are unreliable.")
    return MeanTestResult(
        na, nb, float(a.mean()), float(b.mean()), diff, float(lo), float(hi), float(blo), float(bhi),
        float(np.median(b) - np.median(a)), float(g),
        diff / float(a.mean()) if a.mean() != 0 else float("nan"),
        p, p_mw, alpha, n_boot, warnings,
    )


# ------------------------------------------------------ practical meaning
def practical_assessment(ci_low: float, ci_high: float, min_effect: float) -> str:
    """Place a confidence interval against an analyst-set smallest worthwhile effect.

    meaningful_positive: whole CI above +min_effect
    meaningful_negative: whole CI below -min_effect
    negligible:          whole CI inside [-min_effect, +min_effect] (even if p < alpha)
    inconclusive:        anything else
    """
    if min_effect < 0:
        raise ExperimentError("min_effect must be non-negative.")
    if ci_low > min_effect:
        return "meaningful_positive"
    if ci_high < -min_effect:
        return "meaningful_negative"
    if ci_low >= -min_effect and ci_high <= min_effect:
        return "negligible"
    return "inconclusive"


# ------------------------------------------------------------------- power
def _check_baseline(p: float, mde: float) -> None:
    if not 0 < p < 1:
        raise ExperimentError("Baseline proportion must be between 0 and 1 (exclusive).")
    if mde == 0 or not 0 < p + mde < 1:
        raise ExperimentError("The effect must be non-zero and keep the proportion inside (0, 1).")


def sample_size_two_proportions(p_baseline: float, mde_abs: float, alpha: float = 0.05,
                                power: float = 0.8, ratio: float = 1.0) -> int:
    """Control-arm size to detect an absolute change `mde_abs` (Cohen's h, two-sided z)."""
    _check_baseline(p_baseline, mde_abs)
    h = abs(2 * math.asin(math.sqrt(p_baseline + mde_abs)) - 2 * math.asin(math.sqrt(p_baseline)))
    n = NormalIndPower().solve_power(effect_size=h, alpha=alpha, power=power, ratio=ratio,
                                     alternative="two-sided")
    return int(math.ceil(n))


def minimum_detectable_effect(p_baseline: float, n_per_arm: int, alpha: float = 0.05, power: float = 0.8) -> float:
    """Smallest absolute increase detectable with `n_per_arm` per arm."""
    if not 0 < p_baseline < 1 or n_per_arm < 2:
        raise ExperimentError("Need 0 < baseline < 1 and at least 2 units per arm.")
    h = NormalIndPower().solve_power(effect_size=None, nobs1=n_per_arm, alpha=alpha, power=power,
                                     ratio=1.0, alternative="two-sided")
    angle = min(math.asin(math.sqrt(p_baseline)) + h / 2, math.pi / 2)
    return float(math.sin(angle) ** 2 - p_baseline)


def sample_size_two_means(sd: float, mde: float, alpha: float = 0.05, power: float = 0.8) -> int:
    if sd <= 0 or mde == 0:
        raise ExperimentError("Need a positive standard deviation and a non-zero effect.")
    n = TTestIndPower().solve_power(effect_size=abs(mde) / sd, alpha=alpha, power=power, ratio=1.0)
    return int(math.ceil(n))


# ------------------------------------------- randomization / confounding
def srm_check(counts: dict[Any, int], expected_share: dict[Any, float] | None = None,
              threshold: float = 0.001) -> dict[str, Any]:
    """Sample-ratio-mismatch test: do arm sizes match the intended split?"""
    keys = list(counts)
    obs = np.array([counts[k] for k in keys], dtype=float)
    if len(keys) < 2 or obs.sum() <= 0 or (obs < 0).any():
        raise ExperimentError("Need at least two arms with non-negative counts and a positive total.")
    share = np.full(len(keys), 1 / len(keys)) if expected_share is None else np.array(
        [expected_share[k] for k in keys], dtype=float)
    if (share <= 0).any():
        raise ExperimentError("Expected shares must be positive.")
    share = share / share.sum()
    chi2, p = stats.chisquare(obs, share * obs.sum())
    return {"chi2": float(chi2), "p_value": float(p), "flagged": bool(p < threshold),
            "observed": {k: int(c) for k, c in zip(keys, obs)}, "threshold": threshold}


def balance_check(df: pd.DataFrame, arm_col: str, control: Any, treatment: Any, covariates: list[str],
                  alpha: float = 0.05, effect_threshold: float = 0.1) -> pd.DataFrame:
    """Are pre-treatment covariates balanced across arms?

    Numeric: standardized mean difference (flagged when |SMD| >= effect_threshold).
    Categorical: chi-square p-value and Cramer's V (flagged when p < alpha AND V >= effect_threshold).
    Flags use effect SIZE because with large samples trivial gaps are 'significant'.
    """
    _require_cols(df, [arm_col, *covariates])
    if control == treatment:
        raise ExperimentError("Control and treatment labels must differ.")
    base = df[df[arm_col].isin([control, treatment])]
    for label in (control, treatment):
        if not (base[arm_col] == label).any():
            raise ExperimentError(f"Arm {label!r} has no rows.")
    rows = []
    for col in covariates:
        sub = base[[arm_col, col]].dropna()
        is_t = sub[arm_col] == treatment
        s = sub[col]
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            a, b = s[~is_t].to_numpy(float), s[is_t].to_numpy(float)
            if len(a) < 2 or len(b) < 2:
                effect, p, flagged = float("nan"), float("nan"), False
            else:
                pooled = math.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
                if pooled > 0:
                    effect = float((b.mean() - a.mean()) / pooled)
                else:
                    effect = 0.0 if b.mean() == a.mean() else float("inf")
                p = float(stats.ttest_ind(b, a, equal_var=False).pvalue)
                flagged = abs(effect) >= effect_threshold
            kind = "numeric (SMD)"
        else:
            table = pd.crosstab(s, is_t)
            if table.shape[0] < 2 or table.shape[1] < 2:
                effect, p, flagged = 0.0, 1.0, False
            else:
                chi2, p, _, _ = stats.chi2_contingency(table, correction=False)
                effect = float(math.sqrt(chi2 / table.to_numpy().sum()))
                p = float(p)
                flagged = bool(p < alpha and effect >= effect_threshold)
            kind = "categorical (Cramer's V)"
        rows.append({"covariate": col, "kind": kind, "effect_size": effect, "p_value": p,
                     "n_used": int(len(sub)), "flagged": bool(flagged)})
    return pd.DataFrame(rows)


# --------------------------------------------------- adjusted association
@dataclass
class AdjustedResult:
    outcome: str
    n_used: int
    n_dropped: int
    naive_difference: float
    adjusted_difference: float
    ci_low: float
    ci_high: float
    p_value: float
    covariates: list[str]
    shift_from_adjustment: float
    model: str = "OLS with HC3 robust standard errors (for a 0/1 outcome: linear probability model)"
    tier: EvidenceTier = EvidenceTier.ADJUSTED


def adjusted_effect(df: pd.DataFrame, outcome_col: str, arm_col: str, control: Any, treatment: Any,
                    covariates: list[str], alpha: float = 0.05) -> AdjustedResult:
    """Treatment-minus-control difference after adjusting for covariates.
    An ADJUSTED ASSOCIATION: it removes bias only from the covariates included."""
    _require_cols(df, [outcome_col, arm_col, *covariates])
    if not pd.api.types.is_numeric_dtype(df[outcome_col]):
        raise ExperimentError(f"Outcome '{outcome_col}' must be numeric (use 0/1 for a binary outcome).")
    scoped = df[df[arm_col].isin([control, treatment])][[outcome_col, arm_col, *covariates]]
    sub = scoped.dropna()
    y = sub[outcome_col].astype(float)
    t = (sub[arm_col] == treatment).astype(float).rename("treated")
    if t.nunique() < 2:
        raise ExperimentError("Both arms need rows after dropping missing values.")
    X = pd.concat([t, pd.get_dummies(sub[covariates], drop_first=True, dtype=float)], axis=1)
    X = X.loc[:, X.nunique() > 1]
    X = sm.add_constant(X, has_constant="add")
    if len(sub) <= X.shape[1] + 1:
        raise ExperimentError("Too few rows for the number of covariates.")
    fit = sm.OLS(y, X).fit(cov_type="HC3")
    ci = fit.conf_int(alpha=alpha).loc["treated"]
    naive = float(y[t == 1].mean() - y[t == 0].mean())
    adj = float(fit.params["treated"])
    return AdjustedResult(outcome_col, int(len(sub)), int(len(scoped) - len(sub)), naive, adj,
                          float(ci.iloc[0]), float(ci.iloc[1]), float(fit.pvalues["treated"]),
                          list(covariates), adj - naive)


def evidence_tier(randomized: bool, srm_flagged: bool, imbalance_flagged: bool,
                  adjusted: bool) -> tuple[EvidenceTier, str]:
    """Which kind of claim the analysis can support. `randomized` is the ANALYST'S
    declaration; the tool cannot verify how assignment really happened."""
    if randomized and not srm_flagged and not imbalance_flagged:
        return EvidenceTier.CAUSAL, (
            "Assignment was declared randomized and the sample-ratio and balance checks raised no flag, "
            "so the difference can be read as a causal estimate, provided these assumptions hold: "
            + " ".join(CAUSAL_ASSUMPTIONS)
        )
    reasons = []
    if not randomized:
        reasons.append("assignment was not declared randomized")
    if srm_flagged:
        reasons.append("the sample-ratio check flagged a mismatch (randomization may have failed)")
    if imbalance_flagged:
        reasons.append("pre-treatment covariates are imbalanced across arms")
    tier = EvidenceTier.ADJUSTED if adjusted else EvidenceTier.DESCRIPTIVE
    return tier, ("A causal reading is not supported: " + "; ".join(reasons) + ". "
                  "Differences may reflect who ended up in each group, not the treatment itself.")


# ------------------------------------------------------------ finding card
def experiment_finding(result: ProportionTestResult | MeanTestResult, metric: str, tier: EvidenceTier,
                       tier_text: str, practical: str, min_effect: float, data_source: str = "upload",
                       control_label: str = "control", treatment_label: str = "treatment",
                       parameters: dict[str, Any] | None = None) -> Finding:
    if isinstance(result, ProportionTestResult):
        diff, lo, hi = result.risk_difference, result.ci_low, result.ci_high
        n = result.n_control + result.n_treatment
        method = f"Two-proportion comparison ({result.test_used}); Newcombe hybrid-score interval."
    else:
        diff, lo, hi = result.mean_difference, result.ci_low, result.ci_high
        n = result.n_control + result.n_treatment
        method = "Welch t-test and interval; Mann-Whitney and percentile bootstrap as sensitivity checks."
    if tier == EvidenceTier.CAUSAL:
        title = f"Estimated treatment effect on {metric}: {diff:+.4g} (randomized comparison)"
        alternatives = [
            "Chance: see the confidence interval for the plausible range.",
            "One or more causal assumptions may fail (interference, non-compliance, selective attrition).",
            "A short test window can capture novelty effects that fade.",
        ]
    else:
        title = f"{metric} differs from control by {diff:+.4g} (association, causation not established)"
        alternatives = [
            "Confounding: groups may differ in ways that also drive the outcome.",
            "Selection into the groups rather than an effect of the treatment.",
            "Chance: see the confidence interval for the plausible range.",
        ]
    return Finding(
        title=title,
        category=FindingCategory.EXPERIMENT_RESULT,
        evidence_level=EvidenceLevel.STATISTICAL_LEAD if result.significant else EvidenceLevel.DESCRIPTIVE,
        metric=metric,
        comparison=f"{treatment_label} vs {control_label}",
        evidence={**asdict(result), "evidence_tier": tier.value, "practical_assessment": practical,
                  "min_effect": min_effect, "tier_explanation": tier_text},
        method=method, n=n, alternatives=alternatives,
        next_step=("Confirm the assumptions and check the effect in a holdout period."
                   if tier == EvidenceTier.CAUSAL else
                   "Compare against a randomized test, or adjust for the confounders that plausibly drive selection."),
        parameters={**(parameters or {}), "alpha": result.alpha, "min_effect": min_effect},
        data_source=data_source, affected=str(treatment_label),
        uncertainty={"ci": [lo, hi], "alpha": result.alpha},
        limitations=["The smallest worthwhile effect is an analyst-set threshold, not a statistical constant.",
                     "Testing several outcomes or subgroups raises the chance of a false positive."],
    )