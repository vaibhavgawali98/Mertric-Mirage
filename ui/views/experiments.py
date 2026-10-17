from __future__ import annotations

import pandas as pd
import streamlit as st

from core.experiments import (
    CAUSAL_ASSUMPTIONS, EvidenceTier, ExperimentError, adjusted_effect, balance_check, evidence_tier,
    experiment_finding, minimum_detectable_effect, practical_assessment, sample_size_two_means,
    sample_size_two_proportions, srm_check, two_mean_test, two_proportion_test,
)
from ui import logic, state, theme
from ui.logic import fmt_num, fmt_pct
from viz.charts import build_figure, experiment_interval_spec

PRACTICAL_TEXT = {
    "meaningful_positive": ("ok", "The whole interval lies above the smallest worthwhile effect: meaningful."),
    "meaningful_negative": ("warn", "The whole interval lies below the negative of the smallest worthwhile effect."),
    "negligible": ("info", "The whole interval lies within the smallest worthwhile effect: negligible, even if statistically significant."),
    "inconclusive": ("info", "The interval is compatible with both worthwhile and negligible effects: inconclusive."),
}


def render(ctx: logic.AppContext) -> None:
    info = ctx.info
    theme.page_header("Experiment Lab", "Uncertainty, practical significance and what a comparison can claim")
    if info.source == "synthetic":
        theme.simulated_banner()
    tabs = st.tabs(["Group comparison", "Power and sample size", "Causal reasoning"])
    with tabs[0]:
        _comparison(ctx)
    with tabs[1]:
        _power()
    with tabs[2]:
        _causal()


def _comparison(ctx) -> None:
    info, k = ctx.info, f"e_{ctx.info.key}"
    frame = state.get_frame(info.key, "M")
    access = logic.experiment_access(frame)
    if not access.enabled:
        theme.disabled_panel("Group comparison", access.reason)
        return
    arms, outcomes = logic.group_columns(frame), logic.numeric_columns(frame)
    c1, c2, c3 = st.columns(3)
    arm_col = c1.selectbox("Group column", arms, index=logic.index_of(arms, ["arm"]), key=f"{k}_arm")
    levels = sorted(str(v) for v in frame[arm_col].dropna().unique())
    control = c2.selectbox("Control group", levels, index=logic.index_of(levels, ["control"]), key=f"{k}_ctl_{arm_col}")
    t_levels = [v for v in levels if v != control]
    treatment = c3.selectbox("Treatment group", t_levels, index=logic.index_of(t_levels, ["treatment"]), key=f"{k}_trt_{arm_col}_{control}")
    outcome = st.selectbox("Outcome column", outcomes, index=logic.index_of(outcomes, ["converted"]), key=f"{k}_out")

    work = frame.assign(**{arm_col: frame[arm_col].astype("object").where(frame[arm_col].notna(), None).map(lambda v: None if v is None else str(v))})
    in_arms = work[work[arm_col].isin([control, treatment])]
    sub = in_arms.dropna(subset=[outcome])
    a, b = sub.loc[sub[arm_col] == control, outcome], sub.loc[sub[arm_col] == treatment, outcome]
    if len(a) < 2 or len(b) < 2:
        st.error("Each group needs at least two rows with a value.")
        return
    binary = logic.is_binary(sub[outcome])
    name = logic.metric_name(frame, outcome, None)

    s1, s2 = st.columns(2)
    default_min = 0.01 if binary else round(0.05 * abs(float(a.mean())), 4)
    min_effect = float(s1.number_input("Smallest worthwhile effect (absolute, same units as the outcome)", 0.0, 1e9, default_min, key=f"{k}_min_{outcome}"))
    randomized = s2.checkbox("Units were randomly assigned to groups (my declaration; the tool cannot verify it)", key=f"{k}_rand")
    equal_split = s2.checkbox("The intended split was equal", value=True, key=f"{k}_split")
    others = [c for c in logic.numeric_columns(frame) + logic.group_columns(frame) if c not in (arm_col, outcome)]
    covariates = st.multiselect("Pre-treatment covariates (for balance checks and adjustment)", list(dict.fromkeys(others)), key=f"{k}_cov_{arm_col}_{outcome}")

    try:
        if binary:
            res = two_proportion_test(int(a.sum()), len(a), int(b.sum()), len(b))
            diff, lo, hi = res.risk_difference, res.ci_low, res.ci_high
            ctl_val, trt_val, fmt = res.rate_control, res.rate_treatment, fmt_pct
        else:
            res = two_mean_test(a, b)
            diff, lo, hi = res.mean_difference, res.ci_low, res.ci_high
            ctl_val, trt_val, fmt = res.mean_control, res.mean_treatment, fmt_num
        practical = practical_assessment(lo, hi, min_effect)
        counts = in_arms[arm_col].value_counts().to_dict()
        srm = srm_check(counts) if equal_split else {"flagged": False, "p_value": float("nan")}
        balance = balance_check(work, arm_col, control, treatment, covariates) if covariates else None
        adjusted = adjusted_effect(work, outcome, arm_col, control, treatment, covariates) if covariates else None
    except ExperimentError as exc:
        st.error(str(exc))
        return

    imbalance = bool(balance["flagged"].any()) if balance is not None else False
    tier, tier_text = evidence_tier(randomized, bool(srm["flagged"]), imbalance, adjusted is not None)
    theme.kpi_row([(control, fmt(ctl_val), f"n = {len(a):,}", "blue"), (treatment, fmt(trt_val), f"n = {len(b):,}", "teal"),
                   ("Difference", f"{diff:+.4g}", f"{(1 - res.alpha) * 100:.0f}% CI [{lo:+.4g}, {hi:+.4g}]", "amber"),
                   ("p-value", fmt_num(res.p_value), getattr(res, "test_used", "Welch t-test"), "blue")])
    kind, text = PRACTICAL_TEXT[practical]
    theme.callout(kind, text, title="Practical significance")
    theme.callout("ok" if tier == EvidenceTier.CAUSAL else "warn", tier_text, title=f"Evidence tier: {tier.value.replace('_', ' ')}")
    st.plotly_chart(build_figure(experiment_interval_spec(res, name, min_effect)), key=f"{k}_int")
    if equal_split:
        st.caption(f"Sample-ratio check: p = {fmt_num(srm['p_value'])}" + (" (FLAGGED: arm sizes do not match the intended split)" if srm["flagged"] else " (no mismatch)"))
    if balance is not None:
        st.markdown("**Covariate balance**")
        st.dataframe(balance, hide_index=True)
    if adjusted is not None:
        st.markdown(f"**Adjusted association:** naive difference {adjusted.naive_difference:+.4g}, adjusted {adjusted.adjusted_difference:+.4g} "
                    f"(CI [{adjusted.ci_low:+.4g}, {adjusted.ci_high:+.4g}]). Adjustment removes bias only from the covariates included.")
    if binary and 0 < res.rate_control < 1:
        st.caption(f"With {min(len(a), len(b)):,} per group, the smallest effect detectable with 80% power is about "
                   f"{minimum_detectable_effect(res.rate_control, min(len(a), len(b))) * 100:.2f} percentage points.")
    for w in getattr(res, "warnings", []):
        st.caption(w)

    finding = experiment_finding(res, name, tier, tier_text, practical, min_effect, data_source=info.source,
                                 control_label=control, treatment_label=treatment,
                                 parameters={"dataset": info.key, "arm_col": arm_col, "outcome": outcome,
                                             "control": control, "treatment": treatment,
                                             "randomized_declared": randomized, "covariates": covariates})
    finding.chart_spec = experiment_interval_spec(res, name, min_effect)
    if st.button("Save to Evidence Room", key=f"{k}_save"):
        _, is_new = state.get_store().save(finding)
        st.success("Saved." if is_new else "This exact finding is already saved.")


def _power() -> None:
    mode = st.radio("Outcome type", ["Proportion (such as conversion rate)", "Mean (such as order value)"], key="e_power_mode")
    try:
        if mode.startswith("Proportion"):
            c1, c2, c3, c4 = st.columns(4)
            p = c1.number_input("Baseline rate", 0.001, 0.999, 0.10, 0.01, key="e_p")
            mde = c2.number_input("Absolute effect to detect", 0.001, 0.5, 0.02, 0.005, format="%.3f", key="e_mde")
            alpha = c3.selectbox("Significance level", [0.05, 0.01, 0.10], key="e_a")
            power = c4.selectbox("Power", [0.8, 0.9, 0.95], key="e_pw")
            n = sample_size_two_proportions(p, mde, alpha, power)
            theme.kpi_row([("Per group", f"{n:,}", "", "teal"), ("Total", f"{2 * n:,}", "", "blue"),
                           ("Smallest effect at this size", f"{minimum_detectable_effect(p, n, alpha, power) * 100:.2f} pts", "", "amber")])
        else:
            c1, c2, c3, c4 = st.columns(4)
            sd = c1.number_input("Standard deviation", 0.0001, 1e9, 10.0, key="e_sd")
            mde = c2.number_input("Mean difference to detect", 0.0001, 1e9, 2.0, key="e_dm")
            alpha = c3.selectbox("Significance level", [0.05, 0.01, 0.10], key="e_a2")
            power = c4.selectbox("Power", [0.8, 0.9, 0.95], key="e_pw2")
            n = sample_size_two_means(sd, mde, alpha, power)
            theme.kpi_row([("Per group", f"{n:,}", "", "teal"), ("Total", f"{2 * n:,}", "", "blue")])
            st.caption("Order values are heavy-tailed; the standard deviation you enter drives the answer, so estimate it from real data.")
    except ExperimentError as exc:
        st.error(str(exc))


def _causal() -> None:
    st.markdown("""
Three kinds of claim, from weakest to strongest:

1. **Descriptive comparison.** Group A differs from group B. Says nothing about why.
2. **Adjusted association.** The difference after accounting for the covariates you included. Hidden confounders remain.
3. **Causal estimate.** Needs randomization (or an identification strategy) plus the assumptions below.

The lab only labels a result a causal estimate when you declare randomization and neither the sample-ratio
check nor the balance check flags a problem. A significant coefficient or a strong correlation never changes the tier.
""")
    st.markdown("**Assumptions required for a causal reading**")
    st.markdown("\n".join(f"- {a}" for a in CAUSAL_ASSUMPTIONS))