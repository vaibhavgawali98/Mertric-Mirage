from __future__ import annotations

import pandas as pd
import streamlit as st

from core.cohorts import (
    CohortError, compare_cohort_retention, concentration_finding, customer_table, funnel_available,
    repeat_purchase_gaps, retention_comparison_finding,
)
from ui import logic, state, theme
from ui.logic import fmt_num, fmt_pct
from viz.charts import ChartError, build_figure, distribution_spec, experiment_interval_spec, lorenz_spec, retention_heatmap_spec


def render(ctx: logic.AppContext) -> None:
    info = ctx.info
    theme.page_header("Cohort Observatory", "Retention, repeat behaviour and dependence on major customers")
    if info.source == "synthetic":
        theme.simulated_banner()
    frame = state.get_frame(info.key, "M")
    access = logic.cohort_access(frame)
    ok, reason = funnel_available(frame, "user", "stage", ["view", "cart", "buy"])
    if not access.enabled:
        theme.disabled_panel("Cohort analysis", access.reason)
        theme.disabled_panel("Funnel analysis", reason)
        return
    theme.callout("info", "The first cohort contains every customer already active when the data begins, and later cohorts "
                          "have had less time to return. Blank heatmap cells mean 'not yet observable', not zero.")
    if info.kind == "upload":
        theme.callout("info", "Cohorts assume one row per order. If your file has several rows per order, go back to Upload Data "
                              "and tick 'combine lines into one row per order ID'.")
    tabs = st.tabs(["Retention", "Repeat purchasing", "Revenue concentration", "Value tiers", "Funnel"])
    sections = (lambda: _retention(ctx), lambda: _repeat(ctx, frame), lambda: _concentration(ctx),
                lambda: _tiers(frame), lambda: theme.disabled_panel("Funnel analysis", reason))
    for tab, section in zip(tabs, sections):
        with tab:
            try:
                section()
            except CohortError as exc:     # for example too little data: say so instead of crashing
                st.error(str(exc))


def _retention(ctx) -> None:
    k, info = f"co_{ctx.info.key}", ctx.info
    c1, c2, c3 = st.columns(3)
    freq = c1.radio("Cohort grain", ["M", "Q"], horizontal=True, key=f"{k}_freq", format_func={"M": "Month", "Q": "Quarter"}.get)
    min_size = int(c2.number_input("Minimum cohort size", 1, 1000, 30, key=f"{k}_min"))
    res = state.retention(info.key, freq, min_size)
    available = int(max(res.counts.columns))
    if available < 1:
        st.info("Only one period of data is available. Retention needs customers observed in at least two periods.")
        return
    if available == 1:
        max_off = 1
    else:
        max_off = int(c3.slider("Periods shown", 1, available, min(12, available), key=f"{k}_off"))
    for w in res.warnings:
        st.caption(f"Note: {w}")
    st.plotly_chart(build_figure(retention_heatmap_spec(res, max_off)), key=f"{k}_heat")

    st.markdown("**Compare two cohorts at the same age**")
    st.caption("Use this for a comparison chosen in advance. Scanning many cohort pairs inflates false positives, and different "
               "calendar windows mean seasonality is not controlled.")
    cohorts = list(res.counts.index)
    if len(cohorts) < 2:
        st.info("At least two cohorts are needed to compare them.")
        return
    a_col, b_col = st.columns(2)
    a = a_col.selectbox("Baseline cohort", cohorts, key=f"{k}_a_{freq}")
    b = b_col.selectbox("Comparison cohort", [c for c in cohorts if c != a], index=min(5, len(cohorts) - 2), key=f"{k}_b_{freq}_{a}")
    offsets = [o for o in res.counts.columns if o >= 1 and pd.notna(res.counts.loc[a, o]) and pd.notna(res.counts.loc[b, o])]
    if not offsets:
        st.info("No age is observable for both cohorts.")
        return
    offset = st.selectbox("Periods since first order", offsets, key=f"{k}_o_{freq}_{a}_{b}")
    try:
        test = compare_cohort_retention(res, a, b, int(offset))
    except CohortError as exc:
        st.error(str(exc))
        return
    theme.kpi_row([(a, fmt_pct(test.rate_control), f"{test.n_control:,} customers", "blue"),
                   (b, fmt_pct(test.rate_treatment), f"{test.n_treatment:,} customers", "teal"),
                   ("Difference", f"{test.risk_difference * 100:+.1f} pts", f"95% CI [{test.ci_low * 100:+.1f}, {test.ci_high * 100:+.1f}]", "amber"),
                   ("p-value", fmt_num(test.p_value), test.test_used, "blue")])
    finding = retention_comparison_finding(test, a, b, int(offset), data_source=info.source)
    finding.chart_spec = experiment_interval_spec(test, "retention rate")
    if st.button("Save to Evidence Room", key=f"{k}_save_ret"):
        _, is_new = state.get_store().save(finding)
        st.success("Saved." if is_new else "This exact finding is already saved.")


def _repeat(ctx, frame) -> None:
    k = f"co_{ctx.info.key}_rep"
    summary, gaps = repeat_purchase_gaps(frame)
    theme.kpi_row([("Customers", f"{summary['n_customers']:,}", "", "blue"),
                   ("Repeat customers", fmt_pct(summary["repeat_share"]), "bought on 2+ distinct days", "teal"),
                   ("Median days between purchases", fmt_num(summary["median_days"]), f"middle half: {fmt_num(summary['p25_days'])} to {fmt_num(summary['p75_days'])}", "amber")])
    try:
        st.plotly_chart(build_figure(distribution_spec(gaps, bins=40, title="Days between consecutive purchases", x_title="days", log_x=False)), key=f"{k}_hist")
    except ChartError as exc:
        st.caption(str(exc))
    st.caption("Same-day invoices count as one purchase occasion, so this differs slightly from the invoice-based repeat rate on the Command Center.")


def _concentration(ctx) -> None:
    k, info = f"co_{ctx.info.key}_conc", ctx.info
    conc = state.concentration(info.key)
    items = []
    for name, share in conc.top_shares.items():
        lo, hi = conc.top_share_ci[name]
        items.append((name.replace("_", " ").replace("pct", "%"), fmt_pct(share), f"95% CI {fmt_pct(lo)} to {fmt_pct(hi)}", "amber"))
    theme.kpi_row(items)
    st.write(f"Gini coefficient **{conc.gini:.2f}**; the largest customer holds {fmt_pct(conc.largest_customer_share, 2)} of identified revenue.")
    st.plotly_chart(build_figure(lorenz_spec(conc)), key=f"{k}_lorenz")
    st.caption("Shares are of identified-customer revenue only. Orders without a customer ID are excluded.")
    finding = concentration_finding(conc, data_source=info.source)
    finding.chart_spec = lorenz_spec(conc)
    if st.button("Save to Evidence Room", key=f"{k}_save"):
        _, is_new = state.get_store().save(finding)
        st.success("Saved." if is_new else "This exact finding is already saved.")


def _tiers(frame) -> None:
    t = customer_table(frame)
    summary = t.groupby("value_tier").agg(customers=("revenue", "size"), revenue=("revenue", "sum"),
                                          median_orders=("n_orders", "median"), median_recency_days=("recency_days", "median"))
    summary["revenue_share"] = (summary["revenue"] / summary["revenue"].sum()).map(fmt_pct)
    st.dataframe(summary.reindex(["top_20pct", "middle_30pct", "bottom_50pct"]).reset_index())
    theme.callout("warn", "Tiers are defined by the same revenue they would be used to explain, so using them as a "
                          "segment for a revenue metric is circular.", title="Caution")
