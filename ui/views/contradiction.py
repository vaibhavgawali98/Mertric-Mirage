from __future__ import annotations

import math
from pathlib import Path

import streamlit as st

from core.contradiction import ContradictionError, MetricSpec, investigate_contradiction
from core.metrics import MetricError, matched_window
from core.sql_store import DEFAULT_DB_PATH, SqlStoreError, load_sql, open_readonly, run_named_query
from ui import logic, state, theme
from ui.logic import VERDICTS, fmt_num
from viz.charts import ChartError, build_figure, decomposition_spec, segment_comparison_spec, segment_trend_spec

SEGMENT_COLS = ["segment", "n_baseline", "n_comparison", "weight_baseline", "weight_comparison",
                "value_baseline", "value_comparison", "change", "ci_low", "ci_high", "q_value",
                "evaluable", "total_contribution"]
TONE_TO_CALLOUT = {"amber": "warn", "blue": "info", "teal": "ok"}


def render(ctx: logic.AppContext) -> None:
    info = ctx.info
    theme.page_header("Contradiction Lab", "Does the overall number tell a different story than its segments?")
    if info.source == "synthetic":
        theme.simulated_banner()
    k = f"c_{info.key}"
    freq = None
    if info.kind == "retail":
        freq = st.radio("Period grain", ["Q", "M", "Y"], horizontal=True, key=f"{k}_freq",
                        format_func={"Q": "Quarter", "M": "Month", "Y": "Year"}.get)
    elif info.kind == "upload":
        freq = info.freq          # the grain chosen when the file was prepared (None if no period column)
    frame = state.get_frame(info.key, freq or "M")
    access = logic.contradiction_access(frame)
    if not access.enabled:
        theme.disabled_panel("Contradiction Lab", access.reason)
        return
    nums, groups = logic.numeric_columns(frame), logic.group_columns(frame)

    c1, c2, c3 = st.columns(3)
    numerator = c1.selectbox("Metric column", nums, index=logic.index_of(nums, ["converted", "order_value"]), key=f"{k}_num")
    denoms = ["(none: mean of the column)"] + [c for c in nums if c != numerator]
    choice = c2.selectbox("Denominator column", denoms, key=f"{k}_den_{numerator}")
    denominator = None if choice == denoms[0] else choice
    period_col = c3.selectbox("Period column", groups, index=logic.index_of(groups, ["period"]), key=f"{k}_per")
    seg_options = [c for c in groups if c != period_col]
    segment_col = st.selectbox("Segment column", seg_options, index=logic.index_of(seg_options, ["segment", "Country"]),
                               key=f"{k}_seg_{period_col}")
    labels = logic.period_labels(frame, period_col)
    if len(labels) < 2:
        st.error("The period column needs at least two distinct values.")
        return
    d1, d2 = st.columns(2)
    baseline = d1.selectbox("Baseline period", labels, index=len(labels) - 2, key=f"{k}_b_{period_col}_{freq}")
    comp_options = [x for x in labels if x != baseline]
    comparison = d2.selectbox("Comparison period", comp_options, index=len(comp_options) - 1,
                              key=f"{k}_c_{period_col}_{freq}_{baseline}")

    can_match = (info.kind in ("retail", "upload") and bool(freq) and period_col == "period"
                 and "order_date" in frame.columns)
    incomplete = bool(can_match and logic.period_is_incomplete(frame, comparison, freq))
    use_matched = False
    if can_match:
        use_matched = st.checkbox("Compare the same elapsed days in both periods (matched window)", value=incomplete,
                                  key=f"{k}_match_{comparison}", help="Useful when the comparison period is only partly observed.")
        if incomplete and not use_matched:
            theme.callout("warn", f"{comparison} is only partly observed. Without a matched window the comparison is distorted.")
    with st.expander("Thresholds"):
        t1, t2, t3, t4 = st.columns(4)
        min_n = int(t1.number_input("Minimum rows per segment and period", 2, 10_000, 30, key=f"{k}_minn"))
        alpha = float(t2.selectbox("Significance level", [0.05, 0.01, 0.10], key=f"{k}_alpha"))
        min_cov = float(t3.slider("Minimum evaluable weight", 0.0, 1.0, 0.8, key=f"{k}_cov"))
        flat = float(t4.slider("Flat aggregate if change within", 0.0, 0.05, 0.01, key=f"{k}_flat"))

    prep = {"dataset": info.key, "freq": freq, "matched_window": use_matched}
    try:
        work = frame
        if use_matched:
            work, win = matched_window(frame, "order_date", "period", baseline, comparison, freq)
            st.caption(f"Matched window: first {win['limit_days']} days of each period; "
                       f"{win['baseline_removed']:,} baseline and {win['comparison_removed']:,} comparison rows left out.")
        metric = MetricSpec(logic.metric_name(frame, numerator, denominator), numerator, denominator)
        res = investigate_contradiction(work, metric, period_col, baseline, comparison, segment_col,
                                        min_segment_n=min_n, min_coverage=min_cov, alpha=alpha,
                                        flat_rel_tol=flat, data_source=info.source)
    except (ContradictionError, MetricError) as exc:
        st.error(str(exc))
        return

    label, tone = VERDICTS[res.verdict]
    theme.callout(TONE_TO_CALLOUT[tone], res.explanation, title=f"Verdict: {label}")
    agg = res.aggregate
    ci = f"{(1 - alpha) * 100:.0f}% CI [{agg['ci_low']:+.4g}, {agg['ci_high']:+.4g}]"
    theme.kpi_row([(str(baseline), fmt_num(agg["value_baseline"]), f"n = {agg['n_baseline']:,}", "blue"),
                   (str(comparison), fmt_num(agg["value_comparison"]), f"n = {agg['n_comparison']:,}", "teal"),
                   ("Change", f"{agg['change']:+.4g}", ci, "amber"),
                   ("Overall p-value", fmt_num(agg["p_value"]), "independent rows assumed", "blue")])
    for w in res.warnings:
        st.caption(f"Note: {w}")

    tabs = st.tabs(["Decomposition", "Segments", "Trend", "SQL and checks"])
    with tabs[0]:
        st.plotly_chart(build_figure(decomposition_spec(res)), key=f"{k}_wf")
        share = res.mix_share_of_change
        st.markdown(f"Mix effect **{res.mix_effect:+.4g}**, rate effect **{res.rate_effect:+.4g}**"
                    + (f", entry/exit **{res.entry_exit_effect:+.4g}**" if abs(res.entry_exit_effect) > 1e-12 else "")
                    + (f". Composition accounts for {share:.0%} of the change." if not math.isnan(share) else "."))
        st.caption("Mix effect: shifts in each segment's share of the denominator. Rate effect: changes inside segments, "
                   "weighted at average shares. They add up exactly to the overall change.")
    with tabs[1]:
        st.plotly_chart(build_figure(segment_comparison_spec(res)), key=f"{k}_seg")
        st.dataframe(res.segments[SEGMENT_COLS].head(30), hide_index=True)
    with tabs[2]:
        try:
            st.plotly_chart(build_figure(segment_trend_spec(work, metric, period_col, segment_col,
                                                            min_n=min_n, periods=logic.period_labels(work, period_col))),
                            key=f"{k}_trend")
        except ChartError as exc:
            st.caption(str(exc))
    with tabs[3]:
        _sql_panel(info, freq, period_col, segment_col, numerator, denominator, baseline, comparison, use_matched, res, k)

    finding = res.to_finding()
    if finding is None:
        theme.callout("info", f"The verdict '{label}' is not a reportable finding. A clean result is still a result.")
        return
    finding.chart_spec = decomposition_spec(res)
    finding.parameters["data_prep"] = prep
    with st.expander("Investigation card preview"):
        theme.render_finding(finding, f"{k}_card")
    if st.button("Save to Evidence Room", key=f"{k}_save"):
        _, is_new = state.get_store().save(finding)
        st.success("Saved." if is_new else "This exact finding is already saved.")


def _sql_panel(info, freq, period_col, segment_col, numerator, denominator, baseline, comparison, matched, res, k) -> None:
    if info.kind != "retail":
        st.caption("The SQL panel is available for the retail dataset, where the data lives in SQLite. "
                   "Simulated and uploaded datasets are analysed with the Pandas engine.")
        return
    st.markdown("**SQL behind the segment query** (SQLite, `orders` view = one row per sale invoice)")
    st.code(load_sql("segment_period_aov"), language="sql")
    eligible = (numerator == "order_value" and denominator is None
                and period_col == "period" and segment_col == "Country" and not matched)
    if not eligible:
        st.caption("The SQL check applies to the retail data with order_value, period and Country, without a matched window.")
        return
    if not Path(DEFAULT_DB_PATH).exists():
        st.caption("Build the database first: `python -m data.build_sqlite`.")
        return
    if st.button("Run this SQL and compare with Python", key=f"{k}_sqlrun"):
        try:
            conn = open_readonly()
            try:
                out = run_named_query(conn, "segment_period_aov",
                                      {"freq": freq, "baseline": str(baseline), "comparison": str(comparison)})
            finally:
                conn.close()
        except SqlStoreError as exc:
            st.error(str(exc))
            return
        verdicts = []
        for period, key in ((str(baseline), "value_baseline"), (str(comparison), "value_comparison")):
            part = out[out["period"] == period]
            sql_value = part["total_value"].sum() / part["n"].sum()
            verdicts.append(math.isclose(sql_value, res.aggregate[key], rel_tol=1e-9))
        theme.callout("ok" if all(verdicts) else "error",
                      "SQL and Python agree on both overall values." if all(verdicts) else "SQL and Python DISAGREE.")
        st.dataframe(out.head(20), hide_index=True)
