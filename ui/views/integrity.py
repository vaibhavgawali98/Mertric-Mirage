from __future__ import annotations

import pandas as pd
import streamlit as st

from core.integrity import (
    IntegrityError, change_point_candidate, check_rate_columns, compare_periods_robust,
    distribution_summary, duplicate_finding, duplicate_summary, grain_check, iqr_outliers,
    leakage_warnings, mad_outliers, missing_share_finding, missingness_association,
    missingness_by_column, missingness_by_segment, missingness_by_time,
)
from core.metrics import MetricError, period_series
from ui import logic, state, theme
from ui.logic import fmt_num, fmt_pct
from viz.charts import ChartError, build_figure, distribution_spec


def _save(finding, key: str) -> None:
    if finding is not None and st.button("Save to Evidence Room", key=key):
        _, is_new = state.get_store().save(finding)
        st.success("Saved." if is_new else "This exact finding is already saved.")


def render(ctx: logic.AppContext) -> None:
    info = ctx.info
    theme.page_header("Metric Integrity", "Is the number misleading because of data quality or distribution?")
    if info.source == "synthetic":
        theme.simulated_banner()
    theme.callout("info", "A statistical anomaly is something these checks can measure. A confirmed data error "
                          "needs a person or the source system. Nothing on this page claims the second.")
    tabs = st.tabs(["Missing data", "Duplicates and keys", "Distribution and outliers",
                    "Mean vs median", "Change points", "Rates and leakage"])
    with tabs[0]:
        _missing(ctx)
    with tabs[1]:
        _duplicates(ctx)
    with tabs[2]:
        _distribution(ctx)
    with tabs[3]:
        _mean_median(ctx)
    with tabs[4]:
        _change_point(ctx)
    with tabs[5]:
        _rates_leakage(ctx)


def _visible(ctx) -> list[str]:
    return logic.RAW_COLUMNS if ctx.info.kind == "retail" else [c for c in ctx.raw.columns if c != "is_simulated"]


def _missing(ctx) -> None:
    k, raw, info = f"i_{ctx.info.key}_miss", ctx.raw, ctx.info
    miss = missingness_by_column(raw)
    miss = miss[miss["column"].isin(_visible(ctx))]
    st.dataframe(miss.assign(pct_missing=miss["pct_missing"].map(fmt_pct)), hide_index=True)
    with_missing = miss.loc[miss["n_missing"] > 0, "column"].tolist()
    if not with_missing:
        theme.callout("ok", "No missing values in any column.")
        return
    column = st.selectbox("Investigate missingness in", with_missing, key=f"{k}_col")
    n_missing = int(miss.loc[miss["column"] == column, "n_missing"].iloc[0])
    dates = logic.datetime_columns(raw)
    if dates:
        try:
            by_time = missingness_by_time(raw, dates[0], column, "M")
            spec = logic.series_to_line_spec(by_time.set_index("period")["pct_missing"] * 100,
                                             f"Share of rows missing '{column}' by month", "% missing", "month")
            st.plotly_chart(build_figure(spec), key=f"{k}_time")
        except IntegrityError as exc:
            st.caption(str(exc))
    seg_cols = ["Country"] if info.kind == "retail" else logic.group_columns(raw)
    assoc = None
    if seg_cols:
        seg = st.selectbox("Compare across", seg_cols, key=f"{k}_seg")
        assoc = missingness_association(raw, column, seg)
        if assoc["testable"]:
            st.write(f"Association with {seg}: Cramér's V = {assoc['cramers_v']:.3f}, p = {fmt_num(assoc['p_value'])}.")
            st.caption("With large samples the p-value is tiny for trivial gaps; read Cramér's V for the size of the effect.")
        else:
            st.caption(assoc["reason"])
        with st.expander("Segment table"):
            st.dataframe(missingness_by_segment(raw, column, seg), hide_index=True)
    _save(missing_share_finding(column, n_missing, len(raw), assoc, data_source=info.source), f"{k}_save")


def _duplicates(ctx) -> None:
    k, info = f"i_{ctx.info.key}_dup", ctx.info
    if info.kind == "retail":
        summary = state.dup_check(info.key)
        st.caption("Key: all eight raw columns.")
    else:
        candidates = [c for c in ctx.raw.columns if c != "is_simulated" and not logic.ID_LIKE.search(c)]
        subset = st.multiselect("Key columns", candidates, default=candidates, key=f"{k}_cols")
        if not subset:
            st.warning("Pick at least one key column.")
            return
        summary = duplicate_summary(ctx.raw, subset)
    theme.kpi_row([("Duplicate rows", f"{summary['n_duplicate_rows']:,}", "repeat an earlier row", "amber"),
                   ("Share of rows", fmt_pct(summary["pct_duplicate_rows"], 2), "", "blue"),
                   ("Repeated groups", f"{summary['n_duplicated_groups']:,}", "", "teal")])
    st.caption("Duplicates are kept and flagged, never removed automatically: they may be legitimate repeat scans.")
    _save(duplicate_finding(summary, data_source=info.source), f"{k}_save")
    if info.kind == "retail":
        st.markdown("**Invoice consistency** (does one invoice ever carry two customers, dates or countries?)")
        if st.button("Run check (about a minute)", key=f"{k}_keys"):
            st.dataframe(state.invoice_key_conflicts(info.key), hide_index=True)


def _distribution(ctx) -> None:
    k, info = f"i_{ctx.info.key}_dist", ctx.info
    frame = state.get_frame(info.key, "M")
    cols = logic.numeric_columns(frame)
    if not cols:
        theme.disabled_panel("Distribution checks", "No numeric column is available.")
        return
    col = st.selectbox("Column", cols, index=logic.index_of(cols, ["order_value"]), key=f"{k}_col")
    try:
        s = distribution_summary(frame[col])
        iqr, mad = iqr_outliers(frame[col]), mad_outliers(frame[col])
    except IntegrityError as exc:
        st.error(str(exc))
        return
    theme.kpi_row([("Mean", fmt_num(s["mean"]), "", "blue"), ("Median", fmt_num(s["median"]), "", "teal"),
                   ("Mean above median", fmt_pct(s["mean_median_gap_pct"]), f"skewness {fmt_num(s['skewness'], 3)}", "amber"),
                   (f"Top {s['top_k']:,} values", fmt_pct(s["top_k_share_of_total"]), "share of the total", "amber")])
    st.dataframe(pd.DataFrame([
        {"Method": "IQR fences (1.5 x IQR)", "Flagged": iqr.n_outliers, "Share of rows": fmt_pct(iqr.share_of_rows), "Share of total value": fmt_pct(iqr.share_of_total)},
        {"Method": "Modified z-score (MAD, 3.5)", "Flagged": mad.n_outliers, "Share of rows": fmt_pct(mad.share_of_rows), "Share of total value": fmt_pct(mad.share_of_total)},
    ]), hide_index=True)
    positive = bool((frame[col].dropna() > 0).all())
    log_x = st.checkbox("Log scale", value=positive, disabled=not positive, key=f"{k}_log")
    try:
        st.plotly_chart(build_figure(distribution_spec(frame[col], title=f"Distribution of {col}", x_title=col, log_x=log_x)),
                        key=f"{k}_chart")
    except ChartError as exc:
        st.caption(str(exc))
    st.caption("Flagged values are statistical outliers, not errors. Wholesale buyers legitimately place very large orders.")


def _mean_median(ctx) -> None:
    k, info = f"i_{ctx.info.key}_mm", ctx.info
    frame = state.get_frame(info.key, "Q" if info.kind == "retail" else "M")
    groups, cols = logic.group_columns(frame), logic.numeric_columns(frame)
    if not groups or not cols:
        theme.disabled_panel("Mean vs median", "Needs a period column and a numeric column.")
        return
    c1, c2 = st.columns(2)
    period_col = c1.selectbox("Period column", groups, index=logic.index_of(groups, ["period"]), key=f"{k}_per")
    value = c2.selectbox("Value column", cols, index=logic.index_of(cols, ["order_value"]), key=f"{k}_val")
    labels = logic.period_labels(frame, period_col)
    d1, d2 = st.columns(2)
    baseline = d1.selectbox("Baseline", labels, index=max(len(labels) - 2, 0), key=f"{k}_b_{period_col}")
    options = [x for x in labels if x != baseline]
    if not options:
        st.info("Needs two different periods.")
        return
    comparison = d2.selectbox("Comparison", options, index=len(options) - 1, key=f"{k}_c_{period_col}_{baseline}")
    try:
        res = compare_periods_robust(frame, value, period_col, baseline, comparison)
    except IntegrityError as exc:
        st.error(str(exc))
        return
    st.dataframe(pd.DataFrame({
        "": ["Mean", "Median", "1%-trimmed mean", "n"],
        str(baseline): [fmt_num(res.mean_baseline), fmt_num(res.median_baseline), fmt_num(res.trimmed_baseline), f"{res.n_baseline:,}"],
        str(comparison): [fmt_num(res.mean_comparison), fmt_num(res.median_comparison), fmt_num(res.trimmed_comparison), f"{res.n_comparison:,}"],
        "Change": [fmt_pct(res.rel_change_mean), fmt_pct(res.rel_change_median), fmt_pct(res.rel_change_trimmed), ""],
    }), hide_index=True)
    theme.callout("warn" if res.flagged else "ok", res.reason + (" The change in the mean is carried by extreme values." if res.flagged else " No sign that extreme values carry the change."),
                  title="Flagged" if res.flagged else "Not flagged")
    for w in res.warnings:
        st.caption(w)
    _save(res.to_finding(data_source=info.source), f"{k}_save")


def _change_point(ctx) -> None:
    k, info = f"i_{ctx.info.key}_cp", ctx.info
    frame = state.get_frame(info.key, "M")
    cols, groups = logic.numeric_columns(frame), logic.group_columns(frame)
    if not cols or not groups:
        theme.disabled_panel("Change-point check", "Needs a numeric column and a period column.")
        return
    # The grain of the `period` labels: monthly for retail, whatever was chosen for an upload.
    freq = info.freq if info.kind == "upload" and info.freq else "M"
    c1, c2, c3 = st.columns(3)
    value = c1.selectbox("Value column", cols, index=logic.index_of(cols, ["order_value"]), key=f"{k}_val")
    how = c2.selectbox("Aggregate per period", ["mean", "sum", "median"], key=f"{k}_how")
    period_col = c3.selectbox("Period column", groups, index=logic.index_of(groups, ["period"]), key=f"{k}_per")
    try:
        if "order_date" in frame.columns and period_col == "period":
            series, dropped = period_series(frame, value, "period", "order_date", freq, how)
            if dropped:
                st.caption(f"The incomplete final period {dropped} was dropped so it cannot look like a collapse.")
        else:
            series = frame.groupby(period_col)[value].agg(how).sort_index()
        if len(series) < 6:
            theme.callout("info", f"Only {len(series)} period(s) available; a change-point check needs at least 6.")
            return
        res = change_point_candidate(series)
    except (IntegrityError, MetricError) as exc:
        st.error(str(exc))
        return
    st.plotly_chart(build_figure(logic.series_to_line_spec(series, f"{how.capitalize()} {value} per period", value)), key=f"{k}_chart")
    st.write(f"Best split starts at **{res.label}**: mean {fmt_num(res.mean_before)} before, {fmt_num(res.mean_after)} after "
             f"(permutation p = {fmt_num(res.p_value)}).")
    theme.callout("warn" if res.significant else "ok", res.note, title="Candidate only" if res.significant else "No evidence of a level shift")
    _save(res.to_finding(f"{how} {value}", data_source=info.source), f"{k}_save")


def _rates_leakage(ctx) -> None:
    k, info = f"i_{ctx.info.key}_rl", ctx.info
    frame = state.get_frame(info.key, "M")
    cols = logic.numeric_columns(frame)
    with st.expander("Rate and denominator check"):
        if len(cols) < 2:
            st.caption("Needs two numeric columns.")
        else:
            a, b = st.columns(2)
            num = a.selectbox("Numerator", cols, key=f"{k}_n")
            den = b.selectbox("Denominator", [c for c in cols if c != num], key=f"{k}_d")
            out = check_rate_columns(frame, num, den)
            st.json({x: out[x] for x in out if x != "warnings"})
            for w in out["warnings"]:
                st.warning(w)
    with st.expander("Predictive-use leakage check"):
        if len(cols) < 2:
            st.caption("Needs a numeric target and at least one other column.")
        else:
            target = st.selectbox("Target", cols, key=f"{k}_t")
            others = [c for c in frame.columns if c != target and c != "is_simulated"]
            feats = st.multiselect("Candidate features", others, default=others[:3], key=f"{k}_f")
            if feats:
                warns = leakage_warnings(frame, target, feats)
                for w in warns:
                    st.warning(w)
                if not warns:
                    st.success("No warnings. That is not proof of no leakage.")
    with st.expander("Grain and double-counting check"):
        if info.kind == "retail":
            st.caption("Retail grain is explicit: `transactions` has one row per invoice line and the `orders` view "
                       "one row per sale invoice (see the SQL on the Contradiction Lab page).")
        else:
            keys = [c for c in ctx.raw.columns if c != "is_simulated"]
            key_col = st.selectbox("Key column", keys, key=f"{k}_gk")
            vals = [c for c in logic.numeric_columns(ctx.raw) if c != key_col]
            if vals:
                st.dataframe(grain_check(ctx.raw, key_col, vals), hide_index=True)
