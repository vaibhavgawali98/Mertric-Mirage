from __future__ import annotations

import pandas as pd
import streamlit as st

from core import ingest
from ui import logic, state, theme
from ui.logic import fmt_num, fmt_pct


def render(ctx: logic.AppContext) -> None:
    info = ctx.info
    theme.page_header("Command Center", "Dataset health, headline metrics and where to look next")
    if info.source == "synthetic":
        theme.simulated_banner()
    st.caption(info.description)
    if info.kind == "retail":
        _retail_overview(ctx)
    elif info.kind == "upload":
        _upload_overview(ctx)
    else:
        _table_overview(ctx)
    _availability(ctx)
    st.caption(f"{state.get_store().count()} finding(s) saved in the Evidence Room.")


def _retail_overview(ctx: logic.AppContext) -> None:
    m, h = state.headline_metrics(ctx.info.key), state.retail_health(ctx.info.key)
    theme.kpi_row([
        ("Gross sales", fmt_num(m["gross_sales"]), "genuine sale rows only", "teal"),
        ("Net revenue", fmt_num(m["net_revenue"]), f"after returns of {fmt_num(m['returns_value'])}", "blue"),
        ("Orders", fmt_num(m["orders"]), "distinct sale invoices", "teal"),
        ("Active customers", fmt_num(m["active_customers"]), "rows with a customer ID", "blue"),
        ("Average order value", fmt_num(m["average_order_value"]), "gross sales / orders", "amber"),
    ])
    theme.kpi_row([
        ("Repeat rate", fmt_pct(m["repeat_rate"]), "customers with 2+ invoices; right-censored", "blue"),
        ("Return rate (value)", fmt_pct(m["return_rate_value"]), "returns / gross sales", "teal"),
        ("Rows without customer ID", fmt_pct(h["missing_customer_rows"] / h["n_rows"]),
         f"{fmt_pct(h['missing_customer_revenue_share'])} of gross sales", "amber"),
        ("Exact duplicate rows", f"{h['duplicates']['n_duplicate_rows']:,}",
         f"{fmt_pct(h['duplicates']['pct_duplicate_rows'], 2)} of rows; kept and flagged", "amber"),
    ])
    st.markdown("### Dataset health")
    st.write(f"{h['n_rows']:,} invoice lines from {h['date_min'][:10]} to {h['date_max'][:10]}.")
    st.dataframe(h["classes"].rename(columns={"row_class": "Row class", "rows": "Rows", "revenue": "Line revenue"}),
                 hide_index=True)
    with st.expander("Stock codes set aside as non-product (largest by rows)"):
        st.dataframe(h["non_product"])
        st.caption("A code-pattern heuristic sets these aside; a few genuine-looking items are caught "
                   "too (about 0.013% of gross sales).")
    theme.callout("info", "Currency is not stated in the source; GBP is assumed. Revenue, profit, conversion "
                          "and retention are different metrics here, and only revenue and repeat behaviour "
                          "can be derived from this data.", title="Reading these numbers")


def _table_overview(ctx: logic.AppContext) -> None:
    raw = ctx.raw
    cols = [c for c in raw.columns if c != "is_simulated"]
    theme.kpi_row([
        ("Rows", f"{len(raw):,}", "", "teal"), ("Columns", str(len(cols)), "", "blue"),
        ("Missing cells", fmt_pct(raw[cols].isna().to_numpy().mean()), "", "amber"),
    ])
    st.dataframe(raw.head(10))
    gt = logic.ground_truth_for(ctx.info.key)
    if gt:
        with st.expander("Known ground truth (what a correct engine must find)"):
            st.json(gt)


def _upload_overview(ctx: logic.AppContext) -> None:
    frame = ctx.raw
    prepared = state.get_upload(ctx.info.key)
    cols = [c for c in frame.columns if c != "is_simulated"]
    items = [("Rows", f"{len(frame):,}", "", "teal"), ("Columns", str(len(cols)), "", "blue"),
             ("Missing cells", fmt_pct(frame[cols].isna().to_numpy().mean()), "", "amber")]
    if prepared is not None and prepared.report.date:
        d = prepared.report.date
        items.append(("Date range", f"{d['min']} to {d['max']}", f"{d.get('n_periods', 0)} periods", "teal"))
    theme.kpi_row(items)

    if prepared is not None:
        rep = prepared.report
        for w in rep.warnings:
            theme.callout("warn", w)
        if rep.notes:
            with st.expander(f"What was done to your file ({len(rep.notes)} step(s))"):
                st.markdown("\n".join(f"- {n}" for n in rep.notes))
        roles = prepared.mapping.roles()
        if roles:
            with st.expander("Column mapping used"):
                st.dataframe(pd.DataFrame([{"Role": r.replace("_", " "), "Your column": c} for r, c in roles.items()]),
                             hide_index=True)
    st.markdown("### Columns")
    prof = ingest.profile_columns(frame[cols])
    prof["Missing"] = prof["Missing"].map(fmt_pct)
    st.dataframe(prof, hide_index=True)
    st.markdown("### First rows")
    st.dataframe(frame.head(10), hide_index=True)
    theme.callout("info", "Uploaded data stays in this browser session on the machine running the app. Findings you "
                          "save are stored, but the file is not: upload the same file with the same mapping to re-run them.",
                  title="About your data")


def _availability(ctx: logic.AppContext) -> None:
    st.markdown("### What this dataset supports")
    access = logic.availability(state.get_frame(ctx.info.key, "M"))
    st.dataframe(pd.DataFrame(
        [{"Page": page, "Status": "Available" if a.enabled else "Disabled", "Why": a.reason}
         for page, a in access.items()]), hide_index=True)
