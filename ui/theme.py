"""Visual identity: CSS, KPI cards, callouts and the investigation-card renderer."""
from __future__ import annotations

import html
import json
from typing import Any, Sequence

import pandas as pd
import streamlit as st

from core.contracts import Finding
from core.evidence import canonical_fingerprint, sanitize
from core.reports import _key_numbers as key_numbers   # shared flattening of nested evidence
from viz.charts import ChartError, build_figure

CSS = """
<style>
.block-container { padding-top: 2rem; max-width: 1400px; }
h1, h2, h3, h4 { letter-spacing: -0.01em; }
.mm-brand { font-size: 1.3rem; font-weight: 800; letter-spacing: .16em; color: #2DD4BF; }
.mm-tag { color: #8DA2C0; font-size: .78rem; margin-bottom: 1rem; }
.mm-kpi { background: #111A2E; border: 1px solid #22304D; border-radius: 14px; padding: 14px 18px; height: 100%; }
.mm-kpi .label { color: #8DA2C0; font-size: .74rem; text-transform: uppercase; letter-spacing: .08em; }
.mm-kpi .value { font-size: 1.7rem; font-weight: 700; margin-top: 2px; line-height: 1.2; }
.mm-kpi .sub { color: #8DA2C0; font-size: .78rem; margin-top: 2px; }
.mm-kpi.teal { border-top: 3px solid #2DD4BF; }
.mm-kpi.blue { border-top: 3px solid #3B82F6; }
.mm-kpi.amber { border-top: 3px solid #F5A524; }
.mm-callout { border-left: 4px solid #3B82F6; background: #111A2E; border-radius: 8px; padding: 10px 14px; margin: 10px 0; }
.mm-callout .title { font-weight: 700; margin-bottom: 2px; }
.mm-callout.warn { border-color: #F5A524; } .mm-callout.error { border-color: #F87171; } .mm-callout.ok { border-color: #2DD4BF; }
.mm-badge { display: inline-block; font-size: .72rem; padding: 2px 10px; border-radius: 999px; border: 1px solid #22304D; color: #8DA2C0; margin-right: 6px; }
.mm-badge.sim { background: #F5A524; border-color: #F5A524; color: #0B1220; font-weight: 700; }
.mm-badge.lead { color: #2DD4BF; border-color: #2DD4BF; }
.mm-badge.confirmed { color: #3B82F6; border-color: #3B82F6; }
</style>
"""

DISCLAIMER = ("Metric Mirage surfaces statistical patterns and investigation leads. It does not "
              "establish causal truths.")


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def esc(x: Any) -> str:
    return html.escape(str(x))


def page_header(title: str, subtitle: str) -> None:
    st.title(title)
    st.caption(subtitle)


def kpi(label: str, value: str, sub: str = "", tone: str = "teal") -> str:
    return (f'<div class="mm-kpi {esc(tone)}"><div class="label">{esc(label)}</div>'
            f'<div class="value">{esc(value)}</div><div class="sub">{esc(sub)}</div></div>')


def kpi_row(items: Sequence[Sequence[str]]) -> None:
    """items: (label, value[, sub[, tone]]) tuples."""
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        label, value, *rest = item
        sub = rest[0] if len(rest) > 0 else ""
        tone = rest[1] if len(rest) > 1 else "teal"
        col.markdown(kpi(label, value, sub, tone), unsafe_allow_html=True)


def callout(kind: str, text: str, title: str | None = None) -> None:
    head = f'<div class="title">{esc(title)}</div>' if title else ""
    st.markdown(f'<div class="mm-callout {esc(kind)}">{head}<div>{esc(text)}</div></div>', unsafe_allow_html=True)


def disabled_panel(name: str, reason: str) -> None:
    callout("info", reason, title=f"{name} is not available for this dataset")


def simulated_banner() -> None:
    callout("warn", "This is SIMULATED teaching data with known ground truth. Findings from it are "
                    "labelled simulated and are never discoveries about the public dataset.",
            title="Simulated data")


def _label(value: str) -> str:
    return value.replace("_", " ").capitalize()


def badge(text: str, tone: str = "") -> str:
    return f'<span class="mm-badge {esc(tone)}">{esc(text)}</span>'


def render_finding(f: Finding, key: str) -> None:
    """One investigation card: claim, evidence, chart, alternatives, limits, reproducibility."""
    st.markdown(f"#### {f.title}")
    level_tone = {"statistical_lead": "lead", "confirmed": "confirmed"}.get(f.evidence_level.value, "")
    badges = [badge(_label(f.category.value)), badge(_label(f.evidence_level.value), level_tone),
              badge(f"data: {f.data_source}")]
    if f.is_simulated:
        badges.append(badge("SIMULATED", "sim"))
    st.markdown("".join(badges), unsafe_allow_html=True)
    lines = [f"**Metric:** {f.metric}", f"**Comparison:** {f.comparison}"]
    if f.affected:
        lines.append(f"**Affected:** {f.affected}")
    lines += [f"**Sample size:** {f.n:,}", f"**Method:** {f.method}"]
    st.markdown("  \n".join(lines))
    if f.chart_spec:
        try:
            st.plotly_chart(build_figure(f.chart_spec), key=f"{key}_chart")
        except ChartError as exc:
            st.caption(f"Chart unavailable: {exc}")
    for heading, data in (("Evidence (key numbers)", f.evidence), ("Uncertainty", f.uncertainty)):
        rows = key_numbers(sanitize(data)) if data else []
        if rows:
            st.markdown(f"**{heading}**")
            st.dataframe(pd.DataFrame(rows, columns=["Quantity", "Value"]), hide_index=True)
    st.markdown("**Alternative explanations**")
    st.markdown("\n".join(f"- {a}" for a in f.alternatives))
    if f.limitations:
        st.markdown("**Limitations**")
        st.markdown("\n".join(f"- {x}" for x in f.limitations))
    st.markdown(f"**Recommended next step:** {f.next_step}")
    with st.expander("Reproducibility record"):
        st.caption(f"Fingerprint {canonical_fingerprint(f)}")
        st.code(json.dumps(sanitize(f.parameters), indent=2, sort_keys=True), language="json")