"""Report generation: findings to Markdown or a self-contained HTML page.

Findings on SIMULATED data are always labelled and listed in their own section, so they
cannot be mistaken for discoveries about real data.
"""
from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from core.contracts import Finding
from core.evidence import canonical_fingerprint, sanitize
from viz.charts import ChartError, build_figure

DISCLAIMER = (
    "Metric Mirage reports statistical patterns and investigation leads. It does not establish "
    "causal truths. Every finding lists alternative explanations; read them before acting."
)
REAL_HEADING = "Findings from the loaded data"
SIM_HEADING = "Simulated teaching findings (SIMULATED data, not discoveries about real data)"
_MAX_ROWS = 30


def _fmt(v: Any) -> str:
    if v is None:
        return "undefined"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def _key_numbers(data: Any, prefix: str = "", depth: int = 0, out: list | None = None) -> list[tuple[str, str]]:
    """Scalar leaves of a nested dict (two levels deep) plus short numeric lists such as CIs."""
    out = [] if out is None else out
    if isinstance(data, dict):
        for key, value in data.items():
            if len(out) >= _MAX_ROWS:
                break
            name = f"{prefix}{key}"
            if isinstance(value, dict) and depth < 2:
                _key_numbers(value, name + ".", depth + 1, out)
            elif (isinstance(value, list) and 0 < len(value) <= 4
                  and all(isinstance(x, (int, float)) or x is None for x in value)):
                out.append((name, "[" + ", ".join(_fmt(x) for x in value) + "]"))
            elif value is None or isinstance(value, (str, int, float, bool)):
                out.append((name, _fmt(value)))
    return out


def _label(text: str) -> str:
    return text.replace("_", " ").capitalize()


def _split(findings: Iterable[Finding]) -> tuple[list[Finding], list[Finding]]:
    items = list(findings)
    return [f for f in items if not f.is_simulated], [f for f in items if f.is_simulated]


def _params_json(f: Finding) -> str:
    return json.dumps(sanitize(f.parameters), indent=2, sort_keys=True)


# ------------------------------------------------------------------- markdown
def _md(text: Any) -> str:
    return str(text).replace("\n", " ").replace("|", "\\|")


def _md_card(i: int, f: Finding) -> list[str]:
    badge = " · **SIMULATED**" if f.is_simulated else ""
    out = [f"### {i}. {_md(f.title)}", "",
           f"*{_label(f.category.value)}* · evidence level: **{_label(f.evidence_level.value)}** · "
           f"data: {f.data_source}{badge}", "",
           f"- **Metric:** {_md(f.metric)}", f"- **Comparison:** {_md(f.comparison)}"]
    if f.affected:
        out.append(f"- **Affected:** {_md(f.affected)}")
    out += [f"- **Sample size:** {f.n:,}", f"- **Method:** {_md(f.method)}", ""]
    for heading, data in (("Evidence (key numbers)", f.evidence), ("Uncertainty", f.uncertainty)):
        rows = _key_numbers(sanitize(data)) if data else []
        if rows:
            out += [f"**{heading}**", "", "| Quantity | Value |", "|---|---|"]
            out += [f"| {_md(k)} | {_md(v)} |" for k, v in rows] + [""]
    out += ["**Alternative explanations**", ""] + [f"- {_md(a)}" for a in f.alternatives] + [""]
    if f.limitations:
        out += ["**Limitations**", ""] + [f"- {_md(x)}" for x in f.limitations] + [""]
    out += [f"**Recommended next step:** {_md(f.next_step)}", "",
            f"**Reproducibility:** fingerprint `{canonical_fingerprint(f)}`", "",
            "```json", _params_json(f), "```", ""]
    if f.chart_spec:
        out += [f"*Chart: {f.chart_spec.get('kind', '?')} (shown in the HTML report).*", ""]
    return out


def build_markdown_report(findings: Iterable[Finding], title: str = "Metric Mirage investigation report",
                          dataset_note: str | None = None) -> str:
    real, sim = _split(findings)
    lines = [f"# {title}", "", f"> {DISCLAIMER}", ""]
    if dataset_note:
        lines += [dataset_note, ""]
    lines += [f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} · "
              f"{len(real)} finding(s) from data, {len(sim)} simulated.", ""]
    if not real and not sim:
        lines.append("_No findings have been saved yet._")
    for heading, group in ((REAL_HEADING, real), (SIM_HEADING, sim)):
        if group:
            lines += [f"## {heading}", ""]
            for i, f in enumerate(group, 1):
                lines += _md_card(i, f)
    return "\n".join(lines)


# ----------------------------------------------------------------------- html
_CSS = """
:root { color-scheme: dark; --bg:#0B1220; --surface:#111A2E; --line:#22304D; --text:#E6EDF7;
        --muted:#8DA2C0; --teal:#2DD4BF; --blue:#3B82F6; --amber:#F5A524; }
body { background:var(--bg); color:var(--text); font-family:Inter,'Segoe UI',Arial,sans-serif;
       margin:0; padding:32px; line-height:1.55; }
main { max-width:980px; margin:0 auto; }
h1 { font-size:28px; margin-bottom:4px; } h2 { margin-top:40px; color:var(--teal); }
.disclaimer { border-left:3px solid var(--amber); padding:8px 14px; background:var(--surface); color:var(--muted); }
.card { background:var(--surface); border:1px solid var(--line); border-radius:12px; padding:20px 24px; margin:20px 0; }
.card h3 { margin-top:0; }
.badge { display:inline-block; font-size:12px; padding:2px 10px; border-radius:999px; border:1px solid var(--line);
         color:var(--muted); margin-right:6px; }
.badge.sim { color:#0B1220; background:var(--amber); border-color:var(--amber); font-weight:700; }
.badge.level-statistical_lead { color:var(--teal); border-color:var(--teal); }
.badge.level-confirmed { color:var(--blue); border-color:var(--blue); }
table { border-collapse:collapse; margin:8px 0 16px; } td { border-bottom:1px solid var(--line); padding:3px 14px 3px 0; }
td:first-child { color:var(--muted); } .muted { color:var(--muted); }
pre { background:var(--bg); border:1px solid var(--line); padding:12px; border-radius:8px; overflow-x:auto; font-size:12px; }
ul { margin-top:4px; }
"""


def _chart_html(f: Finding, state: dict[str, bool]) -> str:
    if not f.chart_spec:
        return ""
    try:
        fig = build_figure(f.chart_spec)
    except ChartError as exc:
        return f'<p class="muted">Chart unavailable: {html.escape(str(exc))}</p>'
    include = not state["plotly"]
    state["plotly"] = True
    return fig.to_html(full_html=False, include_plotlyjs=include, config={"displaylogo": False})


def _html_table(rows: list[tuple[str, str]]) -> str:
    body = "".join(f"<tr><td>{html.escape(k)}</td><td>{html.escape(v)}</td></tr>" for k, v in rows)
    return f"<table>{body}</table>"


def _html_card(i: int, f: Finding, state: dict[str, bool]) -> str:
    e = html.escape
    badges = [f'<span class="badge">{e(_label(f.category.value))}</span>',
              f'<span class="badge level-{e(f.evidence_level.value)}">{e(_label(f.evidence_level.value))}</span>',
              f'<span class="badge">data: {e(f.data_source)}</span>']
    if f.is_simulated:
        badges.append('<span class="badge sim">SIMULATED</span>')
    parts = [f'<section class="card"><h3>{i}. {e(f.title)}</h3><p>{"".join(badges)}</p>',
             f"<p><b>Metric:</b> {e(f.metric)}<br><b>Comparison:</b> {e(f.comparison)}"]
    if f.affected:
        parts.append(f"<br><b>Affected:</b> {e(f.affected)}")
    parts.append(f"<br><b>Sample size:</b> {f.n:,}<br><b>Method:</b> {e(f.method)}</p>")
    parts.append(_chart_html(f, state))
    for heading, data in (("Evidence (key numbers)", f.evidence), ("Uncertainty", f.uncertainty)):
        rows = _key_numbers(sanitize(data)) if data else []
        if rows:
            parts.append(f"<h4>{heading}</h4>{_html_table(rows)}")
    parts.append("<h4>Alternative explanations</h4><ul>" + "".join(f"<li>{e(a)}</li>" for a in f.alternatives) + "</ul>")
    if f.limitations:
        parts.append("<h4>Limitations</h4><ul>" + "".join(f"<li>{e(x)}</li>" for x in f.limitations) + "</ul>")
    parts.append(f"<p><b>Recommended next step:</b> {e(f.next_step)}</p>")
    parts.append(f'<details><summary class="muted">Reproducibility record · fingerprint {e(canonical_fingerprint(f))}</summary>'
                 f"<pre>{e(_params_json(f))}</pre></details></section>")
    return "".join(parts)


def build_html_report(findings: Iterable[Finding], title: str = "Metric Mirage investigation report",
                      dataset_note: str | None = None) -> str:
    """Self-contained page: plotly.js is embedded once, so it opens offline."""
    real, sim = _split(findings)
    e = html.escape
    state = {"plotly": False}
    body = [f"<h1>{e(title)}</h1>", f'<p class="disclaimer">{e(DISCLAIMER)}</p>']
    if dataset_note:
        body.append(f'<p class="muted">{e(dataset_note)}</p>')
    body.append(f'<p class="muted">Generated {e(datetime.now(timezone.utc).isoformat(timespec="seconds"))} · '
                f"{len(real)} finding(s) from data, {len(sim)} simulated.</p>")
    if not real and not sim:
        body.append('<p class="muted">No findings have been saved yet.</p>')
    for heading, group in ((REAL_HEADING, real), (SIM_HEADING, sim)):
        if group:
            body.append(f"<h2>{e(heading)}</h2>")
            body += [_html_card(i, f, state) for i, f in enumerate(group, 1)]
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">'
            f"<title>{e(title)}</title><style>{_CSS}</style></head><body><main>"
            + "".join(body) + "</main></body></html>")


def write_report(findings: Iterable[Finding], path: str | Path, fmt: str | None = None,
                 title: str = "Metric Mirage investigation report", dataset_note: str | None = None) -> Path:
    path = Path(path)
    fmt = fmt or path.suffix.lstrip(".").lower()
    if fmt in ("md", "markdown"):
        text = build_markdown_report(findings, title, dataset_note)
    elif fmt in ("html", "htm"):
        text = build_html_report(findings, title, dataset_note)
    else:
        raise ValueError("fmt must be 'md' or 'html'.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path