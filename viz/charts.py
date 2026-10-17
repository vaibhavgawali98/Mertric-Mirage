"""Chart specs and figure building.

Each *_spec function returns a small, strict-JSON dict (plain lists, no NaN) that can be
stored inside a Finding. build_figure turns any spec into a themed Plotly figure, so a
saved finding can be redrawn later without the original data.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from core.cohorts import ConcentrationResult, RetentionResult
from core.contradiction import ContradictionResult, MetricSpec
from core.experiments import MeanTestResult, ProportionTestResult
from viz import theme

KINDS = ("bar", "line", "waterfall", "heatmap", "interval")


class ChartError(ValueError):
    """Raised for an invalid chart request or spec."""


def _num(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def _nums(seq) -> list[float | None]:
    return [_num(v) for v in seq]


# ----------------------------------------------------------------- spec makers
def decomposition_spec(res: ContradictionResult) -> dict[str, Any]:
    """Waterfall: baseline value, mix effect, rate effect (entry/exit if any), comparison value."""
    agg, cfg = res.aggregate, res.config
    labels = [str(cfg["baseline"]), "Mix effect", "Rate effect"]
    values = [agg["value_baseline"], res.mix_effect, res.rate_effect]
    measure = ["absolute", "relative", "relative"]
    text = [f"{agg['value_baseline']:.4g}", f"{res.mix_effect:+.4g}", f"{res.rate_effect:+.4g}"]
    if abs(res.entry_exit_effect) > 1e-12:
        labels.append("Entry/exit")
        values.append(res.entry_exit_effect)
        measure.append("relative")
        text.append(f"{res.entry_exit_effect:+.4g}")
    labels.append(str(cfg["comparison"]))
    values.append(0.0)               # a 'total' bar shows the running sum
    measure.append("total")
    text.append(f"{agg['value_comparison']:.4g}")

    running = [agg["value_baseline"]]
    for step in values[1:-1]:
        running.append(running[-1] + step)
    lo, hi = min(running), max(running)
    pad = (hi - lo) * 0.5 if hi > lo else (abs(hi) * 0.1 or 1.0)
    return {
        "kind": "waterfall",
        "title": f"{res.metric_name}: what moved it ({cfg['baseline']} to {cfg['comparison']})",
        "x": labels, "y": _nums(values), "measure": measure, "text": text,
        "y_title": res.metric_name, "y_range": [lo - pad, hi + pad],
    }


def segment_comparison_spec(res: ContradictionResult, top_n: int = 8) -> dict[str, Any]:
    """Grouped bars, overall plus the segments that contribute most to the change.
    Segments present in only one period are omitted."""
    seg = res.segments
    seg = seg[(seg["n_baseline"] > 0) & (seg["n_comparison"] > 0)].head(top_n)
    agg, cfg = res.aggregate, res.config
    x = ["Overall"] + [str(s) for s in seg["segment"]]

    def trace(period, value_key, n_key, seg_val, seg_n):
        return {
            "name": str(period),
            "x": x,
            "y": _nums([agg[value_key]] + list(seg[seg_val])),
            "n": [int(agg[n_key])] + [int(v) for v in seg[seg_n]],
        }

    return {
        "kind": "bar",
        "title": f"{res.metric_name} by {cfg['segment_col']}",
        "barmode": "group",
        "traces": [
            trace(cfg["baseline"], "value_baseline", "n_baseline", "value_baseline", "n_baseline"),
            trace(cfg["comparison"], "value_comparison", "n_comparison", "value_comparison", "n_comparison"),
        ],
        "y_title": res.metric_name,
    }


def segment_trend_spec(df: pd.DataFrame, metric: MetricSpec, period_col: str, segment_col: str,
                       top_n: int = 5, min_n: int = 30, periods: list[str] | None = None) -> dict[str, Any]:
    """Overall metric per period with the largest segments alongside. Segment points with
    fewer than min_n rows are left blank rather than drawn from thin data."""
    for col in (period_col, segment_col, metric.numerator, metric.denominator):
        if col is not None and col not in df.columns:
            raise ChartError(f"Column '{col}' not found.")
    df = df[df[period_col].notna()]
    frame = pd.DataFrame({
        "p": df[period_col].astype(str),
        "s": df[segment_col].astype("object").where(df[segment_col].notna(), "(missing)").astype(str),
        "x": pd.to_numeric(df[metric.numerator], errors="coerce"),
        "y": 1.0 if metric.denominator is None else pd.to_numeric(df[metric.denominator], errors="coerce"),
    }).dropna(subset=["x", "y"])
    if frame.empty:
        raise ChartError("No usable rows for the trend chart.")
    labels = [str(p) for p in periods] if periods is not None else sorted(frame["p"].unique())

    total = frame.groupby("p").agg(x=("x", "sum"), y=("y", "sum"))
    overall = (total["x"] / total["y"].where(total["y"] > 0)).reindex(labels)
    traces = [{"name": "Overall", "x": labels, "y": _nums(overall), "bold": True}]
    for seg in frame.groupby("s")["y"].sum().nlargest(top_n).index:
        sub = frame[frame["s"] == seg].groupby("p").agg(x=("x", "sum"), y=("y", "sum"), n=("x", "size"))
        value = (sub["x"] / sub["y"].where(sub["y"] > 0)).where(sub["n"] >= min_n).reindex(labels)
        traces.append({"name": str(seg), "x": labels, "y": _nums(value)})
    return {
        "kind": "line", "title": f"{metric.name} over time, overall and by {segment_col}",
        "traces": traces, "x_title": period_col, "y_title": metric.name,
    }


def retention_heatmap_spec(res: RetentionResult, max_offset: int | None = None) -> dict[str, Any]:
    """Cohort retention heatmap. Unobservable cells are blank; * marks small cohorts."""
    rates = res.rates
    if max_offset is not None:
        rates = rates.loc[:, [c for c in rates.columns if c <= max_offset]]
    labels = [f"{c} *" if not bool(res.reliable.get(c, True)) else str(c) for c in rates.index]
    later = rates.iloc[:, 1:].to_numpy(float) if rates.shape[1] > 1 else rates.to_numpy(float)
    zmax = float(np.nanmax(later)) if np.isfinite(later).any() else 1.0
    unit = {"M": "months", "Q": "quarters", "Y": "years"}[res.freq]
    note = " (* = small cohort)" if any(label.endswith(" *") for label in labels) else ""
    return {
        "kind": "heatmap",
        "title": f"Cohort retention: share of each cohort ordering again{note}",
        "z": [[_num(v) for v in row] for row in rates.to_numpy(float)],
        "x": [int(c) for c in rates.columns], "y": labels,
        "zmin": 0.0, "zmax": zmax or 1.0, "reverse_y": True,
        "x_title": f"{unit} since first order", "colorbar_title": "retained",
    }


def lorenz_spec(res: ConcentrationResult, max_points: int = 300) -> dict[str, Any]:
    lz = res.lorenz
    idx = np.unique(np.linspace(0, len(lz) - 1, min(max_points, len(lz))).astype(int))
    sub = lz.iloc[idx]
    return {
        "kind": "line", "title": f"Revenue concentration (Gini {res.gini:.2f})",
        "traces": [
            {"name": "Customers", "x": _nums(sub["customer_share"]), "y": _nums(sub["revenue_share"]), "bold": True},
            {"name": "Perfect equality", "x": [0.0, 1.0], "y": [0.0, 1.0], "dash": "dash"},
        ],
        "x_title": "Share of customers (smallest to largest)", "y_title": "Share of revenue",
        "x_range": [0, 1], "y_range": [0, 1],
    }


def distribution_spec(values, bins: int = 60, title: str = "Distribution", x_title: str = "value",
                      log_x: bool = True) -> dict[str, Any]:
    """Histogram with mean and median marked. Log bins suit skewed money values."""
    x = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(float)
    x = x[np.isfinite(x)]
    if log_x:
        x = x[x > 0]
    if x.size == 0:
        raise ChartError("No usable values for the distribution chart.")
    if x.min() == x.max():
        raise ChartError("All values are identical; there is no distribution to draw.")
    edges = (np.logspace(np.log10(x.min()), np.log10(x.max()), bins + 1) if log_x
             else np.linspace(x.min(), x.max(), bins + 1))
    counts, edges = np.histogram(x, bins=edges)
    centers = np.sqrt(edges[:-1] * edges[1:]) if log_x else (edges[:-1] + edges[1:]) / 2
    return {
        "kind": "bar", "title": title, "barmode": "overlay", "log_x": log_x,
        "traces": [{"name": "count", "x": _nums(centers), "y": [int(c) for c in counts]}],
        "marks": [{"name": f"Mean {x.mean():.4g}", "x": float(x.mean())},
                  {"name": f"Median {np.median(x):.4g}", "x": float(np.median(x))}],
        "x_title": x_title, "y_title": "count",
    }


def experiment_interval_spec(result: ProportionTestResult | MeanTestResult, metric: str,
                             min_effect: float | None = None) -> dict[str, Any]:
    if isinstance(result, ProportionTestResult):
        est, lo, hi = result.risk_difference, result.ci_low, result.ci_high
    else:
        est, lo, hi = result.mean_difference, result.ci_low, result.ci_high
    level = int(round((1 - result.alpha) * 100))
    return {
        "kind": "interval", "title": f"Treatment effect on {metric} ({level}% CI)",
        "labels": ["Treatment minus control"], "estimate": _nums([est]),
        "low": _nums([lo]), "high": _nums([hi]), "band": min_effect,
        "x_title": f"Difference in {metric}",
    }


def contradiction_chart_spec(res: ContradictionResult) -> dict[str, Any]:
    """The default card chart for an aggregate-versus-segment finding."""
    return decomposition_spec(res)


# --------------------------------------------------------------- figure builder
def _bar(fig: go.Figure, spec: dict[str, Any]) -> None:
    for t in spec["traces"]:
        bar = go.Bar(name=t["name"], x=t["x"], y=t["y"])
        if t.get("n") is not None:
            bar.customdata = t["n"]
            bar.hovertemplate = "%{x}<br>%{y:.4g} (n=%{customdata:,})<extra>%{fullData.name}</extra>"
        fig.add_trace(bar)
    fig.update_layout(barmode=spec.get("barmode", "group"))
    if spec.get("log_x"):
        fig.update_xaxes(type="log")
    tops = [v for t in spec["traces"] for v in t["y"] if v is not None]
    top = (max(tops) if tops else 1) * 1.05
    for mark in spec.get("marks", []):
        fig.add_trace(go.Scatter(x=[mark["x"], mark["x"]], y=[0, top], mode="lines",
                                 name=mark["name"], line=dict(dash="dash", width=2)))


def _line(fig: go.Figure, spec: dict[str, Any]) -> None:
    for t in spec["traces"]:
        fig.add_trace(go.Scatter(
            x=t["x"], y=t["y"], name=t["name"], mode="lines+markers", connectgaps=False,
            line=dict(width=4 if t.get("bold") else 2, dash=t.get("dash", "solid")),
        ))
    if spec.get("x_range"):
        fig.update_xaxes(range=spec["x_range"])
    if spec.get("y_range"):
        fig.update_yaxes(range=spec["y_range"])


def _waterfall(fig: go.Figure, spec: dict[str, Any]) -> None:
    fig.add_trace(go.Waterfall(
        x=spec["x"], y=spec["y"], measure=spec["measure"], text=spec.get("text"),
        textposition="outside", connector=dict(line=dict(color=theme.GRID)),
        increasing=dict(marker=dict(color=theme.TEAL)),
        decreasing=dict(marker=dict(color=theme.AMBER)),
        totals=dict(marker=dict(color=theme.BLUE)),
    ))
    if spec.get("y_range"):
        fig.update_yaxes(range=spec["y_range"])


def _heatmap(fig: go.Figure, spec: dict[str, Any]) -> None:
    fig.add_trace(go.Heatmap(
        z=spec["z"], x=spec["x"], y=spec["y"], zmin=spec.get("zmin"), zmax=spec.get("zmax"),
        colorscale=[[0.0, theme.SURFACE], [0.5, theme.BLUE], [1.0, theme.TEAL]],
        hoverongaps=False, xgap=1, ygap=1,
        colorbar=dict(title=spec.get("colorbar_title", ""), tickformat=".0%"),
        hovertemplate="cohort %{y}<br>+%{x}<br>%{z:.1%}<extra></extra>",
    ))
    if spec.get("reverse_y"):
        fig.update_yaxes(autorange="reversed")


def _interval(fig: go.Figure, spec: dict[str, Any]) -> None:
    rows = [(l, e, lo, hi) for l, e, lo, hi in zip(spec["labels"], spec["estimate"], spec["low"], spec["high"])
            if None not in (e, lo, hi)]
    if not rows:
        raise ValueError("no finite intervals to draw")
    labels, est, lo, hi = zip(*rows)
    fig.add_trace(go.Scatter(
        x=list(est), y=list(labels), mode="markers", showlegend=False,
        marker=dict(size=11, color=theme.TEAL),
        error_x=dict(type="data", symmetric=False, color=theme.TEAL, thickness=2,
                     array=[h - e for e, h in zip(est, hi)], arrayminus=[e - l for e, l in zip(est, lo)]),
    ))
    fig.add_vline(x=0, line=dict(color=theme.MUTED, dash="dash"))
    if spec.get("band"):
        fig.add_vrect(x0=-spec["band"], x1=spec["band"], fillcolor=theme.AMBER, opacity=0.12, line_width=0)


_BUILDERS = {"bar": _bar, "line": _line, "waterfall": _waterfall, "heatmap": _heatmap, "interval": _interval}


def build_figure(spec: dict[str, Any]) -> go.Figure:
    kind = spec.get("kind") if isinstance(spec, dict) else None
    if kind not in KINDS:
        raise ChartError(f"Unknown chart kind {kind!r}.")
    fig = go.Figure()
    try:
        _BUILDERS[kind](fig, spec)
    except (KeyError, TypeError, ValueError) as exc:
        raise ChartError(f"Malformed chart spec: {exc}") from exc
    theme.apply_theme(fig, spec.get("title"))
    if spec.get("x_title"):
        fig.update_xaxes(title_text=spec["x_title"])
    if spec.get("y_title"):
        fig.update_yaxes(title_text=spec["y_title"])
    return fig