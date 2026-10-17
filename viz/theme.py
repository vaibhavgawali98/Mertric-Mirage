"""Brand constants and the shared Plotly layout for METRIC MIRAGE."""
import plotly.graph_objects as go

BACKGROUND = "#0B1220"   # midnight
SURFACE = "#111A2E"
GRID = "#22304D"
TEXT = "#E6EDF7"
MUTED = "#8DA2C0"
TEAL = "#2DD4BF"
BLUE = "#3B82F6"
AMBER = "#F5A524"
RED = "#F87171"
COLORWAY = [TEAL, BLUE, AMBER, "#A78BFA", "#F472B6", "#94A3B8"]
FONT_FAMILY = "Inter, Segoe UI, Helvetica Neue, Arial, sans-serif"


def apply_theme(fig: go.Figure, title: str | None = None, height: int = 420) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BACKGROUND,
        plot_bgcolor=SURFACE,
        font=dict(family=FONT_FAMILY, color=TEXT, size=13),
        colorway=COLORWAY,
        height=height,
        margin=dict(l=64, r=32, t=64 if title else 32, b=56),
        title=dict(text=title or "", x=0.01, xanchor="left", font=dict(size=16)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hoverlabel=dict(bgcolor=SURFACE, font_color=TEXT),
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID)
    return fig