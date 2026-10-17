import json

import pandas as pd
import plotly.graph_objects as go
import pytest

from core.cohorts import cohort_retention, revenue_concentration
from core.contradiction import MetricSpec, investigate_contradiction
from core.experiments import two_proportion_test
from data.synthetic import make_simpson
from viz import theme
from viz.charts import (
    ChartError, build_figure, decomposition_spec, distribution_spec, experiment_interval_spec,
    lorenz_spec, retention_heatmap_spec, segment_comparison_spec, segment_trend_spec,
)

RATE = MetricSpec("conversion rate", "converted")


def strict(spec):
    return json.dumps(spec, allow_nan=False)      # raises on NaN or inf


@pytest.fixture(scope="module")
def simpson_result():
    return investigate_contradiction(make_simpson(), RATE, "period", "Q1", "Q2", "segment",
                                     data_source="synthetic")


def _orders():
    rows = [("A", "2010-01-10", 10), ("A", "2010-02-10", 10), ("A", "2010-03-10", 10),
            ("B", "2010-01-15", 10), ("C", "2010-02-05", 10), ("C", "2010-04-30", 10),
            ("D", "2010-02-20", 10), ("E", "2010-03-15", 10)]
    df = pd.DataFrame(rows, columns=["customer_id", "order_date", "order_value"])
    df["order_date"] = pd.to_datetime(df["order_date"])
    return df


def test_waterfall_adds_up_to_the_comparison_value(simpson_result):
    spec = decomposition_spec(simpson_result)
    strict(spec)
    assert spec["x"] == ["Q1", "Mix effect", "Rate effect", "Q2"]
    assert spec["measure"] == ["absolute", "relative", "relative", "total"]
    assert spec["y"][:3] == pytest.approx([0.25, 0.15, -0.02])
    assert sum(spec["y"][:-1]) == pytest.approx(0.38)
    fig = build_figure(spec)
    assert isinstance(fig, go.Figure) and fig.data[0].type == "waterfall"
    assert fig.layout.paper_bgcolor == theme.BACKGROUND


def test_segment_comparison_has_overall_first(simpson_result):
    spec = segment_comparison_spec(simpson_result)
    strict(spec)
    assert spec["traces"][0]["x"][0] == "Overall" and len(spec["traces"]) == 2
    assert spec["traces"][0]["y"][0] == pytest.approx(0.25)
    assert spec["traces"][1]["y"][0] == pytest.approx(0.38)
    assert spec["traces"][0]["n"][0] == 1000
    assert len(build_figure(spec).data) == 2


def test_segment_trend_masks_thin_cells():
    spec = segment_trend_spec(make_simpson(), RATE, "period", "segment")
    strict(spec)
    assert len(spec["traces"]) == 3 and spec["traces"][0]["x"] == ["Q1", "Q2"]
    assert spec["traces"][0]["y"] == pytest.approx([0.25, 0.38])
    thin = segment_trend_spec(make_simpson(), RATE, "period", "segment", min_n=10_000)
    assert all(v is None for t in thin["traces"][1:] for v in t["y"])
    assert all(v is not None for v in thin["traces"][0]["y"])
    with pytest.raises(ChartError):
        segment_trend_spec(make_simpson(), RATE, "period", "nope")


def test_retention_heatmap_blanks_unobservable_cells_and_marks_small_cohorts():
    res = cohort_retention(_orders(), min_cohort_size=1)
    spec = retention_heatmap_spec(res)
    strict(spec)
    assert len(spec["z"]) == 3 and len(spec["z"][0]) == 4
    assert spec["z"][1][3] is None and spec["z"][0][1] == pytest.approx(0.5)
    assert spec["reverse_y"] is True and not any(l.endswith(" *") for l in spec["y"])
    small = retention_heatmap_spec(cohort_retention(_orders()))
    assert all(l.endswith(" *") for l in small["y"])
    assert build_figure(spec).data[0].type == "heatmap"


def test_lorenz_spec_starts_at_origin_and_ends_at_one():
    spec = lorenz_spec(revenue_concentration(_orders(), n_boot=5))
    strict(spec)
    line = spec["traces"][0]
    assert (line["x"][0], line["y"][0]) == (0.0, 0.0)
    assert line["x"][-1] == pytest.approx(1.0) and line["y"][-1] == pytest.approx(1.0)
    assert len(spec["traces"]) == 2


def test_distribution_spec_counts_every_positive_value():
    values = [1, 2, 3, 4, 5, 100, 1000, 0, -5]
    spec = distribution_spec(values, bins=10)
    strict(spec)
    assert sum(spec["traces"][0]["y"]) == 7 and spec["log_x"] is True
    assert spec["marks"][0]["name"].startswith("Mean") and spec["marks"][1]["name"].startswith("Median")
    assert len(build_figure(spec).data) == 3
    with pytest.raises(ChartError):
        distribution_spec([5, 5, 5])
    with pytest.raises(ChartError):
        distribution_spec([-1, -2])


def test_interval_spec_for_an_experiment():
    r = two_proportion_test(1000, 10000, 1200, 10000)
    spec = experiment_interval_spec(r, "conversion rate", min_effect=0.01)
    strict(spec)
    assert spec["estimate"][0] == pytest.approx(0.02) and spec["low"][0] < 0.02 < spec["high"][0]
    assert spec["band"] == 0.01 and len(build_figure(spec).data) == 1


def test_invalid_specs_raise_chart_error():
    for bad in [{"kind": "nope"}, {"kind": "bar"}, {"kind": "interval", "labels": ["a"],
                "estimate": [None], "low": [None], "high": [None]}, None]:
        with pytest.raises(ChartError):
            build_figure(bad)