import math

import pandas as pd
import pytest

from core.data_loader import classify_retail_rows, clean_types
from core.metrics import (
    MetricError, add_period, average_order_value, gross_sales, metric_summary,
    net_revenue, order_level_frame, repeat_rate, return_rate_value, safe_divide,
)

COLS = ["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "Customer ID", "Country"]


def _classified(rows):
    cleaned, _ = clean_types(pd.DataFrame(rows, columns=COLS))
    return classify_retail_rows(cleaned)


@pytest.fixture
def frame():
    return _classified([
        ["1001", "85123A", "x", 2, "2010-01-05", 5.0, 1.0, "UK"],     # sale 10
        ["1001", "85124B", "x", 1, "2010-01-05", 10.0, 1.0, "UK"],    # sale 10
        ["1002", "85123A", "x", 4, "2010-01-20", 2.5, 1.0, "UK"],     # sale 10
        ["1003", "85123A", "x", 1, "2010-01-21", 20.0, 2.0, "France"],# sale 20
        ["1004", "85123A", "x", 3, "2010-01-22", 4.0, None, "UK"],    # sale 12, no customer
        ["C1005", "85123A", "x", -1, "2010-01-23", 5.0, 1.0, "UK"],   # return -5
        ["1006", "POST", "x", 1, "2010-01-24", 15.0, 1.0, "UK"],      # non-product
    ])


def test_headline_metrics_match_hand_calculation(frame):
    s = metric_summary(frame)
    assert s["gross_sales"] == pytest.approx(62.0)
    assert s["returns_value"] == pytest.approx(5.0)
    assert s["net_revenue"] == pytest.approx(57.0)
    assert s["orders"] == 4
    assert s["active_customers"] == 2
    assert s["average_order_value"] == pytest.approx(15.5)
    assert s["revenue_per_customer"] == pytest.approx(25.0)   # (10+10+10+20) / 2
    assert s["return_rate_value"] == pytest.approx(5 / 62)
    assert s["repeat_rate"] == pytest.approx(0.5)


def test_zero_denominators_return_nan():
    empty = _classified([["C1", "85123A", "x", -1, "2010-01-01", 5.0, 1.0, "UK"]])
    assert gross_sales(empty) == 0
    assert math.isnan(average_order_value(empty))
    assert math.isnan(return_rate_value(empty))
    assert math.isnan(repeat_rate(empty))
    assert math.isnan(safe_divide(1, 0))
    assert net_revenue(empty) == pytest.approx(-5.0)


def test_unclassified_frame_is_rejected():
    with pytest.raises(MetricError):
        gross_sales(pd.DataFrame({"a": [1]}))


def test_order_level_frame_has_one_row_per_invoice(frame):
    orders = order_level_frame(frame, freq="M")
    assert len(orders) == 4
    assert orders["order_value"].sum() == pytest.approx(62.0)
    assert set(orders["period"]) == {"2010-01"}
    assert "Country" in orders.columns
    assert orders.set_index("invoice_id").loc["1001", "n_lines"] == 2


def test_add_period_labels_and_errors():
    df = pd.DataFrame({"d": ["2010-11-05", "2011-02-01", None]})
    assert list(add_period(df, "d", "M")["period"].iloc[:2]) == ["2010-11", "2011-02"]
    assert add_period(df, "d", "Q")["period"].iloc[0] == "2010Q4"
    with pytest.raises(MetricError):
        add_period(df, "d", "W")
    with pytest.raises(MetricError):
        add_period(pd.DataFrame({"d": [None]}), "d")