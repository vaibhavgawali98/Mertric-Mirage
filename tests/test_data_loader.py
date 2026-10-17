import numpy as np
import pandas as pd
import pytest

from core.data_loader import (
    classify_retail_rows,
    clean_types,
    reconcile_sheet_overlap,
    validate_retail_frame,
)
from core.schema import SchemaError

COLS = ["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "Customer ID", "Country"]


def _row(inv, code, qty, date, price=2.0, cust=1.0, desc="X", country="UK"):
    return [inv, code, desc, qty, date, price, cust, country]


def _frame(rows):
    return pd.DataFrame(rows, columns=COLS)


def test_row_classification_covers_each_case():
    rows = [
        _row("1001", "85123A", 3, "2010-01-01"),               # sale
        _row("C1002", "85123A", -1, "2010-01-02"),             # return
        _row("1003", "POST", 1, "2010-01-03"),                 # non_product
        _row("1004", "85123A", 1, "2010-01-04", price=0.0),    # invalid_price
        _row("1005", "85123A", -5, "2010-01-05"),              # inconsistent (neg qty, not 'C')
        _row("C1006", "85123A", 2, "2010-01-06"),              # inconsistent ('C' with positive qty)
    ]
    df, _ = clean_types(_frame(rows))
    out = classify_retail_rows(df)
    assert list(out["row_class"]) == [
        "sale", "return", "non_product", "invalid_price", "inconsistent", "inconsistent",
    ]
    assert out.loc[0, "line_revenue"] == pytest.approx(6.0)


def test_clean_types_counts_failed_coercions():
    rows = [
        _row("1", "85123A", 1, "2010-01-01"),
        _row("2", "85123A", "abc", "not a date"),
    ]
    df, coerced = clean_types(_frame(rows))
    assert coerced["InvoiceDate"] == 1
    assert coerced["Quantity"] == 1
    assert pd.isna(df.loc[1, "InvoiceDate"])


def test_validation_counts_missing_customers_and_duplicates():
    rows = [
        _row("1", "85123A", 1, "2010-01-01"),
        _row("1", "85123A", 1, "2010-01-01"),                  # exact duplicate
        _row("2", "85123A", 1, "2010-01-02", cust=None),       # no customer
    ]
    df, coerced = clean_types(_frame(rows))
    report = validate_retail_frame(classify_retail_rows(df), coerced=coerced)
    assert report.n_rows == 3
    assert report.exact_duplicate_rows == 1
    assert report.missing_customer_rows == 1
    assert report.row_class_counts["sale"] == 3


def test_validation_rejects_empty_inputs():
    with pytest.raises(SchemaError):
        validate_retail_frame(pd.DataFrame())
    empty = pd.DataFrame(columns=COLS)
    with pytest.raises(SchemaError):
        validate_retail_frame(empty)


def _overlap_frames(change_second_quantity=False):
    first = _frame([
        _row("A", "85123A", 1, "2010-11-30"),
        _row("B", "85123A", 1, "2010-12-02"),
        _row("C", "85123A", 1, "2010-12-08"),
    ])
    second = _frame([
        _row("B", "85123A", 9 if change_second_quantity else 1, "2010-12-02"),
        _row("C", "85123A", 1, "2010-12-08"),
        _row("D", "85123A", 1, "2010-12-20"),
    ])
    return first, second


def test_identical_overlap_is_removed():
    first, second = _overlap_frames()
    combined, report = reconcile_sheet_overlap(first, second)
    assert report.found and report.identical
    assert report.dropped == 2
    assert len(combined) == 4


def test_different_overlap_is_kept_and_flagged():
    first, second = _overlap_frames(change_second_quantity=True)
    combined, report = reconcile_sheet_overlap(first, second)
    assert report.found and not report.identical
    assert report.dropped == 0
    assert len(combined) == 6


def test_no_overlap_concatenates():
    first = _frame([_row("A", "85123A", 1, "2010-01-01")])
    second = _frame([_row("B", "85123A", 1, "2010-03-01")])
    combined, report = reconcile_sheet_overlap(first, second)
    assert not report.found
    assert len(combined) == 2