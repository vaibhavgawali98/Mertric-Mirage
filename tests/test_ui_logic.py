import numpy as np
import pandas as pd
import pytest

from core.data_loader import classify_retail_rows, clean_types
from ui import logic
from data.synthetic import GROUND_TRUTH, make_ab_test, make_heavy_tail, make_simpson


def test_column_classification_on_simulated_tables():
    s = make_simpson()
    assert logic.numeric_columns(s) == ["converted"]          # ids and the bool flag are excluded
    assert set(logic.group_columns(s)) == {"period", "segment"}
    assert logic.is_binary(s["converted"]) and not logic.is_binary(make_heavy_tail()["order_value"])
    assert logic.datetime_columns(s) == ["session_date"]


def test_access_rules_explain_themselves():
    assert logic.contradiction_access(make_simpson()).enabled
    heavy = logic.contradiction_access(make_heavy_tail())
    assert not heavy.enabled and "two categorical" in heavy.reason
    cohort = logic.cohort_access(make_simpson())
    assert not cohort.enabled and "customer_id" in cohort.reason
    assert not logic.integrity_access(pd.DataFrame()).enabled
    assert logic.experiment_access(make_ab_test()).enabled
    assert set(logic.availability(make_simpson())) == {"Metric Integrity", "Contradiction Lab", "Cohort Observatory", "Experiment Lab"}


def test_period_helpers():
    assert [logic.infer_freq(x) for x in ["2010Q4", "2010-11", "2010", "Q1"]] == ["Q", "M", "Y", None]
    frame = pd.DataFrame({"order_date": pd.to_datetime(["2011-10-01", "2011-12-09"])})
    assert logic.period_is_incomplete(frame, "2011Q4", "Q")
    assert not logic.period_is_incomplete(frame, "2011Q3", "Q")
    assert not logic.period_is_incomplete(pd.DataFrame({"x": [1]}), "2011Q4", "Q")


def test_formatting_and_helpers():
    assert logic.fmt_num(None) == "undefined" and logic.fmt_num(float("nan")) == "undefined"
    assert logic.fmt_num(19699768.84) == "19,699,769" and logic.fmt_num(570.6925) == "570.7"
    assert logic.fmt_pct(0.0365) == "3.6%" or logic.fmt_pct(0.0365) == "3.7%"
    assert logic.index_of(["a", "b", "c"], ["z", "c"]) == 2 and logic.index_of(["a"], ["z"]) == 0
    assert logic.metric_name(make_simpson(), "converted", None) == "converted rate"
    assert logic.metric_name(make_heavy_tail(), "order_value", None) == "mean order_value"
    assert logic.metric_name(make_heavy_tail(), "order_value", "n") == "order_value per n"


def test_default_dataset_respects_the_environment(monkeypatch):
    monkeypatch.setenv(logic.ENV_DATASET, "ab_test")
    assert logic.default_dataset_key() == "ab_test"
    monkeypatch.setenv(logic.ENV_DATASET, "nonsense")
    assert logic.default_dataset_key() in logic.CATALOGUE


def test_ground_truth_keys_become_strings():
    gt = logic.ground_truth_for("simpson")
    assert all(isinstance(k, str) for k in gt["segment_rate"])
    assert gt["total_change"] == GROUND_TRUTH["simpson"]["total_change"]
    assert logic.ground_truth_for("retail") is None


def test_rebuild_frame_for_tables_and_matched_retail_windows():
    info = logic.CATALOGUE["simpson"]
    s = make_simpson()
    assert logic.rebuild_frame(info, s, {"freq": None, "matched_window": False}, "Q1", "Q2") is s
    with pytest.raises(ValueError):
        logic.rebuild_frame(info, s, {"freq": None, "matched_window": True}, "Q1", "Q2")

    cols = ["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "Customer ID", "Country"]
    rows = [[str(i), "85123A", "x", 1, d, 10.0, 1.0, "UK"]
            for i, d in enumerate(["2010-01-05", "2010-03-20", "2010-04-05", "2010-04-30"])]
    cleaned, _ = clean_types(pd.DataFrame(rows, columns=cols))
    raw = classify_retail_rows(cleaned)
    frame = logic.rebuild_frame(logic.CATALOGUE["retail"], raw, {"freq": "Q", "matched_window": True}, "2010Q1", "2010Q2")
    assert len(frame) == 3                      # the 20 March order lies beyond the 29 days observed in Q2