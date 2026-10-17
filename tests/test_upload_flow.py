"""Tests for how uploaded data flows through the interface layer and the analysis engines."""
from types import SimpleNamespace

import pandas as pd
import pytest

from core import ingest as ig
from data.sample_upload import SAMPLE_NAME, sample_csv_bytes
from ui import logic, state


@pytest.fixture(scope="module")
def prepared():
    data = sample_csv_bytes()
    raw, info = ig.read_table(data, SAMPLE_NAME)
    clean, cinfo = ig.clean_table(raw)
    mapping, _ = ig.suggest_mapping(clean)
    return ig.prepare_dataset(clean, mapping, ig.file_digest(data), SAMPLE_NAME, info.notes, cinfo)


@pytest.fixture
def fake_session(monkeypatch):
    """Run the state helpers without a Streamlit server: st.session_state becomes a plain dict."""
    monkeypatch.setattr(state, "st", SimpleNamespace(session_state={}))
    return state.st.session_state


# ------------------------------------------------------------------- logic
def test_upload_dataset_info_describes_the_file(prepared):
    info = logic.upload_dataset_info(prepared)
    assert info.key == prepared.key and info.kind == "upload" and info.freq == "M"
    assert info.label.startswith("SIMULATED FILE") and SAMPLE_NAME in info.label     # is_simulated column set
    assert "not stored on disk" in info.description and "value = revenue" in info.description
    plain = ig.prepare_dataset(pd.DataFrame({"v": ["1", "2"] * 5}), ig.Mapping(value="v"), "s", "mine.csv")
    plain_info = logic.upload_dataset_info(plain)
    assert plain_info.label == "YOUR DATA: mine.csv" and plain_info.source == "upload"


def test_upload_key_detection():
    assert logic.is_upload_key("upload:abc123") and not logic.is_upload_key("retail") and not logic.is_upload_key(None)


def test_prepared_sample_enables_every_analysis(prepared):
    access = logic.availability(prepared.frame)
    assert all(a.enabled for a in access.values()), {k: v.reason for k, v in access.items() if not v.enabled}


def test_identifier_columns_are_never_offered_as_metrics_or_segments(prepared):
    f = prepared.frame
    assert "order_value" in logic.numeric_columns(f)
    for forbidden in ("customer_id", "order_id"):
        assert forbidden not in logic.numeric_columns(f) and forbidden not in logic.group_columns(f)
    assert {"period", "region", "channel"} <= set(logic.group_columns(f))


def test_id_like_names_extended():
    for name in ["id", "ID", "user_id", "Customer ID", "order-id", "invoice"]:
        assert logic.ID_LIKE.search(name), name
    for name in ["paid", "valid", "idea", "revenue"]:
        assert not logic.ID_LIKE.search(name), name


def test_rebuild_frame_for_uploads(prepared):
    info = logic.upload_dataset_info(prepared)
    same = logic.rebuild_frame(info, prepared.frame, {"freq": "M", "matched_window": False}, "2023-01", "2024-06")
    assert same is prepared.frame
    matched = logic.rebuild_frame(info, prepared.frame, {"freq": "M", "matched_window": True}, "2023-01", "2024-06")
    assert 0 < len(matched) <= len(prepared.frame)


# ------------------------------------------------------------------- state
def test_register_get_and_remove_uploads(fake_session, prepared):
    info = state.register_upload(prepared)
    assert info.key == prepared.key and state.get_upload(prepared.key) is prepared
    assert prepared.key in state.upload_infos() and prepared.key in state.all_infos() and "retail" in state.all_infos()
    assert state.info_for(prepared.key).kind == "upload" and state.info_for("nope") is None
    assert state.load_raw(prepared.key) is prepared.frame and state.get_frame(prepared.key, "Q") is prepared.frame
    state.remove_upload(prepared.key)
    assert state.get_upload(prepared.key) is None
    with pytest.raises(KeyError):
        state.load_raw(prepared.key)


def test_only_the_newest_uploads_are_kept(fake_session):
    keys = []
    for i in range(state.MAX_UPLOADS + 2):
        p = ig.prepare_dataset(pd.DataFrame({"v": ["1", "2"] * 5}), ig.Mapping(value="v"), f"src{i}", f"f{i}.csv")
        state.register_upload(p)
        keys.append(p.key)
    assert len(state.upload_infos()) == state.MAX_UPLOADS
    assert keys[0] not in state.upload_infos() and keys[-1] in state.upload_infos()


def test_upload_results_are_computed_once_per_session(fake_session, prepared):
    state.register_upload(prepared)
    first = state.retention(prepared.key, "M", 30)
    assert state.retention(prepared.key, "M", 30) is first
    conc = state.concentration(prepared.key)
    assert state.concentration(prepared.key) is conc and 0 < conc.gini < 1


# ------------------------------------------- the real engines accept uploaded data
def test_contradiction_engine_on_an_uploaded_frame(prepared):
    from core.contradiction import MetricSpec, investigate_contradiction

    res = investigate_contradiction(prepared.frame, MetricSpec("average order value", "order_value"),
                                    "period", "2023-01", "2024-06", "channel", min_segment_n=10,
                                    data_source=prepared.source)
    assert res.verdict in logic.VERDICTS and res.aggregate["n_baseline"] > 0 and res.aggregate["n_comparison"] > 0
    total = res.mix_effect + res.rate_effect + res.entry_exit_effect
    assert total == pytest.approx(res.aggregate["change"], abs=1e-9)
    finding = res.to_finding()
    assert finding is None or finding.is_simulated


def test_cohort_and_concentration_engines_on_an_uploaded_frame(prepared):
    from core.cohorts import cohort_retention, repeat_purchase_gaps, revenue_concentration

    ret = cohort_retention(prepared.frame, "M", 30)
    assert ret.counts.shape[0] >= 15 and (ret.rates.iloc[:, 0].dropna() == 1.0).all()
    conc = revenue_concentration(prepared.frame, n_boot=20)
    assert conc.n_customers == prepared.frame["customer_id"].nunique() and 0 < conc.gini < 1
    summary, gaps = repeat_purchase_gaps(prepared.frame)
    assert summary["n_customers"] == conc.n_customers and len(gaps) > 0


def test_integrity_and_experiment_engines_on_an_uploaded_frame(prepared):
    from core.experiments import two_mean_test
    from core.integrity import (change_point_candidate, compare_periods_robust, distribution_summary,
                                missingness_by_column)
    from core.metrics import period_series

    f = prepared.frame
    assert distribution_summary(f["order_value"])["n"] == len(f)
    assert missingness_by_column(f)["n_missing"].sum() == 0
    cmp = compare_periods_robust(f, "order_value", "period", "2023-01", "2024-06")
    assert cmp.n_baseline > 0 and cmp.n_comparison > 0
    series, _ = period_series(f, "order_value", "period", "order_date", "M", "mean")
    assert change_point_candidate(series, n_perm=99).n == len(series)
    web = f.loc[f["channel"] == "Web", "order_value"]
    store = f.loc[f["channel"] == "Store", "order_value"]
    assert two_mean_test(web, store, n_boot=50).n_control == len(web)
