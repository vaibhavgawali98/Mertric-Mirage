from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from core import ingest as ig
from data.sample_upload import SAMPLE_NAME, sample_csv_bytes

APP = str(Path(__file__).resolve().parent.parent / "app.py")
PAGES = ["Command Center", "Upload Data", "Metric Integrity", "Contradiction Lab",
         "Cohort Observatory", "Experiment Lab", "Evidence Room"]


def _fresh(monkeypatch, tmp_path, dataset=None):
    if dataset:
        monkeypatch.setenv("METRIC_MIRAGE_DATASET", dataset)
    monkeypatch.setenv("METRIC_MIRAGE_EVIDENCE_DB", str(tmp_path / "evidence.db"))
    st.cache_resource.clear()
    st.cache_data.clear()


def _prepared_sample():
    data = sample_csv_bytes()
    raw, info = ig.read_table(data, SAMPLE_NAME)
    clean, cinfo = ig.clean_table(raw)
    mapping, _ = ig.suggest_mapping(clean)
    return ig.prepare_dataset(clean, mapping, ig.file_digest(data), SAMPLE_NAME, info.notes, cinfo)


def _visit_every_page(at):
    for page in PAGES:
        at.sidebar.radio(key="page").set_value(page).run()
        assert not at.exception, (page, [e.value for e in at.exception])


@pytest.mark.parametrize("dataset", ["simpson", "heavy_tail", "ab_test"])
def test_every_page_renders_without_exceptions(dataset, monkeypatch, tmp_path):
    _fresh(monkeypatch, tmp_path, dataset)
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception, at.exception
    _visit_every_page(at)


def test_uploaded_dataset_renders_every_page(monkeypatch, tmp_path):
    _fresh(monkeypatch, tmp_path, "simpson")
    prepared = _prepared_sample()
    at = AppTest.from_file(APP, default_timeout=180)
    at.session_state["_mm_uploads"] = {prepared.key: prepared}
    at.session_state["dataset_key"] = prepared.key
    at.run()
    assert not at.exception, at.exception
    assert at.session_state["dataset_key"] == prepared.key
    _visit_every_page(at)


def test_upload_page_flow_with_the_sample_file(monkeypatch, tmp_path):
    """Sample file -> check and prepare -> use this dataset -> the app switches to it."""
    _fresh(monkeypatch, tmp_path, "simpson")
    at = AppTest.from_file(APP, default_timeout=180).run()
    at.sidebar.radio(key="page").set_value("Upload Data").run()
    assert not at.exception, at.exception
    at.session_state["_mm_use_sample"] = True
    at.run()
    assert not at.exception, at.exception
    submit = [b for b in at.button if "Check and prepare" in (b.label or "")]
    assert submit, "the mapping form should be visible"
    submit[0].click().run()
    assert not at.exception, at.exception
    use = [b for b in at.button if "Use this dataset" in (b.label or "")]
    assert use, "a prepared dataset should offer 'Use this dataset for analysis'"
    use[0].click().run()
    assert not at.exception, at.exception
    assert at.session_state["dataset_key"].startswith("upload:")
    assert at.session_state["page"] == "Command Center"
    assert len(at.session_state["_mm_uploads"]) == 1
