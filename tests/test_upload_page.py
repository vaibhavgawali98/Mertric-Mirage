"""Drive the Upload Data page with custom files, through Streamlit's AppTest harness."""
from pathlib import Path

import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from ui.views import upload as upload_page

APP = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture
def page(monkeypatch, tmp_path):
    """Return a function that opens the Upload Data page showing the given file."""
    monkeypatch.setenv("METRIC_MIRAGE_DATASET", "simpson")
    monkeypatch.setenv("METRIC_MIRAGE_EVIDENCE_DB", str(tmp_path / "evidence.db"))
    st.cache_resource.clear()
    st.cache_data.clear()

    def open_with(data: bytes, filename: str) -> AppTest:
        monkeypatch.setattr(upload_page, "_source", lambda: (data, filename))
        at = AppTest.from_file(APP, default_timeout=120).run()
        at.sidebar.radio(key="page").set_value("Upload Data").run()
        assert not at.exception, at.exception
        return at

    return open_with


def _submit(at):
    button = [b for b in at.button if "Check and prepare" in (b.label or "")]
    assert button, "the mapping form should be visible"
    button[0].click().run()
    assert not at.exception, at.exception


def _use_buttons(at):
    return [b for b in at.button if "Use this dataset" in (b.label or "")]


def _prepared(at):
    return at.session_state["_mm_preview"]["prepared"]


MESSY = ("Order No;Date;Customer;Region;Revenue\n"
         "A1;03/04/2021;C001;North;1.234,50\n"
         "A2;13/04/2021;C002;South;250,00\n"
         "A3;15/04/2021;C001;North;99,90\n"
         "A4;20/05/2021;C003;South;310,25\n").encode("utf-8")

AMBIGUOUS = ("id;when;amount\n"
             "1;03/04/2021;10\n2;05/06/2021;20\n3;07/08/2021;30\n4;02/03/2021;40\n").encode("utf-8")


def test_messy_semicolon_file_is_understood_and_can_be_used(page):
    at = page(MESSY, "messy.csv")
    sid = at.session_state["_mm_read"]["src_id"]
    assert at.session_state["_mm_read"]["rinfo"].delimiter == ";"
    suggestion, _ = at.session_state["_mm_read"]["suggestion"]
    assert (suggestion.date, suggestion.customer, suggestion.order_id, suggestion.value) == (
        "Date", "Customer", "Order No", "Revenue")
    _submit(at)
    p = _prepared(at)
    assert p.frame["order_value"].tolist() == [1234.5, 250.0, 99.9, 310.25]           # decimal commas read correctly
    assert p.frame["order_date"].min() == pd.Timestamp("2021-04-03")  # day-first inferred from 13/04
    assert not p.report.date_ambiguous and _use_buttons(at)
    _use_buttons(at)[0].click().run()
    assert at.session_state["dataset_key"] == p.key and at.session_state["page"] == "Command Center"
    assert sid.startswith(at.session_state["_mm_read"]["digest"])


def test_ambiguous_dates_block_until_the_user_decides(page):
    at = page(AMBIGUOUS, "dates.csv")
    sid = at.session_state["_mm_read"]["src_id"]
    _submit(at)
    assert _prepared(at).report.date_ambiguous and not _use_buttons(at)
    at.radio(key=f"mm_dord_{sid}").set_value("Day first (03/04/2021 = 3 April)")
    _submit(at)
    p = _prepared(at)
    assert not p.report.date_ambiguous and _use_buttons(at)
    assert p.frame["order_date"].min() == pd.Timestamp("2021-03-02")


def test_unusable_mapping_is_reported_not_crashed(page):
    at = page(MESSY, "messy.csv")
    sid = at.session_state["_mm_read"]["src_id"]
    at.selectbox(key=f"mm_value_{sid}").set_value("Region")          # a text column as the value
    _submit(at)
    preview = at.session_state["_mm_preview"]
    assert preview["prepared"] is None and "not numeric" in preview["error"] and not _use_buttons(at)


def test_bad_files_show_an_error_and_no_form(page):
    for data, name in [(b"", "empty.csv"), (b"a,b\n", "header_only.csv"), (b"%PDF-1.4", "scan.pdf")]:
        at = page(data, name)
        assert not [b for b in at.button if "Check and prepare" in (b.label or "")], name
