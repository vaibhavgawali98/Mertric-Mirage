"""Data access for the UI: cached built-in datasets and session-scoped uploads.

Built-in datasets are loaded once per server (st.cache_*). Uploaded datasets live only in the
user's session (st.session_state): they are never written to disk and never shared between
sessions. Their keys contain a hash of the file and the column mapping, so a key always
identifies exactly one dataset.
"""
from __future__ import annotations

import os
from typing import Any, Callable

import pandas as pd
import streamlit as st

from core.cohorts import ConcentrationResult, RetentionResult, cohort_retention, revenue_concentration
from core.data_loader import non_product_summary
from core.evidence import DEFAULT_EVIDENCE_DB, FindingStore
from core.ingest import PreparedDataset
from core.integrity import duplicate_summary, key_conflicts
from core.metrics import metric_summary
from ui import logic

UPLOADS_KEY = "_mm_uploads"
MEMO_KEY = "_mm_memo"
MAX_UPLOADS = 5


# ------------------------------------------------------------------- uploads
def _uploads() -> dict[str, PreparedDataset]:
    return st.session_state.setdefault(UPLOADS_KEY, {})


def register_upload(prepared: PreparedDataset) -> logic.DatasetInfo:
    """Store a prepared upload for this session (oldest are dropped beyond MAX_UPLOADS)."""
    store = _uploads()
    store.pop(prepared.key, None)
    store[prepared.key] = prepared
    while len(store) > MAX_UPLOADS:
        oldest = next(iter(store))
        store.pop(oldest)
        st.session_state.get(MEMO_KEY, {}).clear()
    return logic.upload_dataset_info(prepared)


def get_upload(key: str) -> PreparedDataset | None:
    return _uploads().get(key)


def remove_upload(key: str) -> None:
    _uploads().pop(key, None)
    st.session_state.get(MEMO_KEY, {}).clear()


def upload_infos() -> dict[str, logic.DatasetInfo]:
    return {k: logic.upload_dataset_info(p) for k, p in _uploads().items()}


def all_infos() -> dict[str, logic.DatasetInfo]:
    """Built-in datasets followed by this session's uploads."""
    return {**logic.CATALOGUE, **upload_infos()}


def info_for(key: str) -> logic.DatasetInfo | None:
    return all_infos().get(key)


def _memo(name: str, args: tuple, fn: Callable[[], Any]) -> Any:
    """Per-session memo for results computed from uploads (keeps big frames out of st.cache)."""
    memo = st.session_state.setdefault(MEMO_KEY, {})
    k = (name, args)
    if k not in memo:
        memo[k] = fn()
    return memo[k]


# ----------------------------------------------------------- built-in datasets
@st.cache_resource(show_spinner="Loading data...")
def _load_builtin(key: str) -> pd.DataFrame:
    return logic.load_raw(key)


@st.cache_resource(show_spinner="Preparing the analysis table...")
def _builtin_frame(key: str, freq: str = "M") -> pd.DataFrame:
    return logic.analysis_frame(logic.CATALOGUE[key], _load_builtin(key), freq)


def load_raw(key: str) -> pd.DataFrame:
    if logic.is_upload_key(key):
        prepared = get_upload(key)
        if prepared is None:
            raise KeyError(f"Uploaded dataset {key!r} is not loaded in this session.")
        return prepared.frame
    return _load_builtin(key)


def get_frame(key: str, freq: str = "M") -> pd.DataFrame:
    """The frame analyses run on. An upload is already one row per record, whatever `freq` says."""
    if logic.is_upload_key(key):
        return load_raw(key)
    return _builtin_frame(key, freq)


@st.cache_resource
def get_store() -> FindingStore:
    return FindingStore(os.environ.get(logic.ENV_EVIDENCE_DB) or DEFAULT_EVIDENCE_DB)


# ------------------------------------------------- retail-only cached summaries
@st.cache_data(show_spinner="Computing headline metrics...")
def headline_metrics(key: str) -> dict[str, float]:
    return metric_summary(_load_builtin(key))


@st.cache_data(show_spinner="Checking row classes and duplicates...")
def retail_health(key: str) -> dict:
    lines = _load_builtin(key)
    sales = lines[lines["row_class"] == "sale"]
    classes = (lines.groupby("row_class")["line_revenue"].agg(rows="size", revenue="sum")
               .reindex(["sale", "return", "non_product", "invalid_price", "inconsistent"]).fillna(0)
               .reset_index())
    total = float(sales["line_revenue"].sum())
    return {
        "n_rows": int(len(lines)),
        "date_min": str(lines["InvoiceDate"].min()), "date_max": str(lines["InvoiceDate"].max()),
        "classes": classes,
        "missing_customer_rows": int((~lines["has_customer"]).sum()),
        "missing_customer_revenue_share": float(sales.loc[~sales["has_customer"], "line_revenue"].sum() / total) if total else float("nan"),
        "duplicates": duplicate_summary(lines, logic.RAW_COLUMNS),
        "non_product": non_product_summary(lines).head(10),
    }


@st.cache_data(show_spinner="Checking duplicates...")
def dup_check(key: str) -> dict:
    return duplicate_summary(_load_builtin(key), logic.RAW_COLUMNS)


@st.cache_data(show_spinner="Checking invoice consistency (about a minute)...")
def invoice_key_conflicts(key: str) -> pd.DataFrame:
    return key_conflicts(_load_builtin(key), "Invoice", ["Customer ID", "InvoiceDate", "Country"])


# ----------------------------------------------------- cohorts (built-in and upload)
@st.cache_data(show_spinner="Building cohorts...")
def _retention_cached(key: str, freq: str, min_cohort_size: int) -> RetentionResult:
    return cohort_retention(_builtin_frame(key, "M"), freq=freq, min_cohort_size=min_cohort_size)


def retention(key: str, freq: str, min_cohort_size: int) -> RetentionResult:
    if logic.is_upload_key(key):
        return _memo("retention", (key, freq, min_cohort_size),
                     lambda: cohort_retention(get_frame(key, "M"), freq=freq, min_cohort_size=min_cohort_size))
    return _retention_cached(key, freq, min_cohort_size)


@st.cache_data(show_spinner="Measuring revenue concentration...")
def _concentration_cached(key: str) -> ConcentrationResult:
    return revenue_concentration(_builtin_frame(key, "M"), n_boot=500)


def concentration(key: str) -> ConcentrationResult:
    if logic.is_upload_key(key):
        return _memo("concentration", (key,), lambda: revenue_concentration(get_frame(key, "M"), n_boot=500))
    return _concentration_cached(key)
