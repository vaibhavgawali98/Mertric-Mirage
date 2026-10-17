"""Streamlit-free logic for the interface, so it can be unit-tested."""
from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from core.cohorts import REQUIRED_COLUMNS
from core.data_loader import PROCESSED_PATH, load_processed
from core.ingest import UPLOAD_PREFIX, PreparedDataset
from core.metrics import matched_window, order_level_frame
from data import synthetic as syn

RAW_COLUMNS = ["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate",
               "Price", "Customer ID", "Country"]
ENV_DATASET = "METRIC_MIRAGE_DATASET"
ENV_EVIDENCE_DB = "METRIC_MIRAGE_EVIDENCE_DB"
# Identifier-style column names: id, ID, user_id, "Customer ID", invoice. Never offered as metrics.
ID_LIKE = re.compile(r"(^id$|[\s_\-]id$|^invoice$)", re.IGNORECASE)


# ------------------------------------------------------------------ datasets
@dataclass(frozen=True)
class DatasetInfo:
    key: str
    label: str
    source: str        # "public" | "synthetic" | "upload"
    kind: str          # "retail" (invoice lines) | "table" (analysis-ready) | "upload" (prepared upload)
    description: str
    freq: str | None = None   # period grain of an uploaded dataset's `period` column


CATALOGUE: dict[str, DatasetInfo] = {
    "retail": DatasetInfo(
        "retail", "UCI Online Retail II (real, public)", "public", "retail",
        "Real transactions from a UK online retailer, December 2009 to December 2011. Grain: one "
        "row per invoice line; order-level analyses use one row per sale invoice. The source has "
        "no cost, margin, campaign or acquisition fields, so profit and acquisition cost cannot "
        "be analysed. See the UCI page for licence and citation terms."),
    "simpson": DatasetInfo(
        "simpson", "SIMULATED: Simpson's paradox", "synthetic", "table",
        "Conversion by quarter and customer type, built so the overall rate rises while both "
        "segments fall."),
    "mix_shift": DatasetInfo(
        "mix_shift", "SIMULATED: mix shift only", "synthetic", "table",
        "Segment rates are unchanged; only the customer mix moves."),
    "magnitude": DatasetInfo(
        "magnitude", "SIMULATED: magnitude only (no paradox)", "synthetic", "table",
        "Everything rises and the mix is unchanged. A correct detector must NOT flag this."),
    "null_case": DatasetInfo(
        "null_case", "SIMULATED: null case", "synthetic", "table",
        "No planted difference anywhere. A correct detector reports nothing."),
    "heavy_tail": DatasetInfo(
        "heavy_tail", "SIMULATED: heavy tail", "synthetic", "table",
        "Same order-value distribution in both quarters plus five huge orders in Q2: the mean "
        "jumps, the median does not."),
    "ab_test": DatasetInfo(
        "ab_test", "SIMULATED: randomized A/B test", "synthetic", "table",
        "A randomized test with exact counts: control 10.0% versus treatment 12.0% conversion."),
}

SYNTHETIC_MAKERS = {
    "simpson": syn.make_simpson, "mix_shift": syn.make_mix_shift, "magnitude": syn.make_magnitude_only,
    "null_case": syn.make_null_case, "heavy_tail": syn.make_heavy_tail, "ab_test": syn.make_ab_test,
}


@dataclass(frozen=True)
class AppContext:
    info: DatasetInfo
    raw: pd.DataFrame      # retail: classified invoice lines; others: the table itself


def is_upload_key(key: Any) -> bool:
    return isinstance(key, str) and key.startswith(UPLOAD_PREFIX)


def upload_dataset_info(p: PreparedDataset) -> DatasetInfo:
    """Catalogue entry for a prepared upload."""
    simulated = p.source == "synthetic"
    label = ("SIMULATED FILE: " if simulated else "YOUR DATA: ") + p.name
    roles = ", ".join(f"{role.replace('_', ' ')} = {col}" for role, col in p.mapping.roles().items()) or "no roles mapped"
    description = (f"Uploaded file '{p.name}': {p.report.rows_out:,} rows, {p.report.columns_out} columns "
                   f"({roles}). The data stays in this browser session and is not stored on disk.")
    return DatasetInfo(p.key, label, p.source, "upload", description, p.freq)


def default_dataset_key() -> str:
    env = os.environ.get(ENV_DATASET)
    if env in CATALOGUE:
        return env
    return "retail" if Path(PROCESSED_PATH).exists() else "simpson"


def load_raw(key: str) -> pd.DataFrame:
    if key not in CATALOGUE:
        raise KeyError(f"Unknown dataset {key!r}.")
    return load_processed() if key == "retail" else SYNTHETIC_MAKERS[key]()


def analysis_frame(info: DatasetInfo, raw: pd.DataFrame, freq: str = "M") -> pd.DataFrame:
    """The frame analyses run on: one row per sale invoice for retail, the table otherwise."""
    return order_level_frame(raw, freq=freq) if info.kind == "retail" else raw


def rebuild_frame(info: DatasetInfo, raw: pd.DataFrame, prep: dict[str, Any],
                  baseline: Any, comparison: Any) -> pd.DataFrame:
    """Rebuild the exact frame a saved finding was computed on, from its `data_prep` record."""
    freq = prep.get("freq")
    frame = analysis_frame(info, raw, freq or "M")
    if prep.get("matched_window"):
        if not freq:
            raise ValueError("A matched window needs a period grain.")
        frame, _ = matched_window(frame, "order_date", "period", baseline, comparison, freq)
    return frame


def ground_truth_for(key: str) -> dict[str, Any] | None:
    gt = syn.GROUND_TRUTH.get(key)
    return None if gt is None else _stringify(gt)


def _stringify(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _stringify(v) for k, v in obj.items()}
    return obj


# ------------------------------------------------------------ column helpers
def is_binary(series: pd.Series) -> bool:
    vals = pd.to_numeric(series, errors="coerce").dropna().unique()
    return len(vals) > 0 and set(vals.tolist()) <= {0, 1}


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])
            and not ID_LIKE.search(str(c))]


def group_columns(df: pd.DataFrame, max_levels: int = 60) -> list[str]:
    """Categorical columns usable as period, segment or arm (2 to max_levels distinct values)."""
    out = []
    for c in df.columns:
        s = df[c]
        if ID_LIKE.search(str(c)) or pd.api.types.is_bool_dtype(s):
            continue
        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s):
            continue
        if 2 <= s.nunique(dropna=True) <= max_levels:
            out.append(c)
    return out


def datetime_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]


def period_labels(df: pd.DataFrame, col: str) -> list[str]:
    return sorted(str(v) for v in df[col].dropna().unique())


def index_of(options: list, preferred: list) -> int:
    opts = list(options)
    for p in preferred:
        if p in opts:
            return opts.index(p)
    return 0


def metric_name(frame: pd.DataFrame, numerator: str, denominator: str | None) -> str:
    if denominator:
        return f"{numerator} per {denominator}"
    return f"{numerator} rate" if is_binary(frame[numerator]) else f"mean {numerator}"


def infer_freq(label: Any) -> str | None:
    s = str(label)
    if re.fullmatch(r"\d{4}Q[1-4]", s):
        return "Q"
    if re.fullmatch(r"\d{4}-\d{2}", s):
        return "M"
    if re.fullmatch(r"\d{4}", s):
        return "Y"
    return None


def period_is_incomplete(frame: pd.DataFrame, label: Any, freq: str | None, date_col: str = "order_date") -> bool:
    """True when the data stops before the labelled period ends."""
    if date_col not in frame.columns or freq not in ("M", "Q", "Y"):
        return False
    try:
        end = pd.Period(str(label), freq=freq).end_time.normalize()
    except (ValueError, TypeError):
        return False
    last = pd.to_datetime(frame[date_col], errors="coerce").max()
    return bool(pd.notna(last) and last.normalize() < end)


# ------------------------------------------------------------ page capability
@dataclass(frozen=True)
class Access:
    enabled: bool
    reason: str = ""


def integrity_access(df: pd.DataFrame) -> Access:
    return Access(False, "The dataset is empty.") if df is None or len(df) == 0 else Access(True)


def contradiction_access(df: pd.DataFrame) -> Access:
    if df is None or len(df) == 0:
        return Access(False, "The dataset is empty.")
    if not numeric_columns(df):
        return Access(False, "No numeric metric column is available.")
    if len(group_columns(df)) < 2:
        return Access(False, "Needs two categorical columns (a period column and a separate segment column); "
                             "this dataset has fewer than two.")
    return Access(True)


def cohort_access(df: pd.DataFrame) -> Access:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return Access(False, f"Cohort analysis needs {', '.join(REQUIRED_COLUMNS)}; this dataset has no "
                             f"{', '.join(missing)}. The analysis stays off instead of inventing data.")
    return Access(True)


def experiment_access(df: pd.DataFrame) -> Access:
    if df is None or len(df) == 0:
        return Access(False, "The dataset is empty.")
    if not group_columns(df) or not numeric_columns(df):
        return Access(False, "Needs a categorical column to define groups and a numeric outcome column.")
    return Access(True)


def availability(frame: pd.DataFrame) -> dict[str, Access]:
    return {"Metric Integrity": integrity_access(frame), "Contradiction Lab": contradiction_access(frame),
            "Cohort Observatory": cohort_access(frame), "Experiment Lab": experiment_access(frame)}


# ------------------------------------------------------------ formatting
def fmt_num(v: Any, digits: int = 4) -> str:
    if v is None:
        return "undefined"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if math.isnan(f) or math.isinf(f):
        return "undefined"
    return f"{f:,.0f}" if abs(f) >= 1000 else f"{f:.{digits}g}"


def fmt_pct(v: Any, digits: int = 1) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "undefined"
    return "undefined" if math.isnan(f) or math.isinf(f) else f"{f * 100:.{digits}f}%"


def series_to_line_spec(series: pd.Series, title: str, y_title: str, x_title: str = "period") -> dict[str, Any]:
    return {"kind": "line", "title": title, "x_title": x_title, "y_title": y_title,
            "traces": [{"name": y_title, "bold": True, "x": [str(i) for i in series.index],
                        "y": [None if pd.isna(v) else float(v) for v in series]}]}


VERDICTS = {
    "simpson_reversal": ("Simpson-type reversal", "amber"),
    "partial_reversal": ("Partial reversal", "amber"),
    "mix_driven": ("Driven by a shift in mix", "amber"),
    "hidden_movement": ("Hidden movement under a flat aggregate", "amber"),
    "heterogeneous": ("Heterogeneous segments, no reversal", "blue"),
    "consistent": ("Consistent, no contradiction", "teal"),
    "insufficient_data": ("Insufficient data", "blue"),
}
