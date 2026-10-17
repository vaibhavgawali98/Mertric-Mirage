"""Upload Data page: bring your own CSV or Excel file.

Flow: choose a file -> it is read and cleaned (with a report) -> you confirm which column plays
which role -> the dataset is prepared and previewed (including which analyses it will support) ->
"Use this dataset" switches the whole app to it.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from core import ingest
from core.ingest import IngestError, Mapping
from data.sample_upload import SAMPLE_NAME, sample_csv_bytes
from ui import logic, state, theme
from ui.logic import fmt_num, fmt_pct

NONE = "(none)"
READ_KEY = "_mm_read"
PREVIEW_KEY = "_mm_preview"
SAMPLE_FLAG = "_mm_use_sample"
DELIMITERS = {"Detect automatically": "auto", "Comma ,": ",", "Semicolon ;": ";", "Tab": "\t", "Pipe |": "|"}
ENCODINGS = ["auto", "utf-8-sig", "cp1252", "latin-1"]
PERIODS = {"Automatic": "auto", "Month": "M", "Quarter": "Q", "Year": "Y", "No period column": "none"}
DATE_ORDERS = {"Detect automatically": None, "Day first (03/04/2021 = 3 April)": True,
               "Month first (03/04/2021 = 4 March)": False}


def render(ctx: logic.AppContext | None = None) -> None:
    theme.page_header("Upload Data", "Bring your own CSV or Excel file. Metric Mirage checks it, maps it, and switches on "
                                     "every analysis it can support")
    theme.callout("info", "Your file is read on the machine running this app and kept only in this browser session. "
                          "Findings you save are stored; the file itself is not.", title="Privacy")
    _manager()

    data, filename = _source()
    if data is None:
        _empty_state()
        return
    if st.session_state.get(SAMPLE_FLAG):
        st.caption(f"Using the built-in sample file ({SAMPLE_NAME}). It is simulated.")
        if st.button("Stop using the sample", key="mm_stop_sample"):
            st.session_state.pop(SAMPLE_FLAG, None)
            st.session_state.pop(PREVIEW_KEY, None)
            st.rerun()

    entry = _read_with_options(data, filename)
    if entry is None:
        return
    _show_file_summary(entry)
    _mapping_form(entry, filename)
    _show_preview(entry)


# ------------------------------------------------------------------ source
def _source() -> tuple[bytes | None, str | None]:
    uploaded = st.file_uploader("Choose a CSV, TSV, TXT or Excel (.xlsx) file",
                                type=["csv", "tsv", "txt", "xlsx", "xlsm"], key="mm_file")
    if uploaded is not None:
        st.session_state.pop(SAMPLE_FLAG, None)
        return uploaded.getvalue(), uploaded.name
    if st.session_state.get(SAMPLE_FLAG):
        return sample_csv_bytes(), SAMPLE_NAME
    return None, None


def _empty_state() -> None:
    st.markdown("""
**What works best**

- One row per order, session or event, with an ID, a date, and a value. Line-item exports work too: tick the
  option to combine lines into orders.
- Any columns you like. Text columns with a few categories become segments; numeric columns become metrics.
- A date column enables period comparisons, change-point checks and cohort analysis.
""")
    c1, c2 = st.columns(2)
    if c1.button("Try the sample file (simulated)", key="mm_use_sample_btn"):
        st.session_state[SAMPLE_FLAG] = True
        st.session_state.pop(PREVIEW_KEY, None)
        st.rerun()
    c2.download_button("Download the sample CSV", sample_csv_bytes(), file_name=SAMPLE_NAME, mime="text/csv",
                       key="mm_sample_dl")
    with st.expander("Limits and supported formats"):
        st.markdown(f"""
- Formats: .csv, .tsv, .txt (comma, semicolon, tab or pipe separated) and .xlsx. Old .xls files are not supported.
- Up to {ingest.MAX_ROWS:,} rows and {ingest.MAX_COLUMNS} columns; the uploader accepts files up to 200 MB.
- Encodings are detected (UTF-8, Windows-1252, Latin-1, UTF-16). Dates in most common formats are understood;
  ambiguous ones such as 03/04/2021 are never guessed, you will be asked.
""")


def _manager() -> None:
    infos = state.upload_infos()
    if not infos:
        return
    with st.expander(f"Uploaded datasets in this session ({len(infos)})", expanded=False):
        for key, info in infos.items():
            prepared = state.get_upload(key)
            c1, c2, c3 = st.columns([6, 2, 2])
            c1.write(info.label)
            c2.caption(f"{len(prepared.frame):,} rows" if prepared else "")
            if c3.button("Switch to", key=f"mm_sw_{key}"):
                st.session_state["_pending_dataset"] = key
                st.session_state["_pending_page"] = "Command Center"
                st.rerun()
            if c3.button("Remove", key=f"mm_rm_{key}"):
                if st.session_state.get("dataset_key") == key:
                    st.session_state["_pending_dataset"] = logic.default_dataset_key()
                state.remove_upload(key)
                st.rerun()
        st.caption(f"Up to {state.MAX_UPLOADS} uploads are kept per session; the oldest is dropped beyond that.")


# ------------------------------------------------------------------ reading
def _read_with_options(data: bytes, filename: str) -> dict | None:
    suffix = Path(filename).suffix.lower()
    sheet, delimiter, encoding = None, "auto", "auto"
    with st.expander("Reading options", expanded=False):
        if suffix in ingest.EXCEL_SUFFIXES:
            try:
                sheets = ingest.list_sheets(data)
            except IngestError as exc:
                theme.callout("error", str(exc), title="Could not open this workbook")
                return None
            sheet = st.selectbox("Sheet", sheets, key="mm_sheet")
        else:
            label = st.selectbox("Delimiter", list(DELIMITERS), key="mm_delim")
            delimiter = DELIMITERS[label]
            encoding = st.selectbox("Encoding", ENCODINGS, key="mm_enc")
    src_id = f"{ingest.file_digest(data)}|{sheet}|{delimiter}|{encoding}"
    cached = st.session_state.get(READ_KEY)
    if cached and cached["src_id"] == src_id:
        return cached
    try:
        with st.spinner("Reading and checking your file (large files can take a few seconds)..."):
            raw, rinfo = ingest.read_table(data, filename, sheet=sheet, delimiter=delimiter, encoding=encoding)
            clean, cinfo = ingest.clean_table(raw)
            suggestion = ingest.suggest_mapping(clean)
            binary = ingest.binary_text_candidates(clean)
    except IngestError as exc:
        st.session_state.pop(READ_KEY, None)
        theme.callout("error", str(exc), title="Could not read this file")
        return None
    except Exception as exc:  # unexpected: show it instead of crashing the page
        st.session_state.pop(READ_KEY, None)
        theme.callout("error", f"Unexpected problem while reading the file: {exc}", title="Could not read this file")
        return None
    entry = {"src_id": src_id, "clean": clean, "rinfo": rinfo, "cinfo": cinfo, "suggestion": suggestion,
             "binary": binary, "digest": ingest.file_digest(data)}
    st.session_state[READ_KEY] = entry
    st.session_state.pop(PREVIEW_KEY, None)
    return entry


def _show_file_summary(entry: dict) -> None:
    clean, rinfo, cinfo = entry["clean"], entry["rinfo"], entry["cinfo"]
    theme.kpi_row([("Rows", f"{len(clean):,}", "", "teal"), ("Columns", str(clean.shape[1]), "", "blue"),
                   ("Missing cells", fmt_pct(clean.isna().to_numpy().mean()), "", "amber"),
                   ("Exact duplicate rows", f"{cinfo['duplicate_rows']:,}", "kept, not removed", "amber")])
    facts = list(rinfo.notes)
    if rinfo.kind == "text":
        facts.append(f"Read as text separated by {_delimiter_name(rinfo.delimiter)}, encoding {rinfo.encoding}.")
    if cinfo["renamed"]:
        facts.append(f"{len(cinfo['renamed'])} column header(s) were blank, duplicated or 'Unnamed' and were renamed.")
    if cinfo["dropped_empty_columns"]:
        facts.append("Empty columns removed: " + ", ".join(cinfo["dropped_empty_columns"]) + ".")
    if cinfo["dropped_empty_rows"]:
        facts.append(f"{cinfo['dropped_empty_rows']:,} completely empty rows removed.")
    for f in facts:
        st.caption(f)
    with st.expander("First rows of your file", expanded=True):
        st.dataframe(clean.head(15), hide_index=True)


def _delimiter_name(d: str | None) -> str:
    return {",": "commas", ";": "semicolons", "\t": "tabs", "|": "pipes"}.get(d or ",", repr(d))


# ------------------------------------------------------------------ mapping
def _mapping_form(entry: dict, filename: str) -> None:
    clean: pd.DataFrame = entry["clean"]
    sug, why = entry["suggestion"]
    cols = list(clean.columns)
    options = [NONE] + cols
    sid = entry["src_id"]

    def index_of(col: str | None) -> int:
        return options.index(col) if col in options else 0

    st.markdown("### Tell Metric Mirage what your columns mean")
    st.caption("Every choice is optional. Roles you skip simply keep the analyses that need them switched off.")
    with st.form(f"mm_form_{sid}"):
        c1, c2 = st.columns(2)
        date_col = c1.selectbox("Date or timestamp", options, index=index_of(sug.date), key=f"mm_date_{sid}",
                                help="Enables periods, change points and cohorts.")
        customer = c2.selectbox("Customer or user ID", options, index=index_of(sug.customer), key=f"mm_cust_{sid}",
                                help="Enables cohorts, repeat purchasing and revenue concentration.")
        order_id = c1.selectbox("Order or transaction ID", options, index=index_of(sug.order_id), key=f"mm_order_{sid}")
        value = c2.selectbox("Value (revenue, amount, spend)", options, index=index_of(sug.value), key=f"mm_value_{sid}",
                             help="A numeric column. Enables concentration and order-value analyses.")
        c3, c4 = st.columns(2)
        date_order = c3.radio("Date order, for dates like 03/04/2021", list(DATE_ORDERS), key=f"mm_dord_{sid}")
        period = c4.selectbox("Period grain", list(PERIODS), key=f"mm_period_{sid}")
        aggregate = st.checkbox("Each row is a line item: combine lines into one row per order ID", key=f"mm_agg_{sid}",
                                help="Needs an order ID. Value is summed, the earliest date and the first customer are used.")
        binary = st.multiselect("Convert these yes/no columns to 0/1 (so they can be outcomes)", entry["binary"],
                                key=f"mm_bin_{sid}", help="Leave a yes/no column out if you want to use it as a segment instead.")
        drop = st.multiselect("Leave these columns out", cols, key=f"mm_drop_{sid}")
        submitted = st.form_submit_button("Check and prepare data", type="primary")
    if why:
        st.caption("Suggestions: " + "; ".join(f"{r.replace('_', ' ')} ({t})" for r, t in why.items()) + ".")
    if not submitted:
        return
    mapping = Mapping(
        date=None if date_col == NONE else date_col, customer=None if customer == NONE else customer,
        order_id=None if order_id == NONE else order_id, value=None if value == NONE else value,
        dayfirst=DATE_ORDERS[date_order], period=PERIODS[period], aggregate_lines=bool(aggregate),
        binary_text=tuple(binary), drop=tuple(drop),
    )
    try:
        with st.spinner("Preparing your dataset..."):
            prepared = ingest.prepare_dataset(clean, mapping, sid, filename, read_notes=entry["rinfo"].notes,
                                              clean_info=entry["cinfo"])
        st.session_state[PREVIEW_KEY] = {"src_id": sid, "prepared": prepared, "error": None}
    except IngestError as exc:
        st.session_state[PREVIEW_KEY] = {"src_id": sid, "prepared": None, "error": str(exc)}
    except Exception as exc:
        st.session_state[PREVIEW_KEY] = {"src_id": sid, "prepared": None,
                                         "error": f"Unexpected problem while preparing the data: {exc}"}


# ------------------------------------------------------------------ preview
def _show_preview(entry: dict) -> None:
    prev = st.session_state.get(PREVIEW_KEY)
    if not prev or prev["src_id"] != entry["src_id"]:
        return
    if prev["error"]:
        theme.callout("error", prev["error"], title="This mapping did not work")
        return
    p = prev["prepared"]
    rep = p.report
    st.markdown("### Prepared dataset")
    items = [("Rows", f"{rep.rows_out:,}", f"from {rep.rows_in:,} read", "teal"),
             ("Columns", str(rep.columns_out), "", "blue")]
    if rep.date:
        items.append(("Date range", f"{rep.date['min']} to {rep.date['max']}",
                      f"{rep.date.get('n_periods', 0)} periods" if rep.date.get("freq") else "no period column", "teal"))
    if rep.customer:
        items.append(("Customers", f"{rep.customer['n_distinct']:,}", f"{rep.customer['n_missing']:,} rows without an ID", "blue"))
    if rep.value:
        items.append(("Total value", fmt_num(rep.value["total"]), "sum of the value column", "amber"))
    theme.kpi_row(items)

    if rep.date_ambiguous:
        theme.callout("error", "Dates like 03/04/2021 could be 3 April or 4 March, and nothing in the file settles it. "
                               "Choose 'Day first' or 'Month first' in the form above and press 'Check and prepare data' "
                               "again. Metric Mirage will not guess.", title="Date order needed")
        return
    for w in rep.warnings:
        theme.callout("warn", w)
    if rep.notes:
        with st.expander(f"What was done to your file ({len(rep.notes)} step(s))"):
            st.markdown("\n".join(f"- {n}" for n in rep.notes))

    st.markdown("#### What this dataset will support")
    access = logic.availability(p.frame)
    st.dataframe(pd.DataFrame([{"Page": page, "Status": "Available" if a.enabled else "Disabled", "Why": a.reason}
                               for page, a in access.items()]), hide_index=True)
    with st.expander("Columns after preparation"):
        prof = ingest.profile_columns(p.frame)
        prof["Missing"] = prof["Missing"].map(fmt_pct)
        st.dataframe(prof, hide_index=True)
    with st.expander("First rows after preparation"):
        st.dataframe(p.frame.head(15), hide_index=True)
    if p.source == "synthetic":
        theme.callout("warn", "This file is marked as simulated (is_simulated = true). Findings from it will be labelled SIMULATED.")

    if st.button("Use this dataset for analysis", type="primary", key=f"mm_use_{p.key}"):
        info = state.register_upload(p)
        st.session_state["_pending_dataset"] = info.key
        st.session_state["_pending_page"] = "Command Center"
        st.session_state.pop(PREVIEW_KEY, None)
        st.rerun()
