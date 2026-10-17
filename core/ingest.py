"""Ingestion of user-uploaded tables (CSV, TSV, TXT, Excel).

Pipeline (pure pandas, no Streamlit, fully unit-tested):

    read_table      bytes -> raw DataFrame (encoding, delimiter and sheet handled)
    clean_table     tidy headers, trim text, mark empty tokens as missing, drop empty rows/columns
    prepare_dataset type inference, robust date parsing, column mapping, optional line->order
                    aggregation, period labels  ->  PreparedDataset the analysis engines can use

Principles
* Nothing is silently dropped or guessed. Every coercion, failed parse and exclusion is counted
  and reported so the interface can show it.
* Ambiguous dates (03/04/2021) are never resolved silently: the result says "ambiguous" and the
  caller must ask the user.
* The prepared frame uses fixed names for the mapped roles (order_date, customer_id, order_id,
  order_value, period) so every analysis page recognises them.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

UPLOAD_PREFIX = "upload:"
MAX_ROWS = 2_000_000
MAX_COLUMNS = 300

STD_DATE = "order_date"
STD_CUSTOMER = "customer_id"
STD_ORDER = "order_id"
STD_VALUE = "order_value"
STD_PERIOD = "period"
STD_LINES = "n_lines"
RESERVED_NAMES = {STD_DATE, STD_CUSTOMER, STD_ORDER, STD_VALUE, STD_PERIOD, STD_LINES}
PERIOD_CHOICES = ("auto", "M", "Q", "Y", "none")

NA_TOKENS = frozenset({"", "-", "--", "n.a.", "n/a", "na", "nan", "null", "none", "nil", "#n/a"})
BINARY_TRUE = frozenset({"yes", "y", "true", "t", "1"})
BINARY_FALSE = frozenset({"no", "n", "false", "f", "0"})

TEXT_SUFFIXES = {".csv", ".tsv", ".txt", ""}
EXCEL_SUFFIXES = {".xlsx", ".xlsm"}


class IngestError(ValueError):
    """Raised when a file cannot be read, understood or mapped. The message is user-facing."""


# ------------------------------------------------------------------ small helpers
def _is_str(v: Any) -> bool:
    return isinstance(v, str)


def file_digest(data: bytes) -> str:
    """Short content hash used to give each uploaded dataset a stable id."""
    return hashlib.sha1(data).hexdigest()[:16]


def _sample(series: pd.Series, n: int = 500) -> pd.Series:
    s = series.dropna()
    return s if len(s) <= n else s.sample(n, random_state=0)


def _truncate(text: Any, width: int = 40) -> str:
    t = str(text)
    return t if len(t) <= width else t[: width - 1] + "…"


# ===================================================================== reading
@dataclass
class ReadInfo:
    filename: str
    kind: str                                   # "text" | "excel"
    delimiter: str | None = None
    encoding: str | None = None
    sheet: str | None = None
    sheet_names: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def detect_delimiter(sample: str) -> str:
    """Pick the delimiter that splits the first lines most consistently."""
    lines = [ln for ln in sample.splitlines() if ln.strip()][:30]
    if not lines:
        return ","
    best, best_score = ",", 0.0
    for d in (",", ";", "\t", "|"):
        counts = [ln.count(d) for ln in lines]
        nonzero = [c for c in counts if c > 0]
        if not nonzero:
            continue
        mode = max(set(nonzero), key=nonzero.count)
        consistency = sum(1 for c in counts if c == mode) / len(counts)
        score = consistency * mode
        if consistency >= 0.6 and score > best_score:
            best, best_score = d, score
    return best


def list_sheets(data: bytes) -> list[str]:
    try:
        return list(pd.ExcelFile(io.BytesIO(data), engine="openpyxl").sheet_names)
    except Exception as exc:  # openpyxl raises several unrelated types for corrupt files
        raise IngestError(f"This does not look like a valid .xlsx file: {exc}") from exc


def read_table(
    data: bytes,
    filename: str,
    sheet: str | None = None,
    delimiter: str | None = None,
    encoding: str | None = None,
) -> tuple[pd.DataFrame, ReadInfo]:
    """Read CSV/TSV/TXT/Excel bytes into a raw table. Text files are read as strings so that
    nothing (leading zeros, IDs) is altered before the user has a say."""
    if not data or not data.strip():
        raise IngestError("The file is empty.")
    suffix = Path(filename).suffix.lower()
    if suffix == ".xls":
        raise IngestError("The old .xls format is not supported. Save the sheet as .xlsx or .csv and upload again.")
    if suffix in EXCEL_SUFFIXES:
        df, info = _read_excel(data, filename, sheet)
    elif suffix in TEXT_SUFFIXES:
        df, info = _read_text(data, filename, delimiter, encoding)
    else:
        raise IngestError(f"Unsupported file type '{suffix}'. Use .csv, .tsv, .txt, .xlsx or .xlsm.")
    if df.shape[1] == 0 or len(df) == 0:
        raise IngestError("The file has a header but no data rows.")
    if df.shape[1] > MAX_COLUMNS:
        raise IngestError(f"The file has {df.shape[1]:,} columns; the limit is {MAX_COLUMNS}.")
    if len(df) > MAX_ROWS:
        raise IngestError(f"The file has {len(df):,} rows; the limit is {MAX_ROWS:,}. Filter or sample it first.")
    return df.reset_index(drop=True), info


def _read_excel(data: bytes, filename: str, sheet: str | None) -> tuple[pd.DataFrame, ReadInfo]:
    names = list_sheets(data)
    if not names:
        raise IngestError("The workbook has no sheets.")
    if sheet is not None and sheet not in names:
        raise IngestError(f"Sheet {sheet!r} not found. Available sheets: {', '.join(names)}.")
    chosen = sheet or names[0]
    try:
        # dtype=object keeps cells as written ("007" stays text); typing is done by infer_types.
        df = pd.read_excel(io.BytesIO(data), sheet_name=chosen, engine="openpyxl", dtype=object)
    except Exception as exc:
        raise IngestError(f"Could not read sheet {chosen!r}: {exc}") from exc
    notes = [f"Read sheet '{chosen}' of {len(names)}."] if len(names) > 1 else []
    return df, ReadInfo(filename, "excel", sheet=chosen, sheet_names=names, notes=notes)


def _read_text(data: bytes, filename: str, delimiter: str | None, encoding: str | None) -> tuple[pd.DataFrame, ReadInfo]:
    if encoding in (None, "auto"):
        if data.startswith((b"\xff\xfe", b"\xfe\xff")):
            candidates = ["utf-16"]
        else:
            candidates = ["utf-8-sig", "cp1252", "latin-1"]
    else:
        candidates = [encoding]
    last_error: Exception | None = None
    for enc in candidates:
        try:
            head = data[:65536].decode(enc, errors="replace")
        except LookupError as exc:
            raise IngestError(f"Unknown encoding {enc!r}.") from exc
        if delimiter not in (None, "auto"):
            delim = delimiter
        elif Path(filename).suffix.lower() == ".tsv":
            delim = "\t"
        else:
            delim = detect_delimiter(head)
        notes: list[str] = []
        try:
            df = _csv_read(data, delim, enc, notes)
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
        if enc != "utf-8-sig" and encoding in (None, "auto"):
            notes.append(f"The file is not UTF-8; it was read as {enc}. Check accented characters.")
        return df, ReadInfo(filename, "text", delimiter=delim, encoding=enc, notes=notes)
    raise IngestError(f"Could not decode the file ({last_error}). Choose the encoding manually.")


def _csv_read(data: bytes, delim: str, enc: str, notes: list[str]) -> pd.DataFrame:
    try:
        df = pd.read_csv(io.BytesIO(data), sep=delim, dtype=str, encoding=enc)
    except pd.errors.EmptyDataError as exc:
        raise IngestError("The file has no data.") from exc
    except pd.errors.ParserError:
        try:
            df = pd.read_csv(io.BytesIO(data), sep=delim, dtype=str, encoding=enc, on_bad_lines="skip")
        except pd.errors.ParserError as exc:
            raise IngestError(f"The file could not be parsed as text separated by {delim!r}: {exc}") from exc
        notes.append("Some malformed lines (wrong number of fields) were skipped. Check the source file.")
    return df.astype(object)


# ===================================================================== cleaning
def _clean_name(raw: Any, i: int) -> str:
    name = re.sub(r"\s+", " ", str(raw).replace("\ufeff", "")).strip()
    if not name or re.fullmatch(r"Unnamed: \d+(_level_\d+)?", name):
        return f"column_{i + 1}"
    return name


def _unique_names(names: list[str]) -> list[str]:
    used: set[str] = set()
    out: list[str] = []
    for n in names:
        candidate, k = n, 1
        while candidate in used:
            k += 1
            candidate = f"{n}_{k}"
        used.add(candidate)
        out.append(candidate)
    return out


def _clean_text_column(s: pd.Series) -> pd.Series:
    """Trim whitespace, turn empty/NA tokens into missing, make stray non-text values text."""
    s = s.astype(object)
    is_str = s.map(_is_str).to_numpy(dtype=bool)
    other = s.notna().to_numpy(dtype=bool) & ~is_str
    if other.any():
        s = s.copy()
        s.loc[other] = s[other].map(str).to_numpy(dtype=object)
        is_str = is_str | other
    if is_str.any():
        stripped = s[is_str].str.strip()
        stripped = stripped.mask(stripped.str.lower().isin(NA_TOKENS))
        s = s.copy()
        s.loc[is_str] = stripped.to_numpy(dtype=object)
    return s


def clean_table(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Tidy a raw table. Returns the cleaned frame and a report dict:
    renamed, dropped_empty_columns, dropped_empty_rows, duplicate_rows."""
    if df is None or df.shape[1] == 0 or len(df) == 0:
        raise IngestError("The table has no data.")
    out = df.copy()
    old_names = [str(c) for c in out.columns]
    new_names = _unique_names([_clean_name(c, i) for i, c in enumerate(out.columns)])
    renamed = {o: n for o, n in zip(old_names, new_names) if o != n}
    out.columns = new_names
    for col in out.columns:
        dtype = out[col].dtype
        if dtype == object or isinstance(dtype, pd.StringDtype):
            out[col] = _clean_text_column(out[col])
    empty_cols = [c for c in out.columns if out[c].isna().all()]
    out = out.drop(columns=empty_cols)
    before = len(out)
    out = out.dropna(axis=0, how="all").reset_index(drop=True)
    if out.shape[1] == 0 or len(out) == 0:
        raise IngestError("The table has no usable data after removing empty rows and columns.")
    return out, {
        "renamed": renamed,
        "dropped_empty_columns": empty_cols,
        "dropped_empty_rows": int(before - len(out)),
        "duplicate_rows": int(out.duplicated().sum()),
    }


# ============================================================= numbers and flags
# Plain string patterns on purpose: pandas 3 rejects compiled patterns inside .str methods, and its
# pyarrow backend (RE2) rejects \u escapes, so the non-breaking space is written as a literal character.
_NOISE = "[\\s\u00a0$€£₹¥]"
_THOUSANDS = r"^[-+]?\d{1,3}(,\d{3})+(\.\d+)?$"
_COMMA_DECIMAL = r"^[-+]?\d{1,3}(\.\d{3})+,\d+$|^[-+]?\d+,\d{1,2}$"
_DOT_DECIMAL = r"^[-+]?\d+\.\d+$"
_LEADING_ZERO = re.compile(r"^0\d+$")
_ID_NAME = re.compile(r"(^|[\s_\-.])(id|key|sku|code|ref|uuid|number|num|no)$|^id[\s_\-.]", re.IGNORECASE)
_CAMEL_ID = re.compile(r"[a-z]Id$")


def looks_like_id_name(name: str) -> bool:
    return bool(_ID_NAME.search(name) or _CAMEL_ID.search(name))


def column_decimal(series: pd.Series, forced: bool | None = None) -> str:
    """Decimal separator for one text column: ',' only when most values are clearly comma-decimals
    (12,5 or 1.234,56) and dot-decimals are rare; otherwise '.'. `forced` overrides detection."""
    if forced is not None:
        return "," if forced else "."
    txt = _sample(series, 300)
    txt = txt[txt.map(_is_str)].astype(str)
    if len(txt) == 0:
        return "."
    txt = txt.str.replace(_NOISE, "", regex=True)
    comma = int(txt.str.match(_COMMA_DECIMAL).sum())
    dot = int(txt.str.match(_DOT_DECIMAL).sum())
    return "," if comma / len(txt) >= 0.5 and comma > 2 * dot else "."


def _to_number(series: pd.Series, decimal: str = ".") -> pd.Series:
    """Parse numbers from text: currency symbols, thousands separators, (negative) brackets,
    and either decimal point or decimal comma. Unparseable values become NaN."""
    s = series.astype(object)
    is_str = s.map(_is_str).to_numpy(dtype=bool)
    out = pd.to_numeric(s.where(~is_str), errors="coerce").astype("float64")
    if is_str.any():
        t = s[is_str].astype(str).str.replace(_NOISE, "", regex=True)
        negative = t.str.match(r"^\(.*\)$")
        t = t.str.replace(r"^\((.*)\)$", r"\1", regex=True)
        if decimal == ",":
            t = t.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
        else:
            grouped = t.str.match(_THOUSANDS)
            t = t.mask(grouped, t.str.replace(",", "", regex=False))
        parsed = pd.to_numeric(t, errors="coerce").astype("float64")
        parsed = parsed.where(~negative, -parsed.abs())
        parsed = parsed.replace([np.inf, -np.inf], np.nan)
        out.loc[is_str] = parsed.to_numpy(dtype="float64")
    return out


def _tidy_numeric(values: pd.Series) -> pd.Series:
    """Integer dtype when every value is a whole number and none is missing."""
    v = values.dropna()
    if len(v) == len(values) and len(v) and bool((v % 1 == 0).all()) and float(v.abs().max()) < 9e15:
        return values.astype("int64")
    return values.astype("float64")


def _int_text(values: pd.Series) -> pd.Series:
    """Whole numbers as text ('123'), missing stays missing."""
    out = pd.Series(np.nan, index=values.index, dtype=object)
    v = values.dropna()
    out.loc[v.index] = v.astype("int64").astype(str).to_numpy(dtype=object)
    return out


def _id_text(series: pd.Series) -> pd.Series:
    """Identifier column as clean text: '123.0' -> '123', whitespace trimmed."""
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        num = pd.to_numeric(series, errors="coerce")
        v = num.dropna()
        if len(v) and bool((v % 1 == 0).all()):
            return _int_text(num)
        return series.astype(object).map(lambda x: x if pd.isna(x) else str(x))
    s = _clean_text_column(series)
    return s.mask(s.notna(), s.astype(object).str.replace(r"^(\d+)\.0+$", r"\1", regex=True))


def binary_text_candidates(df: pd.DataFrame) -> list[str]:
    """Text columns that look like yes/no or true/false flags (candidates for 0/1 conversion)."""
    out = []
    for col in df.columns:
        s = df[col]
        if s.dtype != object:
            continue
        vals = {str(v).strip().lower() for v in _sample(s, 2000).unique() if _is_str(v)}
        if len(vals) == 2 and vals <= (BINARY_TRUE | BINARY_FALSE) and vals & BINARY_TRUE and vals & BINARY_FALSE:
            out.append(col)
    return out


def _to_binary(series: pd.Series) -> tuple[pd.Series, int]:
    low = series.map(lambda v: v.strip().lower() if _is_str(v) else v)
    out = pd.Series(np.nan, index=series.index, dtype="float64")
    out[low.isin(BINARY_TRUE)] = 1.0
    out[low.isin(BINARY_FALSE)] = 0.0
    failed = int(series.notna().sum() - out.notna().sum())
    return out, failed


# ====================================================================== dates
_YEAR_FIRST = r"^\s*(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})(.*)$"
_DAY_MONTH_YEAR = r"^\s*(\d{1,2})[-/.](\d{1,2})[-/.](\d{2,4})(.*)$"
_DIGITS = r"^\d+(\.0+)?$"


def _norm_year_first(m: re.Match) -> str:
    return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}{m.group(4)}"


@dataclass
class DateParse:
    values: pd.Series                 # datetime64, NaT where missing or unparseable
    n_total: int
    n_missing: int                    # missing before parsing
    n_failed: int                     # present but unparseable
    dayfirst: bool | None             # order used for ambiguous numeric dates (None: not needed)
    ambiguity: str                    # "none" | "ambiguous" | "inconsistent"
    failed_examples: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _to_naive(values: pd.Series) -> pd.Series:
    if isinstance(values.dtype, pd.DatetimeTZDtype):
        return values.dt.tz_convert(None)
    return values


def _parse_numeric_dates(s: pd.Series) -> tuple[pd.Series, str]:
    """Numeric columns: yyyymmdd, Excel serial days, or Unix seconds/milliseconds."""
    v = pd.to_numeric(s, errors="coerce")
    med = float(v.dropna().median()) if v.notna().any() else float("nan")
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    if np.isnan(med):
        return out, "no numeric values"
    if 19000101 <= med <= 21001231:
        txt = v.dropna().astype("int64").astype(str)
        out.loc[txt.index] = pd.to_datetime(txt, format="%Y%m%d", errors="coerce")
        return out, "read as yyyymmdd"
    if 20000 <= med <= 80000:
        ok = v.between(1, 2958465)
        out.loc[ok] = pd.to_datetime(v[ok], unit="D", origin="1899-12-30", errors="coerce")
        return out, "read as Excel serial dates"
    if 1e9 <= med < 1e10:
        out.loc[v.notna()] = pd.to_datetime(v.dropna(), unit="s", errors="coerce")
        return out, "read as Unix seconds"
    if 1e12 <= med < 1e13:
        out.loc[v.notna()] = pd.to_datetime(v.dropna(), unit="ms", errors="coerce")
        return out, "read as Unix milliseconds"
    return out, "numbers that are not recognisable as dates"


def parse_dates(series: pd.Series, dayfirst: bool | None = None) -> DateParse:
    """Parse a column of dates without guessing silently.

    * Year-first strings (2021-04-03, 2021/4/3) are unambiguous and parsed as Y-M-D.
    * Numeric day/month strings (03/04/2021) are ambiguous unless some value has a first part
      above 12 (day first) or a second part above 12 (month first). With no evidence the result
      has ambiguity == 'ambiguous' and month-first is used provisionally; the caller must ask the
      user and call again with dayfirst=True/False.
    * Month-name formats, times, AM/PM and time zones are handled by pandas.
    """
    n_total = len(series)
    notes: list[str] = []
    n_missing = int(series.isna().sum())

    if pd.api.types.is_datetime64_any_dtype(series):
        vals = _to_naive(series)
        return DateParse(vals, n_total, n_missing, 0, None, "none", [], notes)

    s = series.astype(object)
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        vals, how = _parse_numeric_dates(series)
        n_failed = int(series.notna().sum() - vals.notna().sum())
        notes.append(f"Numeric column {how}.")
        ex = [str(v) for v in series[series.notna() & vals.isna()].head(3)]
        return DateParse(vals, n_total, n_missing, n_failed, None, "none", ex, notes)

    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    is_str = s.map(_is_str).to_numpy(dtype=bool)
    other = s.notna().to_numpy(dtype=bool) & ~is_str
    if other.any():  # Timestamp / datetime / date objects (Excel, mixed columns)
        conv = pd.to_datetime(s[other], errors="coerce", utc=True)
        out.loc[other] = _to_naive(conv)

    txt = s[is_str].astype(str).str.strip()
    # Digit-only strings (20210403, 44197, 1609459200) go through the numeric path.
    digits = txt.str.match(_DIGITS)
    if len(txt) and digits.mean() >= 0.9:
        vals, how = _parse_numeric_dates(pd.to_numeric(txt, errors="coerce"))
        out.loc[txt.index] = vals
        notes.append(f"Digit-only text {how}.")
        dayfirst_used, ambiguity = None, "none"
    else:
        dayfirst_used, ambiguity = dayfirst, "none"
        year_first = txt.str.match(_YEAR_FIRST)
        if year_first.any():
            norm = txt[year_first].str.replace(_YEAR_FIRST, _norm_year_first, regex=True)
            parsed = pd.to_datetime(norm, format="ISO8601", errors="coerce", utc=True)
            out.loc[norm.index] = _to_naive(parsed)

        dmy = txt.str.match(_DAY_MONTH_YEAR)
        if dmy.any():
            parts = txt[dmy].str.extract(_DAY_MONTH_YEAR)
            first = parts[0].astype(int)
            second = parts[1].astype(int)
            day_first_evidence = bool((first > 12).any())
            month_first_evidence = bool((second > 12).any())
            if dayfirst is None:
                if day_first_evidence and not month_first_evidence:
                    dayfirst_used = True
                    notes.append("Day-first order detected (a value has a first part above 12).")
                elif month_first_evidence and not day_first_evidence:
                    dayfirst_used = False
                    notes.append("Month-first order detected (a value has a second part above 12).")
                elif day_first_evidence and month_first_evidence:
                    dayfirst_used, ambiguity = False, "inconsistent"
                    notes.append("The column mixes day-first and month-first dates; unparseable values are counted as failures.")
                else:
                    differs = bool(((first != second) & (first <= 12) & (second <= 12)).any())
                    dayfirst_used = False
                    if differs:
                        ambiguity = "ambiguous"
                        notes.append("Dates like 03/04/2021 could be 3 April or 4 March. Month-first is assumed until you choose.")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                parsed = pd.to_datetime(txt[dmy], format="mixed", dayfirst=bool(dayfirst_used), errors="coerce", utc=True)
            out.loc[txt[dmy].index] = _to_naive(parsed)

        rest = ~(year_first | dmy)
        if rest.any():  # month names, 'March 2021', etc.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                parsed = pd.to_datetime(txt[rest], format="mixed", dayfirst=False, errors="coerce", utc=True)
            out.loc[txt[rest].index] = _to_naive(parsed)

        retry = txt.index[out.loc[txt.index].isna().to_numpy()]
        retry = retry[year_first.loc[retry].to_numpy()]
        if len(retry):  # year-first strings with an unusual time part (for example AM/PM)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                parsed = pd.to_datetime(txt.loc[retry], format="mixed", dayfirst=False, errors="coerce", utc=True)
            out.loc[retry] = _to_naive(parsed)

    present = s.notna().to_numpy(dtype=bool)
    failed_mask = present & out.isna().to_numpy()
    n_failed = int(failed_mask.sum())
    examples = [_truncate(v, 30) for v in s[failed_mask].head(3)]
    return DateParse(out, n_total, n_missing, n_failed, dayfirst_used, ambiguity, examples, notes)


def choose_freq(dates: pd.Series) -> str | None:
    """Finest period grain (M, Q or Y) that gives between 2 and 60 periods, or None."""
    d = dates.dropna()
    if d.empty:
        return None
    months = d.dt.to_period("M").nunique()
    if months < 2:
        return None
    if months <= 60:
        return "M"
    return "Q" if d.dt.to_period("Q").nunique() <= 60 else "Y"


def period_labels(dates: pd.Series, freq: str) -> pd.Series:
    """Period labels as text: 2021-03 (M), 2021Q1 (Q), 2021 (Y); missing dates stay missing."""
    per = dates.dt.to_period(freq)
    return per.astype(str).where(per.notna()).astype(object)


# =============================================================== type inference
@dataclass
class TypeReport:
    numeric: list[str] = field(default_factory=list)
    coerced_to_missing: dict[str, int] = field(default_factory=dict)
    mostly_numeric_kept_text: dict[str, float] = field(default_factory=dict)
    id_as_text: list[str] = field(default_factory=list)
    leading_zero_text: list[str] = field(default_factory=list)
    binary_converted: list[str] = field(default_factory=list)
    bool_converted: list[str] = field(default_factory=list)


def infer_types(
    df: pd.DataFrame,
    skip: set[str] | frozenset[str] = frozenset(),
    binary_text: tuple[str, ...] = (),
    decimal_comma: bool | None = None,
) -> tuple[pd.DataFrame, TypeReport]:
    """Convert text columns to numbers where they clearly are numeric (>= 95% parse), flag
    identifier-like columns as text, and optionally convert yes/no columns to 0/1."""
    out = df.copy()
    rep = TypeReport()
    n = len(out)
    for col in list(out.columns):
        if col in skip:
            continue
        s = out[col]
        if pd.api.types.is_bool_dtype(s):
            out[col] = s.astype("int64")
            rep.bool_converted.append(col)
            continue
        if pd.api.types.is_datetime64_any_dtype(s):
            continue
        if pd.api.types.is_numeric_dtype(s):
            if _maybe_id(col, s, n):
                out[col] = _id_text(s)
                rep.id_as_text.append(col)
            else:
                rep.numeric.append(col)
            continue
        if col in binary_text:
            conv, failed = _to_binary(s)
            if s.notna().sum() and failed / s.notna().sum() > 0.05:
                raise IngestError(f"Column '{col}' is not a clean yes/no column ({failed:,} values are neither).")
            out[col] = _tidy_numeric(conv) if conv.notna().all() else conv
            rep.binary_converted.append(col)
            rep.numeric.append(col)
            if failed:
                rep.coerced_to_missing[col] = failed
            continue
        present = s.notna()
        if not present.any():
            continue
        if bool(s[present].map(lambda v: _is_str(v) and bool(_LEADING_ZERO.match(v))).any()):
            rep.leading_zero_text.append(col)
            continue
        parsed = _to_number(s, column_decimal(s, decimal_comma))
        ok = int(parsed.notna().sum())
        rate = ok / int(present.sum())
        if rate >= 0.95:
            vals = _tidy_numeric(parsed) if ok == n else parsed
            if _maybe_id(col, vals, n):
                out[col] = _id_text(vals)
                rep.id_as_text.append(col)
            else:
                out[col] = vals
                rep.numeric.append(col)
            failed = int(present.sum()) - ok
            if failed:
                rep.coerced_to_missing[col] = failed
        elif rate >= 0.6:
            rep.mostly_numeric_kept_text[col] = rate
    return out, rep


def _maybe_id(name: str, s: pd.Series, n: int) -> bool:
    """Whole-number column with an identifier-style name and nearly unique values."""
    if not looks_like_id_name(name):
        return False
    v = pd.to_numeric(s, errors="coerce").dropna()
    if len(v) < 20 or not bool((v % 1 == 0).all()):
        return False
    return v.nunique() / len(v) >= 0.9


# ============================================================ mapping and result
@dataclass(frozen=True)
class Mapping:
    """Which column plays which role. Every role is optional; analyses that need a missing
    role simply stay disabled with an explanation."""

    date: str | None = None
    customer: str | None = None
    order_id: str | None = None
    value: str | None = None
    dayfirst: bool | None = None            # None = detect; the caller must ask when ambiguous
    period: str = "auto"                    # auto | M | Q | Y | none
    aggregate_lines: bool = False           # combine line items into one row per order_id
    binary_text: tuple[str, ...] = ()
    drop: tuple[str, ...] = ()
    decimal_comma: bool | None = None

    def roles(self) -> dict[str, str]:
        return {r: c for r, c in (("date", self.date), ("customer", self.customer),
                                  ("order_id", self.order_id), ("value", self.value)) if c}


@dataclass
class IngestReport:
    rows_in: int = 0
    rows_out: int = 0
    columns_out: int = 0
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    date: dict[str, Any] | None = None
    date_ambiguous: bool = False
    customer: dict[str, Any] | None = None
    value: dict[str, Any] | None = None
    aggregation: dict[str, Any] | None = None
    coerced_to_missing: dict[str, int] = field(default_factory=dict)
    duplicate_rows: int = 0


@dataclass
class PreparedDataset:
    key: str
    name: str
    frame: pd.DataFrame
    report: IngestReport
    mapping: Mapping
    source: str                      # "upload" or "synthetic"
    freq: str | None                 # period grain used for the `period` column


def _mapping_key(source_id: str, name: str, mapping: Mapping) -> str:
    payload = json.dumps({"s": source_id, "n": name, "m": asdict(mapping)}, sort_keys=True, default=list)
    return UPLOAD_PREFIX + hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def _validate_mapping(df: pd.DataFrame, mapping: Mapping) -> None:
    if mapping.period not in PERIOD_CHOICES:
        raise IngestError(f"period must be one of {PERIOD_CHOICES}.")
    roles = mapping.roles()
    missing = [c for c in roles.values() if c not in df.columns]
    if missing:
        raise IngestError(f"Mapped column(s) not found in the data: {missing}")
    used = list(roles.values())
    dupes = sorted({c for c in used if used.count(c) > 1})
    if dupes:
        raise IngestError(f"The same column is mapped to more than one role: {dupes}")
    protected = [c for c in mapping.drop if c in used]
    if protected:
        raise IngestError(f"Column(s) {protected} are mapped to a role and cannot also be dropped.")
    if mapping.aggregate_lines and not mapping.order_id:
        raise IngestError("Combining line items needs an order ID column. Map one, or untick the option.")


def _is_simulated_flag(frame: pd.DataFrame) -> bool:
    for col in frame.columns:
        if str(col).lower() == "is_simulated":
            s = frame[col].dropna()
            if s.empty:
                return False
            if pd.api.types.is_numeric_dtype(s):
                return bool((s == 1).all())
            return bool(s.astype(str).str.lower().isin({"true", "1", "yes", "y"}).all())
    return False


def _aggregate_lines(frame: pd.DataFrame, rep: IngestReport) -> pd.DataFrame:
    """One row per order_id: date = earliest, value = sum, customer = first, n_lines = count.
    Other columns are kept only when they are constant within >= 95% of orders."""
    ids = frame[STD_ORDER]
    has_id = ids.notna()
    if not has_id.all():
        rep.warnings.append(f"{int((~has_id).sum()):,} rows have no order ID and were left out of the order table.")
    work = frame.loc[has_id]
    g = work.groupby(STD_ORDER, sort=False)
    agg: dict[str, pd.Series] = {}
    if STD_DATE in work.columns:
        agg[STD_DATE] = g[STD_DATE].min()
    if STD_CUSTOMER in work.columns:
        agg[STD_CUSTOMER] = g[STD_CUSTOMER].first()
        conflicts = int((g[STD_CUSTOMER].nunique() > 1).sum())
        if conflicts:
            rep.warnings.append(f"{conflicts:,} orders list more than one customer; the first is used.")
    if STD_VALUE in work.columns:
        agg[STD_VALUE] = g[STD_VALUE].sum(min_count=1)
    agg[STD_LINES] = g.size()
    kept, dropped = [], []
    for col in work.columns:
        if col in (STD_ORDER, STD_DATE, STD_CUSTOMER, STD_VALUE):
            continue
        constant = (g[col].nunique(dropna=True) <= 1).mean()
        if constant >= 0.95:
            agg[col] = g[col].first()
            kept.append(col)
        else:
            dropped.append(col)
    out = pd.DataFrame(agg)
    out.index.name = STD_ORDER
    out = out.reset_index()
    rep.aggregation = {"lines": int(has_id.sum()), "orders": int(len(out)),
                       "kept_order_level_columns": kept, "dropped_line_level_columns": dropped}
    rep.notes.append(f"Combined {int(has_id.sum()):,} line items into {len(out):,} orders "
                     f"(value summed, earliest date, first customer).")
    if dropped:
        rep.notes.append("Dropped line-level columns that vary within an order: " + ", ".join(dropped)
                         + ". Map one as the value column or pre-aggregate it if you need it.")
    return out


def prepare_dataset(
    df: pd.DataFrame,
    mapping: Mapping,
    source_id: str,
    name: str,
    read_notes: list[str] | None = None,
    clean_info: dict[str, Any] | None = None,
) -> PreparedDataset:
    """Turn a cleaned table plus a column mapping into a dataset the analysis pages can use."""
    if df is None or len(df) == 0 or df.shape[1] == 0:
        raise IngestError("The table is empty.")
    _validate_mapping(df, mapping)
    rep = IngestReport(rows_in=len(df))
    rep.notes.extend(read_notes or [])
    if clean_info:
        if clean_info.get("renamed"):
            rep.notes.append(f"Renamed {len(clean_info['renamed'])} column header(s) (blank, duplicate or 'Unnamed').")
        if clean_info.get("dropped_empty_columns"):
            rep.notes.append("Removed empty columns: " + ", ".join(clean_info["dropped_empty_columns"]) + ".")
        if clean_info.get("dropped_empty_rows"):
            rep.notes.append(f"Removed {clean_info['dropped_empty_rows']:,} completely empty rows.")
        rep.duplicate_rows = int(clean_info.get("duplicate_rows", 0))

    work = df.drop(columns=[c for c in mapping.drop if c in df.columns]).copy()
    roles = mapping.roles()
    skip = set(roles.values())

    # 1. generic type inference for unmapped columns
    work, trep = infer_types(work, skip=skip, binary_text=mapping.binary_text, decimal_comma=mapping.decimal_comma)
    rep.coerced_to_missing.update(trep.coerced_to_missing)
    if trep.id_as_text:
        rep.notes.append("Treated as identifiers (text, not metrics): " + ", ".join(trep.id_as_text) + ".")
    if trep.leading_zero_text:
        rep.notes.append("Kept as text because values start with zero: " + ", ".join(trep.leading_zero_text) + ".")
    if trep.binary_converted:
        rep.notes.append("Converted yes/no columns to 0/1: " + ", ".join(trep.binary_converted) + ".")
    if trep.bool_converted:
        rep.notes.append("Converted true/false columns to 0/1: " + ", ".join(trep.bool_converted) + ".")
    for col, rate in trep.mostly_numeric_kept_text.items():
        rep.warnings.append(f"Column '{col}' is {rate:.0%} numeric but has other values; kept as text.")
    for col, n_bad in trep.coerced_to_missing.items():
        rep.warnings.append(f"{n_bad:,} non-numeric value(s) in '{col}' were set to missing.")

    # 2. mapped columns
    freq: str | None = None
    if mapping.date:
        dp = parse_dates(work[mapping.date], mapping.dayfirst)
        n_present = dp.n_total - dp.n_missing
        if n_present == 0:
            raise IngestError(f"The date column '{mapping.date}' is empty.")
        if dp.n_failed / n_present > 0.5:
            raise IngestError(
                f"Column '{mapping.date}' does not look like dates: {dp.n_failed:,} of {n_present:,} values could not be read "
                f"(for example {', '.join(map(repr, dp.failed_examples))}).")
        work[mapping.date] = dp.values
        rep.notes.extend(dp.notes)
        rep.date_ambiguous = dp.ambiguity == "ambiguous" and mapping.dayfirst is None
        if dp.ambiguity == "inconsistent":
            rep.warnings.append("The date column mixes day-first and month-first formats; check the failures.")
        if dp.n_failed:
            rep.warnings.append(f"{dp.n_failed:,} date value(s) could not be read (for example "
                                f"{', '.join(map(repr, dp.failed_examples))}); those rows have no period.")
        d = dp.values.dropna()
        if len(d) and (d.min().year < 1970 or d.max().year > 2100):
            rep.warnings.append(f"Dates span {d.min().date()} to {d.max().date()}; check the date format.")
        freq = choose_freq(dp.values) if mapping.period == "auto" else (None if mapping.period == "none" else mapping.period)
        if mapping.period == "auto" and freq is None:
            rep.warnings.append("All dates fall in one month, so no period column was created. Use another column "
                                "as the period, or choose a period grain explicitly.")
        rep.date = {"min": str(d.min().date()) if len(d) else None, "max": str(d.max().date()) if len(d) else None,
                    "n_failed": dp.n_failed, "n_missing": dp.n_missing, "dayfirst": dp.dayfirst,
                    "ambiguity": dp.ambiguity, "freq": freq}
    if mapping.value:
        raw_vals = work[mapping.value]
        vals = _to_number(raw_vals, column_decimal(raw_vals, mapping.decimal_comma))
        present = int(raw_vals.notna().sum())
        ok = int(vals.notna().sum())
        if present == 0 or ok / present < 0.5:
            raise IngestError(f"Column '{mapping.value}' is not numeric: only {ok:,} of {present:,} values could be read as numbers.")
        work[mapping.value] = vals
        if present - ok:
            rep.warnings.append(f"{present - ok:,} non-numeric value(s) in the value column '{mapping.value}' were set to missing.")
            rep.coerced_to_missing[mapping.value] = present - ok
        v = vals.dropna()
        rep.value = {"total": float(v.sum()), "n_negative": int((v < 0).sum()), "n_zero": int((v == 0).sum())}
        if rep.value["n_negative"]:
            rep.notes.append(f"{rep.value['n_negative']:,} negative value(s) in '{mapping.value}' (refunds?). They are kept.")
    for role in ("customer", "order_id"):
        col = roles.get(role)
        if col:
            work[col] = _id_text(work[col])
    if mapping.customer:
        s = work[mapping.customer]
        rep.customer = {"n_distinct": int(s.nunique()), "n_missing": int(s.isna().sum())}
        if rep.customer["n_missing"]:
            rep.notes.append(f"{rep.customer['n_missing']:,} rows have no customer ID; customer analyses exclude them.")

    # 3. standard names (a clashing unmapped column is kept as <name>_original)
    std = {"date": STD_DATE, "customer": STD_CUSTOMER, "order_id": STD_ORDER, "value": STD_VALUE}
    rename = {col: std[role] for role, col in roles.items()}
    taken = set(rename.values()) | ({STD_PERIOD} if mapping.date else set()) | ({STD_LINES} if mapping.aggregate_lines else set())
    for col in work.columns:
        if col in rename or col not in taken:
            continue
        new, k = f"{col}_original", 1
        while new in work.columns or new in rename.values():
            k += 1
            new = f"{col}_original{k}"
        rename[col] = new
        rep.notes.append(f"Column '{col}' clashes with a reserved name; it is kept as '{new}'.")
    work = work.rename(columns=rename)

    # 4. grain
    if STD_ORDER in work.columns:
        dupes = int(work[STD_ORDER].dropna().duplicated(keep=False).sum())
        if dupes and not mapping.aggregate_lines:
            rep.warnings.append(f"{dupes:,} rows share an order ID with another row. Cohort and concentration results "
                                "assume one row per order; tick 'Combine line items into orders' if these are line items.")
    if mapping.aggregate_lines:
        work = _aggregate_lines(work, rep)

    # 5. period labels
    if STD_DATE in work.columns and freq:
        work[STD_PERIOD] = period_labels(work[STD_DATE], freq)
        rep.date["n_periods"] = int(work[STD_PERIOD].nunique())

    # 6. tidy column order: standard roles first
    front = [c for c in (STD_ORDER, STD_DATE, STD_PERIOD, STD_CUSTOMER, STD_VALUE, STD_LINES) if c in work.columns]
    work = work[front + [c for c in work.columns if c not in front]].reset_index(drop=True)

    source = "synthetic" if _is_simulated_flag(work) else "upload"
    if source == "synthetic":
        rep.notes.append("The file has an is_simulated column set to true: results are labelled SIMULATED.")
    rep.rows_out, rep.columns_out = len(work), work.shape[1]
    return PreparedDataset(_mapping_key(source_id, name, mapping), name, work, rep, mapping, source, freq)


# ===================================================================== helpers for the UI
def profile_columns(df: pd.DataFrame) -> pd.DataFrame:
    """One row per column: type, share missing, distinct values, example."""
    rows = []
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_datetime64_any_dtype(s):
            kind = "date"
        elif pd.api.types.is_numeric_dtype(s):
            vals = set(pd.to_numeric(s, errors="coerce").dropna().unique().tolist())
            kind = "numeric (0/1)" if vals and vals <= {0, 1} else "numeric"
        else:
            kind = "text"
        first = s.dropna()
        rows.append({"Column": col, "Type": kind, "Missing": float(s.isna().mean()),
                     "Distinct": int(s.nunique(dropna=True)),
                     "Example": _truncate(first.iloc[0]) if len(first) else ""})
    return pd.DataFrame(rows)


_DATE_HINTS = ("date", "time", "day", "month", "created", "ordered", "timestamp", "purchase", "placed", "invoice date")
_CUSTOMER_HINTS = ("customer", "client", "buyer", "user", "account", "member", "cust", "shopper", "subscriber", "visitor")
_ORDER_HINTS = ("order", "invoice", "transaction", "receipt", "session", "ticket", "basket", "txn")
_VALUE_HINTS = ("revenue", "sales", "amount", "total", "spend", "gmv", "order_value", "value", "payment",
                "income", "turnover", "price", "cost")


def _hint(name: str, words: tuple[str, ...]) -> int:
    low = name.lower()
    for rank, w in enumerate(words):
        if w in low:
            return rank
    return -1


def suggest_mapping(df: pd.DataFrame) -> tuple[Mapping, dict[str, str]]:
    """Best-guess roles with a plain-language reason for each. Always a suggestion, never final."""
    reasons: dict[str, str] = {}
    chosen: set[str] = set()

    # date
    best, best_score = None, 0
    for col in df.columns:
        s = df[col]
        score = 0
        if pd.api.types.is_datetime64_any_dtype(s):
            score = 3
        elif _hint(col, _DATE_HINTS) >= 0 or s.dtype == object:
            samp = _sample(s, 300)
            if len(samp) >= 3:
                numeric_share = float(_to_number(samp).notna().mean())
                hinted = _hint(col, _DATE_HINTS) >= 0
                if hinted or numeric_share < 0.5:
                    p = parse_dates(samp.reset_index(drop=True))
                    ok = 1 - p.n_failed / max(len(samp), 1)
                    if ok >= 0.9:
                        score = 2 if hinted else 1
        if score > best_score:
            best, best_score = col, score
    if best:
        chosen.add(best)
        reasons["date"] = "values parse as dates" + (" and the name mentions a date" if _hint(best, _DATE_HINTS) >= 0 else "")

    def pick(hints, role, need_numeric=False, why=""):
        cands = []
        for col in df.columns:
            if col in chosen:
                continue
            rank = _hint(col, hints)
            if rank < 0:
                continue
            s = df[col]
            if need_numeric:
                samp = _sample(s, 300)
                if len(samp) == 0 or _to_number(samp, column_decimal(samp)).notna().mean() < 0.95:
                    continue
            elif s.nunique(dropna=True) < 2:
                continue
            id_bonus = 0 if looks_like_id_name(col) else 1
            cands.append((id_bonus if not need_numeric else 0, rank, list(df.columns).index(col), col))
        if cands:
            col = sorted(cands)[0][3]
            chosen.add(col)
            reasons[role] = why
            return col
        return None

    customer = pick(_CUSTOMER_HINTS, "customer", why="the name mentions a customer or user")
    order_id = pick(_ORDER_HINTS, "order_id", why="the name mentions an order, invoice or transaction")
    value = pick(_VALUE_HINTS, "value", need_numeric=True, why="a numeric column whose name suggests money (check it is the right one)")
    return Mapping(date=best, customer=customer, order_id=order_id, value=value), reasons


def sample_text_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if df[c].dtype == object]
