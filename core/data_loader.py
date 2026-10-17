"""Loading, type cleaning, row classification and validation for transaction data.

Design rule: nothing is silently dropped. Rows are classified and counted;
exclusion happens later, inside metric definitions, where it is documented.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

import numpy as np
import pandas as pd

from core.schema import ONLINE_RETAIL_II, DatasetSchema, Role, SchemaError

# Heuristic: genuine product codes look like 85123 or 85123A. Everything else
# (POST, M, BANK CHARGES, ...) is set aside as "non_product" and reported, so
# you can inspect exactly what was excluded and how much revenue it carried.
PRODUCT_CODE_PATTERN = r"^\d{5}[A-Za-z]{0,2}$"

ROW_CLASSES = ("sale", "return", "non_product", "invalid_price", "inconsistent")

RETAIL_TEXT_DTYPES = {
    "Invoice": "string",
    "StockCode": "string",
    "Description": "string",
    "Country": "string",
}


# --------------------------------------------------------------------- reading
def read_retail_excel(
    path: str | Path, schema: DatasetSchema = ONLINE_RETAIL_II
) -> dict[str, pd.DataFrame]:
    """Read every sheet of the Online Retail II workbook (slow: ~1M rows)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    sheets = pd.read_excel(path, sheet_name=None, dtype=RETAIL_TEXT_DTYPES)
    if not sheets or all(df.empty for df in sheets.values()):
        raise SchemaError("The workbook contains no data.")
    for df in sheets.values():
        schema.validate_against(df)
    return sheets


def load_uploaded_table(buffer: BinaryIO | str | Path, filename: str) -> pd.DataFrame:
    """Load a user-uploaded CSV or .xlsx into a DataFrame."""
    suffix = Path(filename).suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(buffer)
    elif suffix == ".xlsx":
        df = pd.read_excel(buffer)
    else:
        raise SchemaError(f"Unsupported file type '{suffix}'. Use .csv or .xlsx.")
    if df.empty:
        raise SchemaError("The uploaded file has no rows.")
    return df


# --------------------------------------------------------- sheet overlap check
@dataclass
class OverlapReport:
    found: bool
    rows_in_first: int = 0
    rows_in_second: int = 0
    identical: bool = False
    dropped: int = 0
    note: str = ""


def reconcile_sheet_overlap(
    first: pd.DataFrame, second: pd.DataFrame, date_col: str = "InvoiceDate"
) -> tuple[pd.DataFrame, OverlapReport]:
    """Combine two chronological sheets, removing the overlap only if it is an
    exact duplicate. `first` must be the earlier sheet.

    Overlap window = [earliest date in `second`, latest date in `first`].
    If the rows inside that window are identical in both sheets (compared as
    multisets via row hashes), the copy in `second` is dropped. Otherwise
    nothing is dropped and the report says so.
    """
    first, second = first.copy(), second.copy()
    first[date_col] = pd.to_datetime(first[date_col], errors="coerce")
    second[date_col] = pd.to_datetime(second[date_col], errors="coerce")

    def _concat() -> pd.DataFrame:
        return pd.concat([first, second], ignore_index=True)

    if first[date_col].notna().sum() == 0 or second[date_col].notna().sum() == 0:
        return _concat(), OverlapReport(False, note="Dates unreadable; sheets concatenated as-is.")

    start, end = second[date_col].min(), first[date_col].max()
    if start > end:
        return _concat(), OverlapReport(False, note="No date overlap between sheets.")

    in_first = first[first[date_col] >= start]
    second_mask = second[date_col] <= end
    in_second = second[second_mask]

    h1 = np.sort(pd.util.hash_pandas_object(in_first, index=False).to_numpy())
    h2 = np.sort(pd.util.hash_pandas_object(in_second, index=False).to_numpy())
    identical = len(h1) == len(h2) and bool(np.array_equal(h1, h2))

    report = OverlapReport(
        found=True,
        rows_in_first=len(in_first),
        rows_in_second=len(in_second),
        identical=identical,
    )
    if identical:
        combined = pd.concat([first, second[~second_mask]], ignore_index=True)
        report.dropped = len(in_second)
        report.note = "Overlap rows were identical in both sheets; the second copy was removed."
        return combined, report

    report.note = (
        "Sheets overlap in time but the rows differ. NOTHING was removed; "
        "investigate before trusting totals for this window."
    )
    return _concat(), report


# --------------------------------------------------------------- type cleaning
def _clean_id(series: pd.Series) -> pd.Series:
    """Whole-number IDs become nullable Int64; anything else becomes string."""
    numeric = pd.to_numeric(series, errors="coerce")
    if bool((numeric.notna() == series.notna()).all()):
        non_null = numeric.dropna()
        if bool((non_null % 1 == 0).all()):
            return numeric.astype("Int64")
    return series.astype("string")


def clean_types(
    df: pd.DataFrame, schema: DatasetSchema = ONLINE_RETAIL_II
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Coerce timestamp, quantity, price and customer ID to proper types.

    Returns the cleaned frame and a dict counting values that became missing
    *because* coercion failed (e.g. an unparseable date).
    """
    m = schema.mapping
    out = df.copy()
    coerced: dict[str, int] = {}

    def _count(col: str, before_na: int) -> None:
        coerced[col] = int(out[col].isna().sum()) - before_na

    ts, qty, price = m[Role.TIMESTAMP], m[Role.QUANTITY], m[Role.UNIT_PRICE]
    for col, fn in (
        (ts, lambda s: pd.to_datetime(s, errors="coerce")),
        (qty, lambda s: pd.to_numeric(s, errors="coerce")),
        (price, lambda s: pd.to_numeric(s, errors="coerce")),
    ):
        before = int(out[col].isna().sum())
        out[col] = fn(out[col])
        _count(col, before)

    if Role.CUSTOMER_ID in m:
        out[m[Role.CUSTOMER_ID]] = _clean_id(out[m[Role.CUSTOMER_ID]])
    return out, coerced


# -------------------------------------------------------------- row classifying
def classify_retail_rows(
    df: pd.DataFrame,
    schema: DatasetSchema = ONLINE_RETAIL_II,
    product_pattern: str = PRODUCT_CODE_PATTERN,
) -> pd.DataFrame:
    """Add flag columns and a `row_class` to every row. Nothing is removed.

    row_class, evaluated in this order:
      non_product   stock code does not look like a real product
      invalid_price price missing or <= 0
      return        invoice starts with 'C' and quantity < 0
      inconsistent  'C' invoice with qty >= 0, or normal invoice with qty <= 0/missing
      sale          everything else
    """
    m = schema.mapping
    inv = df[m[Role.INVOICE_ID]].astype("string")
    code = df[m[Role.PRODUCT_ID]].astype("string")
    qty = df[m[Role.QUANTITY]]
    price = df[m[Role.UNIT_PRICE]]

    is_cancel = inv.str.startswith("C", na=False).to_numpy(dtype=bool)
    is_product = code.str.match(product_pattern, na=False).to_numpy(dtype=bool)
    bad_price = (price.isna() | (price <= 0)).to_numpy(dtype=bool)
    neg_qty = (qty < 0).to_numpy(dtype=bool)
    bad_qty = (qty.isna() | (qty <= 0)).to_numpy(dtype=bool)

    conditions = [
        ~is_product,
        bad_price,
        is_cancel & neg_qty,
        is_cancel | bad_qty,
    ]
    choices = ["non_product", "invalid_price", "return", "inconsistent"]

    out = df.copy()
    out["is_cancellation"] = is_cancel
    out["is_product"] = is_product
    out["has_customer"] = out[m[Role.CUSTOMER_ID]].notna().to_numpy(dtype=bool)
    out["line_revenue"] = (qty * price).astype(float)
    out["row_class"] = np.select(conditions, choices, default="sale")
    return out


def non_product_summary(
    df: pd.DataFrame, schema: DatasetSchema = ONLINE_RETAIL_II, top: int = 25
) -> pd.DataFrame:
    """The stock codes set aside as non-product, with rows and revenue, for inspection."""
    m = schema.mapping
    sub = df[df["row_class"] == "non_product"]
    grouped = sub.groupby(m[Role.PRODUCT_ID]).agg(
        rows=("line_revenue", "size"),
        revenue=("line_revenue", "sum"),
        example_description=(m[Role.DESCRIPTION], "first"),
    )
    return grouped.sort_values("rows", ascending=False).head(top)


# ------------------------------------------------------------------ validation
@dataclass
class ValidationReport:
    n_rows: int
    date_min: pd.Timestamp | None
    date_max: pd.Timestamp | None
    coerced_to_missing: dict[str, int]
    missing_customer_rows: int
    exact_duplicate_rows: int
    row_class_counts: dict[str, int]
    row_class_revenue: dict[str, float]
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Rows: {self.n_rows:,}",
            f"Date range: {self.date_min} -> {self.date_max}",
            f"Values lost to failed type coercion: {self.coerced_to_missing}",
            f"Rows without customer ID: {self.missing_customer_rows:,}",
            f"Exact duplicate rows (kept, flagged): {self.exact_duplicate_rows:,}",
            "Row classes (count | revenue):",
        ]
        for c in ROW_CLASSES:
            lines.append(
                f"  {c:<14}{self.row_class_counts.get(c, 0):>10,} | "
                f"{self.row_class_revenue.get(c, 0.0):>16,.2f}"
            )
        lines += [f"WARNING: {w}" for w in self.warnings]
        return "\n".join(lines)


def validate_retail_frame(
    df: pd.DataFrame,
    schema: DatasetSchema = ONLINE_RETAIL_II,
    coerced: dict[str, int] | None = None,
) -> ValidationReport:
    """Summarise data health. Expects `clean_types` + `classify_retail_rows` output."""
    schema.validate_against(df)
    if len(df) == 0:
        raise SchemaError("Dataset has no rows.")
    for needed in ("row_class", "line_revenue", "has_customer"):
        if needed not in df.columns:
            raise SchemaError(f"Run classify_retail_rows first (missing '{needed}').")

    m = schema.mapping
    ts = df[m[Role.TIMESTAMP]]
    key_cols = list(m.values())
    n_dupes = int(df.duplicated(subset=key_cols).sum())
    counts = df["row_class"].value_counts().to_dict()
    revenue = df.groupby("row_class")["line_revenue"].sum().to_dict()
    missing_cust = int((~df["has_customer"]).sum())
    coerced = coerced or {}

    warnings: list[str] = []
    if any(v > 0 for v in coerced.values()):
        warnings.append("Some values could not be parsed and were set to missing.")
    if missing_cust / len(df) > 0.10:
        warnings.append(
            f"{missing_cust / len(df):.1%} of rows have no customer ID; "
            "customer-level metrics exclude them."
        )
    if n_dupes > 0:
        warnings.append(
            f"{n_dupes:,} exact duplicate rows detected. They are kept; these are "
            "anomalies, not confirmed errors."
        )
    if ts.notna().sum() == 0:
        warnings.append("No valid timestamps.")

    return ValidationReport(
        n_rows=len(df),
        date_min=ts.min() if ts.notna().any() else None,
        date_max=ts.max() if ts.notna().any() else None,
        coerced_to_missing=coerced,
        missing_customer_rows=missing_cust,
        exact_duplicate_rows=n_dupes,
        row_class_counts={c: int(counts.get(c, 0)) for c in ROW_CLASSES},
        row_class_revenue={c: float(revenue.get(c, 0.0)) for c in ROW_CLASSES},
        warnings=warnings,
    )


# ------------------------------------------------------------ processed cache
PROCESSED_PATH = Path("data/processed/retail_clean.csv.gz")


def load_processed(
    path: str | Path = PROCESSED_PATH, schema: DatasetSchema = ONLINE_RETAIL_II
) -> pd.DataFrame:
    """Load the cleaned, classified retail file written by data/prepare_retail.py."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m data.prepare_retail` first."
        )
    m = schema.mapping
    df = pd.read_csv(
        path,
        dtype={
            m[Role.INVOICE_ID]: "string",
            m[Role.PRODUCT_ID]: "string",
            m[Role.DESCRIPTION]: "string",
            "Country": "string",
            "row_class": "string",
        },
        parse_dates=[m[Role.TIMESTAMP]],
    )
    df[m[Role.CUSTOMER_ID]] = df[m[Role.CUSTOMER_ID]].astype("Int64")
    return df