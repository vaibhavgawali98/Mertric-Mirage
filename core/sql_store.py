"""SQLite layer for the retail transaction data.

Grain (stated explicitly):
  transactions  one row per INVOICE LINE (a product on an invoice)
  orders        VIEW, one row per SALE INVOICE; order_date is the earliest line timestamp

Queries live in sql/*.sql so they can be read, shown in the UI and tested.
Free-form SQL only runs on a read-only connection whose authorizer permits SELECT
statements alone, with a row cap and a time limit.
"""
from __future__ import annotations

import math
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.cohorts import cohort_retention, repeat_purchase_gaps, revenue_concentration
from core.contradiction import MetricSpec, investigate_contradiction
from core.metrics import metric_summary, order_level_frame
from core.schema import ONLINE_RETAIL_II, DatasetSchema, Role

SQL_DIR = Path(__file__).resolve().parent.parent / "sql"
DEFAULT_DB_PATH = Path("data/processed/retail.db")

NAMED_QUERIES = (
    "headline_metrics",
    "monthly_orders",
    "cohort_retention",
    "repeat_gaps",
    "revenue_concentration",
    "segment_period_aov",
)
_SETUP_FILES = ("schema", "indexes")

# SQLite C-API values: authorizer action codes and return codes.
_ALLOWED_ACTIONS = {20, 21, 31, 33}   # READ, SELECT, FUNCTION, RECURSIVE
_SQLITE_OK, _SQLITE_DENY = 0, 1


class SqlStoreError(RuntimeError):
    """Raised for any problem in the SQL layer."""


class SqlSafetyError(SqlStoreError):
    """Raised when a query is refused before it reaches the database."""


# ------------------------------------------------------------------ SQL files
def load_sql(name: str) -> str:
    """Read a whitelisted SQL file (also used by the UI's query-inspection panel)."""
    if name not in NAMED_QUERIES and name not in _SETUP_FILES:
        raise SqlStoreError(f"Unknown SQL file {name!r}.")
    path = SQL_DIR / f"{name}.sql"
    if not path.exists():
        raise SqlStoreError(f"SQL file not found: {path}")
    return path.read_text(encoding="utf-8")


# ------------------------------------------------------------------- building
_INSERT = (
    "INSERT INTO transactions (invoice, stock_code, description, quantity, invoice_date, "
    "unit_price, customer_id, country, row_class, line_revenue) "
    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)


def _col(series: pd.Series, cast=None) -> list:
    """Column as plain Python values with missing values as None (sqlite3 binds these)."""
    out = []
    for v in series.astype(object):
        if v is None or (not isinstance(v, str) and pd.isna(v)):
            out.append(None)
        else:
            out.append(cast(v) if cast else v)
    return out


def build_database(
    df: pd.DataFrame,
    path: str | Path = DEFAULT_DB_PATH,
    schema: DatasetSchema = ONLINE_RETAIL_II,
    overwrite: bool = True,
    chunk_rows: int = 100_000,
) -> dict[str, Any]:
    """Write a classified retail frame (core.data_loader.classify_retail_rows) to SQLite."""
    if df is None or len(df) == 0:
        raise SqlStoreError("The dataset is empty.")
    m = schema.mapping
    inv, code, desc = m[Role.INVOICE_ID], m[Role.PRODUCT_ID], m[Role.DESCRIPTION]
    qty, ts_col, price = m[Role.QUANTITY], m[Role.TIMESTAMP], m[Role.UNIT_PRICE]
    cust = m[Role.CUSTOMER_ID]
    if not schema.segment_columns:
        raise SqlStoreError("The schema needs a country (segment) column.")
    country = schema.segment_columns[0]

    needed = [inv, code, desc, qty, ts_col, price, cust, country, "row_class", "line_revenue"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise SqlStoreError(f"Missing columns {missing}. Run clean_types and classify_retail_rows first.")

    ts = pd.to_datetime(df[ts_col], errors="coerce")
    bad_dates, bad_invoices = int(ts.isna().sum()), int(df[inv].isna().sum())
    if bad_dates or bad_invoices:
        raise SqlStoreError(
            f"{bad_dates} rows have no valid date and {bad_invoices} have no invoice number; "
            "resolve these before building the database."
        )
    dates = ts.dt.strftime("%Y-%m-%d %H:%M:%S")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not overwrite:
            raise SqlStoreError(f"{path} already exists; pass overwrite=True to rebuild it.")
        path.unlink()

    conn = sqlite3.connect(path)
    try:
        conn.executescript(load_sql("schema"))
        for start in range(0, len(df), chunk_rows):
            end = start + chunk_rows
            chunk = df.iloc[start:end]
            rows = zip(
                _col(chunk[inv], str), _col(chunk[code], str), _col(chunk[desc], str),
                _col(chunk[qty], float), _col(dates.iloc[start:end], str), _col(chunk[price], float),
                _col(chunk[cust], int), _col(chunk[country], str),
                _col(chunk["row_class"], str), _col(chunk["line_revenue"], float),
            )
            conn.executemany(_INSERT, rows)
        conn.executescript(load_sql("indexes"))
        meta = {
            "source": schema.name,
            "source_kind": schema.source,
            "n_lines": str(len(df)),
            "built_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        conn.executemany("INSERT INTO meta (key, value) VALUES (?, ?)", list(meta.items()))
        conn.commit()
    finally:
        conn.close()
    return {"path": str(path), "n_lines": len(df), "size_mb": round(path.stat().st_size / 1e6, 1)}


# ------------------------------------------------------------------- reading
def _authorizer(action, arg1, arg2, db_name, source) -> int:
    return _SQLITE_OK if action in _ALLOWED_ACTIONS else _SQLITE_DENY


def open_readonly(path: str | Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Read-only connection that can only run SELECT statements."""
    p = Path(path).resolve()
    if not p.exists():
        raise SqlStoreError(f"Database not found: {p}. Run `python -m data.build_sqlite` first.")
    conn = sqlite3.connect(f"{p.as_uri()}?mode=ro", uri=True, check_same_thread=False)
    conn.set_authorizer(_authorizer)
    return conn


def _execute(conn: sqlite3.Connection, sql: str, params: dict | None,
             max_rows: int, timeout_s: float) -> pd.DataFrame:
    deadline = time.monotonic() + timeout_s

    def _stop() -> int:
        return 1 if time.monotonic() > deadline else 0

    conn.set_progress_handler(_stop, 10_000)
    try:
        cur = conn.execute(sql, params or {})
        columns = [d[0] for d in cur.description] if cur.description else []
        rows = cur.fetchmany(max_rows + 1)
    except (sqlite3.Error, sqlite3.Warning) as exc:
        raise SqlStoreError(f"SQL error: {exc}") from exc
    finally:
        conn.set_progress_handler(None, 0)
    out = pd.DataFrame(rows[:max_rows], columns=columns)
    out.attrs["truncated"] = len(rows) > max_rows
    return out


def validate_select(sql: str) -> str:
    """Cheap pre-check. The authorizer and the read-only connection are the real guard."""
    if not isinstance(sql, str) or not sql.strip():
        raise SqlSafetyError("Empty query.")
    cleaned = sql.strip()
    while cleaned.endswith(";"):
        cleaned = cleaned[:-1].rstrip()
    if ";" in cleaned:
        raise SqlSafetyError("Only one statement is allowed.")
    if not cleaned or cleaned.split(None, 1)[0].lower() not in ("select", "with"):
        raise SqlSafetyError("Only SELECT queries are allowed.")
    return cleaned


def run_select(conn: sqlite3.Connection, sql: str, params: dict | None = None,
               max_rows: int = 1000, timeout_s: float = 10.0) -> pd.DataFrame:
    """Run one user-supplied SELECT with a row cap and a time limit."""
    return _execute(conn, validate_select(sql), params, max_rows, timeout_s)


def run_named_query(conn: sqlite3.Connection, name: str, params: dict | None = None,
                    max_rows: int = 100_000, timeout_s: float = 120.0) -> pd.DataFrame:
    """Run one of the whitelisted queries in sql/ with bound parameters."""
    if name not in NAMED_QUERIES:
        raise SqlStoreError(f"Unknown query {name!r}. Available: {', '.join(NAMED_QUERIES)}")
    return _execute(conn, load_sql(name), params, max_rows, timeout_s)


def describe_database(conn: sqlite3.Connection) -> pd.DataFrame:
    """Tables and views with row counts, to show the grain in the UI."""
    objs = _execute(
        conn,
        "SELECT name, type FROM sqlite_master WHERE type IN ('table', 'view') "
        "AND name NOT LIKE 'sqlite_%' ORDER BY type, name",
        None, 1000, 10.0,
    )
    rows = []
    for name, kind in objs.itertuples(index=False):
        n = _execute(conn, f'SELECT COUNT(*) AS n FROM "{name}"', None, 1, 120.0).iloc[0, 0]
        rows.append({"name": name, "type": kind, "rows": int(n)})
    return pd.DataFrame(rows)


def get_meta(conn: sqlite3.Connection) -> dict[str, str]:
    out = _execute(conn, "SELECT key, value FROM meta", None, 100, 10.0)
    return dict(zip(out["key"], out["value"]))


# --------------------------------------------------------- SQL/Python parity
def _close(a: Any, b: Any, rel: float = 1e-9, abs_tol: float = 1e-9) -> bool:
    a = float("nan") if a is None else float(a)
    b = float("nan") if b is None else float(b)
    if math.isnan(a) and math.isnan(b):
        return True
    if math.isnan(a) or math.isnan(b):
        return False
    return math.isclose(a, b, rel_tol=rel, abs_tol=abs_tol)


class _Collector:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def add(self, name: str, sql_value: Any, python_value: Any) -> None:
        try:
            diff = abs(float(sql_value) - float(python_value))
        except (TypeError, ValueError):
            diff = float("nan")
        self.rows.append({"check": name, "sql": sql_value, "python": python_value,
                          "abs_diff": diff, "ok": _close(sql_value, python_value)})

    def add_array(self, name: str, sql_values, python_values) -> None:
        a = np.asarray(sql_values, dtype="float64")
        b = np.asarray(python_values, dtype="float64")
        if a.shape != b.shape:
            self.rows.append({"check": name, "sql": f"{a.size} values", "python": f"{b.size} values",
                              "abs_diff": float("nan"), "ok": False})
            return
        gap = np.abs(a - b)
        diff = float(np.nanmax(gap)) if np.isfinite(gap).any() else 0.0
        ok = bool(np.allclose(a, b, rtol=1e-9, atol=1e-9, equal_nan=True))
        self.rows.append({"check": name, "sql": f"{a.size} values", "python": f"{b.size} values",
                          "abs_diff": diff, "ok": ok})

    def fail(self, name: str, detail: str) -> None:
        self.rows.append({"check": name, "sql": detail, "python": "", "abs_diff": float("nan"), "ok": False})


def parity_report(
    conn: sqlite3.Connection,
    df: pd.DataFrame,
    schema: DatasetSchema = ONLINE_RETAIL_II,
    period_freq: str = "Q",
    baseline: str | None = None,
    comparison: str | None = None,
) -> pd.DataFrame:
    """Compute the same metrics in SQL and in Python and compare them.
    `df` must be the classified frame the database was built from."""
    c = _Collector()

    # 1. headline metrics
    sql_head = run_named_query(conn, "headline_metrics").iloc[0]
    for key, value in metric_summary(df, schema).items():
        c.add(f"headline.{key}", sql_head[key], value)

    # 2. monthly orders (order grain)
    orders_m = order_level_frame(df, schema, "M")
    py_m = orders_m.groupby("period").agg(
        orders=("order_value", "size"),
        gross_sales=("order_value", "sum"),
        average_order_value=("order_value", "mean"),
    )
    sq_m = run_named_query(conn, "monthly_orders").set_index("period")[
        ["orders", "gross_sales", "average_order_value"]
    ]
    c.add("monthly.n_periods", len(sq_m), len(py_m))
    if set(sq_m.index) == set(py_m.index):
        aligned = sq_m.reindex(py_m.index)
        for col in py_m.columns:
            c.add_array(f"monthly.{col}", aligned[col].to_numpy(float), py_m[col].to_numpy(float))
    else:
        c.fail("monthly.period_labels", "SQL and Python produced different months")

    # 3. cohort retention counts
    res = cohort_retention(orders_m, freq="M", min_cohort_size=1)
    sq_c = run_named_query(conn, "cohort_retention")
    pivot = sq_c.pivot(index="cohort", columns="months_since_first", values="active_customers")
    outside = (set(pivot.index) - set(res.counts.index)) | (set(pivot.columns) - set(res.counts.columns))
    c.add("cohort.sql_only_cells", len(outside), 0)
    c.add("cohort.cohorts_with_customers", len(pivot), int((res.cohort_sizes > 0).sum()))
    full = pivot.reindex(index=res.counts.index, columns=res.counts.columns).fillna(0.0)
    observable = res.counts.notna().to_numpy()
    c.add_array("cohort.counts", full.to_numpy(float)[observable], res.counts.to_numpy(float)[observable])

    # 4. repeat purchase gaps
    py_rep, _ = repeat_purchase_gaps(orders_m)
    sq_rep = run_named_query(conn, "repeat_gaps").iloc[0]
    for key in ("n_customers", "n_repeat_customers", "n_gaps", "mean_days"):
        c.add(f"repeat.{key}", sq_rep[key], py_rep[key])

    # 5. revenue concentration
    fracs = (0.01, 0.10, 0.20)
    py_conc = revenue_concentration(orders_m, top_fracs=fracs, n_boot=1)
    for i, frac in enumerate(fracs):
        row = run_named_query(conn, "revenue_concentration", {"frac": frac}).iloc[0]
        key = f"top_{int(round(frac * 100))}pct"
        c.add(f"concentration.{key}", row["top_share"], py_conc.top_shares[key])
        if i == 0:
            c.add("concentration.n_customers", row["n_customers"], py_conc.n_customers)
            c.add("concentration.gini", row["gini"], py_conc.gini)

    # 6. contradiction engine vs SQL group-by
    orders_p = orders_m if period_freq == "M" else order_level_frame(df, schema, period_freq)
    labels = sorted(orders_p["period"].dropna().unique())
    if baseline is None or comparison is None:
        if len(labels) < 2:
            raise SqlStoreError("Need at least two periods for the segment parity check.")
        baseline, comparison = labels[-2], labels[-1]
    seg_res = investigate_contradiction(
        orders_p, MetricSpec("average order value", "order_value"),
        "period", baseline, comparison, schema.segment_columns[0],
    )
    sq = run_named_query(conn, "segment_period_aov",
                         {"freq": period_freq, "baseline": baseline, "comparison": comparison})
    lookup = {(r.segment, r.period): (r.n, r.average_order_value) for r in sq.itertuples(index=False)}
    seg = seg_res.segments
    c.add("segment.n_segments", sq["segment"].nunique(), len(seg))
    for label, period, n_col, v_col in (("baseline", baseline, "n_baseline", "value_baseline"),
                                        ("comparison", comparison, "n_comparison", "value_comparison")):
        c.add_array(f"segment.{n_col}", [lookup.get((s, period), (0, np.nan))[0] for s in seg["segment"]], seg[n_col])
        c.add_array(f"segment.{v_col}", [lookup.get((s, period), (0, np.nan))[1] for s in seg["segment"]], seg[v_col])
        part = sq[sq["period"] == period]
        c.add(f"segment.overall_{label}", part["total_value"].sum() / part["n"].sum(),
              seg_res.aggregate[f"value_{label}"])

    return pd.DataFrame(c.rows)