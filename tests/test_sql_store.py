import numpy as np
import pandas as pd
import pytest

from core.data_loader import classify_retail_rows, clean_types
from core.sql_store import (
    NAMED_QUERIES, SqlSafetyError, SqlStoreError, build_database, describe_database,
    load_sql, open_readonly, parity_report, run_named_query, run_select,
)

COLS = ["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "Customer ID", "Country"]


def _classified(rows):
    cleaned, _ = clean_types(pd.DataFrame(rows, columns=COLS))
    return classify_retail_rows(cleaned)


def _random_retail_frame(seed=7, n_customers=120):
    rng = np.random.default_rng(seed)
    rows, inv = [], 1000
    start = pd.Timestamp("2010-01-03")

    def stamp():
        return start + pd.Timedelta(days=int(rng.integers(0, 330)), hours=int(rng.integers(8, 17)),
                                    minutes=int(rng.integers(0, 60)))

    def lines(invoice, customer, country, date, sign=1):
        for _ in range(int(rng.integers(1, 4))):
            rows.append([invoice, "85123A", "x", sign * int(rng.integers(1, 10)), date,
                         round(float(rng.uniform(1, 20)), 2), customer, country])

    for cust in range(1, n_customers + 1):
        country = str(rng.choice(["UK", "France", "Germany"], p=[0.7, 0.2, 0.1]))
        for _ in range(int(rng.integers(1, 6))):
            inv += 1
            lines(str(inv), float(cust), country, stamp())
    for _ in range(30):                                   # anonymous orders
        inv += 1
        lines(str(inv), np.nan, "UK", stamp())
    for _ in range(15):                                   # returns
        inv += 1
        lines(f"C{inv}", float(rng.integers(1, n_customers + 1)), "UK", stamp(), sign=-1)
    rows.append([str(inv + 1), "POST", "postage", 1, stamp(), 5.0, 1.0, "UK"])      # non-product
    rows.append([str(inv + 2), "85123A", "free", 1, stamp(), 0.0, 1.0, "UK"])       # zero price
    return _classified(rows)


@pytest.fixture(scope="module")
def retail(tmp_path_factory):
    df = _random_retail_frame()
    path = tmp_path_factory.mktemp("sql") / "retail.db"
    build_database(df, path)
    conn = open_readonly(path)
    yield df, conn
    conn.close()


def test_database_row_counts_match_the_frame(retail):
    df, conn = retail
    assert run_select(conn, "SELECT COUNT(*) AS n FROM transactions").iloc[0, 0] == len(df)
    n_orders = run_select(conn, "SELECT COUNT(*) AS n FROM orders").iloc[0, 0]
    assert n_orders == df.loc[df["row_class"] == "sale", "Invoice"].nunique()
    by_class = run_select(conn, "SELECT row_class, COUNT(*) AS n FROM transactions GROUP BY row_class")
    assert by_class.set_index("row_class")["n"].to_dict() == df["row_class"].value_counts().to_dict()


def test_every_sql_and_python_metric_agrees(retail):
    df, conn = retail
    report = parity_report(conn, df, period_freq="Q", baseline="2010Q1", comparison="2010Q3")
    assert len(report) > 20
    failed = report[~report["ok"]]
    assert failed.empty, failed.to_string()


def test_sql_headline_matches_hand_calculation(tmp_path):
    rows = [
        ["1001", "85123A", "x", 2, "2010-01-05", 5.0, 1.0, "UK"],      # sale 10
        ["1001", "85124B", "x", 1, "2010-01-05", 10.0, 1.0, "UK"],     # sale 10
        ["1002", "85123A", "x", 4, "2010-01-20", 2.5, 1.0, "UK"],      # sale 10
        ["1003", "85123A", "x", 1, "2010-01-21", 20.0, 2.0, "France"], # sale 20
        ["1004", "85123A", "x", 3, "2010-01-22", 4.0, None, "UK"],     # sale 12, no customer
        ["C1005", "85123A", "x", -1, "2010-01-23", 5.0, 1.0, "UK"],    # return -5
        ["1006", "POST", "x", 1, "2010-01-24", 15.0, 1.0, "UK"],       # non-product
    ]
    path = tmp_path / "tiny.db"
    build_database(_classified(rows), path)
    conn = open_readonly(path)
    try:
        r = run_named_query(conn, "headline_metrics").iloc[0]
    finally:
        conn.close()
    assert r["gross_sales"] == pytest.approx(62.0)
    assert r["returns_value"] == pytest.approx(5.0)
    assert r["net_revenue"] == pytest.approx(57.0)
    assert r["orders"] == 4 and r["active_customers"] == 2
    assert r["average_order_value"] == pytest.approx(15.5)
    assert r["revenue_per_customer"] == pytest.approx(25.0)
    assert r["return_rate_value"] == pytest.approx(5 / 62)
    assert r["repeat_rate"] == pytest.approx(0.5)


def test_only_select_queries_are_accepted(retail):
    _, conn = retail
    for bad in ["DROP TABLE transactions", "DELETE FROM transactions", "SELECT 1; SELECT 2",
                "PRAGMA table_info(transactions)", "", "   "]:
        with pytest.raises(SqlSafetyError):
            run_select(conn, bad)


def test_write_hidden_inside_a_cte_is_blocked_by_the_database(retail):
    df, conn = retail
    with pytest.raises(SqlStoreError):
        run_select(conn, "WITH t AS (SELECT 1) DELETE FROM transactions")
    assert run_select(conn, "SELECT COUNT(*) AS n FROM transactions").iloc[0, 0] == len(df)


def test_runaway_query_is_interrupted(retail):
    _, conn = retail
    with pytest.raises(SqlStoreError, match="interrupted"):
        run_select(
            conn,
            "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT COUNT(*) FROM c",
            timeout_s=0.3,
        )


def test_row_cap_truncates_and_says_so(retail):
    _, conn = retail
    out = run_select(conn, "SELECT * FROM transactions", max_rows=5)
    assert len(out) == 5 and out.attrs["truncated"] is True


def test_unknown_named_queries_are_rejected_and_files_exist(retail):
    _, conn = retail
    for name in ["nope", "../schema", "headline_metrics.sql"]:
        with pytest.raises(SqlStoreError):
            run_named_query(conn, name)
    for name in NAMED_QUERIES:
        assert "SELECT" in load_sql(name).upper()


def test_build_and_open_reject_bad_inputs(tmp_path):
    with pytest.raises(SqlStoreError):
        build_database(pd.DataFrame(), tmp_path / "a.db")
    with pytest.raises(SqlStoreError):
        build_database(pd.DataFrame({"x": [1]}), tmp_path / "b.db")
    good = _classified([["1", "85123A", "x", 1, "2010-01-05", 5.0, 1.0, "UK"]])
    bad_date = good.copy()
    bad_date.loc[0, "InvoiceDate"] = pd.NaT
    with pytest.raises(SqlStoreError):
        build_database(bad_date, tmp_path / "c.db")
    path = tmp_path / "d.db"
    build_database(good, path)
    with pytest.raises(SqlStoreError):
        build_database(good, path, overwrite=False)
    with pytest.raises(SqlStoreError):
        open_readonly(tmp_path / "missing.db")


def test_describe_database_lists_the_grain(retail):
    _, conn = retail
    d = describe_database(conn).set_index("name")
    assert {"transactions", "orders", "meta"} <= set(d.index)
    assert d.loc["orders", "type"] == "view" and d.loc["transactions", "type"] == "table"