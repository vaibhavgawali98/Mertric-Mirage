"""Stage 6 smoke run: SQL panel, matched windows, complete-month change-point, evidence
store and report export. Prints results only; makes no claims.

Run after `python -m data.build_sqlite`.
"""
from pathlib import Path

import pandas as pd

from core.cohorts import concentration_finding, revenue_concentration
from core.contradiction import MetricSpec, investigate_contradiction
from core.data_loader import load_processed
from core.evidence import FindingStore
from core.integrity import (
    change_point_candidate, duplicate_finding, duplicate_summary, missing_share_finding,
    missingness_association,
)
from core.metrics import matched_window, order_level_frame, period_series
from core.reports import write_report
from core.sql_store import load_sql, open_readonly, run_named_query
from data.synthetic import make_simpson
from viz.charts import decomposition_spec, lorenz_spec

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
RAW_COLS = ["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "Customer ID", "Country"]
SMOKE_DB = Path("data/processed/evidence_smoke.db")
AOV = MetricSpec("average order value", "order_value")


def main() -> None:
    df = load_processed()
    orders_m = order_level_frame(df, freq="M")

    print("=== INVOICES WHOSE LINES STRADDLE A MONTH BOUNDARY ===")
    sales = df[df["row_class"] == "sale"]
    span = sales.groupby("Invoice")["InvoiceDate"].agg(["min", "max"])
    straddle = (span["min"].dt.to_period("M") != span["max"].dt.to_period("M")).sum()
    print(f"{int(straddle)} of {len(span):,} sale invoices")

    print("\n=== SQL PANEL: monthly_orders (first 6 lines of the query, then 5 result rows) ===")
    print("\n".join(load_sql("monthly_orders").splitlines()[:6]))
    conn = open_readonly()
    print(run_named_query(conn, "monthly_orders").head(5).to_string(index=False))
    conn.close()

    print("\n=== MATCHED WINDOW: AOV by Country, 2010Q4 vs 2011Q4 ===")
    orders_q = order_level_frame(df, freq="Q")
    matched, info = matched_window(orders_q, "order_date", "period", "2010Q4", "2011Q4", "Q")
    print(info)
    res = investigate_contradiction(matched, AOV, "period", "2010Q4", "2011Q4", "Country",
                                    data_source="public")
    print("verdict:", res.verdict)
    print({k: round(v, 4) for k, v in res.aggregate.items() if k in ("value_baseline", "value_comparison", "change", "ci_low", "ci_high", "p_value")})

    print("\n=== CHANGE-POINT ON COMPLETE MONTHS ONLY ===")
    series, dropped = period_series(orders_m)
    print(f"months: {len(series)}, incomplete month dropped: {dropped}")
    print(change_point_candidate(series))

    print("\n=== EVIDENCE STORE ===")
    if SMOKE_DB.exists():
        SMOKE_DB.unlink()
    store = FindingStore(SMOKE_DB)
    ids = []

    summary = duplicate_summary(df, RAW_COLS)
    f = duplicate_finding(summary, data_source="public")
    if f:
        ids.append(store.save(f)[0])
    assoc = missingness_association(df, "Customer ID", "Country")
    f = missing_share_finding("Customer ID", int(df["Customer ID"].isna().sum()), len(df), assoc,
                              data_source="public")
    if f:
        ids.append(store.save(f)[0])
    conc = revenue_concentration(orders_m)
    f = concentration_finding(conc, data_source="public")
    f.chart_spec = lorenz_spec(conc)
    ids.append(store.save(f)[0])

    real_finding = res.to_finding()
    if real_finding:
        real_finding.chart_spec = decomposition_spec(res)
        ids.append(store.save(real_finding)[0])
    else:
        print("matched-window AOV result is not a reportable finding (no card created)")

    sim = investigate_contradiction(make_simpson(), MetricSpec("conversion rate", "converted"),
                                    "period", "Q1", "Q2", "segment", data_source="synthetic")
    sim_finding = sim.to_finding()
    sim_finding.chart_spec = decomposition_spec(sim)
    sim_id, _ = store.save(sim_finding)
    ids.append(sim_id)

    print(store.list_findings()[["id", "category", "evidence_level", "data_source", "title"]].to_string(index=False))

    print("\n=== REPRODUCIBILITY CHECK (simulated finding) ===")
    check = store.verify(sim_id, make_simpson())
    print(f"matched={check.matched} exact={check.exact} -> {check.message}")

    print("\n=== REPORTS ===")
    findings = store.get_many(ids)
    for name in ("smoke_report.md", "smoke_report.html"):
        path = write_report(findings, Path("reports") / name,
                            dataset_note="UCI Online Retail II plus one SIMULATED teaching example.")
        print("wrote", path, f"({path.stat().st_size / 1e6:.1f} MB)")
    store.close()


if __name__ == "__main__":
    main()