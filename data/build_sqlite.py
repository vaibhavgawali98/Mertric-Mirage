"""Build data/processed/retail.db from the cleaned CSV and run the SQL/Python parity checks.

Run from the project root:  python -m data.build_sqlite
"""
import time

import pandas as pd

from core.data_loader import load_processed
from core.sql_store import (
    DEFAULT_DB_PATH, build_database, describe_database, open_readonly, parity_report,
)

pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 100)


def main() -> None:
    start = time.time()
    print("Loading cleaned data...")
    df = load_processed()
    print(f"Building {DEFAULT_DB_PATH} ...")
    print(build_database(df, DEFAULT_DB_PATH))
    conn = open_readonly(DEFAULT_DB_PATH)
    print("\n=== TABLES AND VIEWS (grain) ===")
    print(describe_database(conn).to_string(index=False))
    print("\nRunning SQL vs Python parity checks (this can take a minute)...")
    report = parity_report(conn, df, period_freq="Q", baseline="2010Q4", comparison="2011Q4")
    print("\n=== PARITY REPORT ===")
    print(report.to_string(index=False))
    failed = report[~report["ok"]]
    print(f"\n{len(report) - len(failed)}/{len(report)} checks agree.  ({time.time() - start:.0f}s)")
    if len(failed):
        print("MISMATCHES:")
        print(failed.to_string(index=False))


if __name__ == "__main__":
    main()