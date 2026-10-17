"""One-time preparation of the Online Retail II workbook.

Run from the project root:  python -m data.prepare_retail
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from core.data_loader import (
    PROCESSED_PATH,
    classify_retail_rows,
    clean_types,
    non_product_summary,
    read_retail_excel,
    reconcile_sheet_overlap,
    validate_retail_frame,
)
from core.schema import ONLINE_RETAIL_II, Role

RAW_PATH = Path("data/raw/online_retail_II.xlsx")


def main(raw_path: Path = RAW_PATH, out_path: Path = PROCESSED_PATH) -> None:
    schema = ONLINE_RETAIL_II
    date_col = schema.mapping[Role.TIMESTAMP]

    print(f"Reading {raw_path} (this takes a few minutes)...")
    sheets = read_retail_excel(raw_path, schema)

    # Order sheets chronologically, then fold them together one by one.
    ordered = sorted(
        sheets.items(), key=lambda kv: pd.to_datetime(kv[1][date_col], errors="coerce").min()
    )
    combined = ordered[0][1]
    for name, sheet in ordered[1:]:
        combined, overlap = reconcile_sheet_overlap(combined, sheet, date_col)
        print(f"\nSheet overlap with '{name}': {overlap}")

    cleaned, coerced = clean_types(combined, schema)
    classified = classify_retail_rows(cleaned, schema)
    report = validate_retail_frame(classified, schema, coerced)
    print("\n=== VALIDATION REPORT ===")
    print(report.summary())

    print("\n=== STOCK CODES SET ASIDE AS NON-PRODUCT (top 25; review these) ===")
    print(non_product_summary(classified, schema).to_string())

    out_path.parent.mkdir(parents=True, exist_ok=True)
    classified.to_csv(out_path, index=False, compression="gzip")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as exc:
        sys.exit(str(exc))