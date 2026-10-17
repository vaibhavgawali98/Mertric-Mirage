"""Stage 4 smoke run on the real Online Retail II data. Prints results only."""
import pandas as pd

from core.contradiction import MetricSpec, investigate_contradiction
from core.data_loader import load_processed
from core.integrity import (
    change_point_candidate, compare_periods_robust, distribution_summary, duplicate_summary,
    iqr_outliers, key_conflicts, mad_outliers, missingness_by_column,
)
from core.metrics import metric_summary, order_level_frame

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)


def main() -> None:
    df = load_processed()
    print("=== HEADLINE METRICS ===")
    for k, v in metric_summary(df).items():
        print(f"{k:>22}: {v:,.4f}")
    sales = df[df["row_class"] == "sale"]
    no_id = sales.loc[~sales["has_customer"], "line_revenue"].sum()
    print(f"\nShare of gross sales with no customer ID: {no_id / sales['line_revenue'].sum():.1%}")

    print("\n=== MISSINGNESS BY COLUMN ===")
    print(missingness_by_column(df))
    print("\n=== DUPLICATES (all raw columns) ===")
    raw_cols = ["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "Customer ID", "Country"]
    print(duplicate_summary(df, raw_cols))
    print("\n=== KEY CONFLICTS (one invoice, several values?) ===")
    print(key_conflicts(df, "Invoice", ["Customer ID", "InvoiceDate", "Country"]))

    orders = order_level_frame(df, freq="Q")
    print("\n=== ORDER VALUE DISTRIBUTION ===")
    print(distribution_summary(orders["order_value"]))
    for fn in (iqr_outliers, mad_outliers):
        r = fn(orders["order_value"])
        print(fn.__name__, r.n_outliers, f"{r.share_of_rows:.2%} of rows,", f"{r.share_of_total:.2%} of value")

    print("\n=== CONTRADICTION: average order value by Country, 2010Q4 vs 2011Q4 ===")
    res = investigate_contradiction(
        orders, MetricSpec("average order value", "order_value"),
        "period", "2010Q4", "2011Q4", "Country", data_source="public",
    )
    print("verdict:", res.verdict)
    print(res.explanation)
    for w in res.warnings:
        print("WARNING:", w)
    print(res.segments[["segment", "n_baseline", "n_comparison", "weight_baseline", "weight_comparison",
                        "value_baseline", "value_comparison", "change", "q_value", "total_contribution"]].head(8))

    print("\n=== MEAN vs MEDIAN, 2010Q4 vs 2011Q4 ===")
    print(compare_periods_robust(orders, "order_value", "period", "2010Q4", "2011Q4"))

    print("\n=== CHANGE-POINT CANDIDATE: monthly mean order value ===")
    monthly = order_level_frame(df, freq="M").groupby("period")["order_value"].mean()
    print(change_point_candidate(monthly))


if __name__ == "__main__":
    main()