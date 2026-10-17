"""Stage 5 smoke run. Prints results only; makes no claims."""
import pandas as pd

from core.cohorts import (
    cohort_retention, compare_cohort_retention, customer_table, funnel_available,
    repeat_purchase_gaps, revenue_concentration,
)
from core.contradiction import MetricSpec, investigate_contradiction
from core.data_loader import load_processed
from core.experiments import (
    balance_check, evidence_tier, minimum_detectable_effect, sample_size_two_proportions,
    srm_check, two_proportion_test,
)
from core.metrics import order_level_frame
from data.synthetic import make_ab_test

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)


def main() -> None:
    df = load_processed()
    orders = order_level_frame(df, freq="M")

    print("=== AGGREGATE AOV CHANGE (the p-value I did not print last time) ===")
    q = order_level_frame(df, freq="Q")
    res = investigate_contradiction(q, MetricSpec("average order value", "order_value"),
                                    "period", "2010Q4", "2011Q4", "Country", data_source="public")
    print({k: round(v, 4) for k, v in res.aggregate.items()})

    print("\n=== INVOICES WITH MORE THAN ONE TIMESTAMP ===")
    spread = (df[df["row_class"] == "sale"].groupby("Invoice")["InvoiceDate"]
              .agg(lambda s: (s.max() - s.min()).total_seconds() / 60))
    multi = spread[spread > 0]
    print(f"{len(multi)} invoices; spread in minutes: {multi.describe().round(1).to_dict()}")

    print("\n=== COHORT RETENTION (rates, first 7 offsets) ===")
    ret = cohort_retention(orders, freq="M")
    print(ret.rates.iloc[:, :7].round(3).to_string())
    print("cohort sizes:", ret.cohort_sizes.astype(int).to_dict())
    for w in ret.warnings:
        print("WARNING:", w)

    print("\n=== DEMO COMPARISON (arbitrary pair; machinery check only) ===")
    t = compare_cohort_retention(ret, "2010-01", "2010-06", 3)
    print(f"retention +3: {t.rate_control:.3f} vs {t.rate_treatment:.3f}, diff {t.risk_difference:+.3f}, "
          f"95% CI [{t.ci_low:+.3f}, {t.ci_high:+.3f}], p={t.p_value:.3g}")

    print("\n=== REPEAT PURCHASE GAPS ===")
    print(repeat_purchase_gaps(orders)[0])
    print("\n=== REVENUE CONCENTRATION (identified customers) ===")
    conc = revenue_concentration(orders)
    print(conc.top_shares, conc.top_share_ci, f"gini={conc.gini:.3f}", f"largest={conc.largest_customer_share:.3%}")
    print("\n=== CUSTOMER TIERS ===")
    print(customer_table(orders)["value_tier"].value_counts().to_dict())
    print("\n=== FUNNEL AVAILABILITY ===")
    print(funnel_available(df, "Customer ID", "stage", ["view", "cart", "buy"]))

    print("\n=== SYNTHETIC A/B TEST (SIMULATED) ===")
    ab = make_ab_test()
    c, tr = ab[ab.arm == "control"]["converted"], ab[ab.arm == "treatment"]["converted"]
    r = two_proportion_test(int(c.sum()), len(c), int(tr.sum()), len(tr))
    print(f"diff {r.risk_difference:+.4f}  CI [{r.ci_low:+.4f}, {r.ci_high:+.4f}]  p={r.p_value:.3g}  ({r.test_used})")
    srm = srm_check(ab["arm"].value_counts().to_dict())
    bal = balance_check(ab, "arm", "control", "treatment", ["device"])
    print("SRM flagged:", srm["flagged"], "| imbalance flagged:", bool(bal["flagged"].any()))
    print(evidence_tier(True, srm["flagged"], bool(bal["flagged"].any()), False)[0].value)
    n = sample_size_two_proportions(0.10, 0.02)
    print(f"n per arm for 10% -> 12%: {n}; MDE at that n: {minimum_detectable_effect(0.10, n):.4f}")


if __name__ == "__main__":
    main()