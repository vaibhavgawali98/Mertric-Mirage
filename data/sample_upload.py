"""A small, reproducible, clearly SIMULATED orders file for trying the upload flow.

It promises no particular finding. It has the shape most business exports have: one row per
order with an ID, a date, a customer, two segment columns and a revenue amount. The file carries
an `is_simulated` column, so anything derived from it is labelled SIMULATED in the app.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SAMPLE_NAME = "sample_orders_SIMULATED.csv"
REGIONS = ["North", "South", "East", "West"]
CHANNELS = ["Web", "Store", "Partner"]
REGION_MULT = {"North": 1.1, "South": 0.9, "East": 1.0, "West": 1.2}
CHANNEL_MULT = {"Web": 1.0, "Store": 1.25, "Partner": 0.7}


def make_sample_orders(seed: int = 2024, n_customers: int = 900) -> pd.DataFrame:
    """Orders from January 2023 to June 2024. The share of the lower-value 'Partner' channel
    grows over time, so the mix changes while each channel's order value is stable."""
    rng = np.random.default_rng(seed)
    months = pd.period_range("2023-01", "2024-06", freq="M")
    n_months = len(months)
    weights = np.linspace(1.4, 0.8, n_months)
    weights = weights / weights.sum()
    acquired = rng.choice(n_months, size=n_customers, p=weights)
    region_of = rng.choice(REGIONS, size=n_customers, p=[0.35, 0.25, 0.25, 0.15])

    rows = []
    for c in range(n_customers):
        n_orders = 1 + int(rng.poisson(1.8))
        span = max(n_months - int(acquired[c]), 1)
        offsets = np.sort(rng.integers(0, span, size=n_orders))
        offsets[0] = 0
        for off in offsets:
            m = int(acquired[c]) + int(off)
            partner = 0.10 + 0.25 * m / (n_months - 1)
            channel = str(rng.choice(CHANNELS, p=[(1 - partner) * 0.6, (1 - partner) * 0.4, partner]))
            region = str(region_of[c])
            amount = 80 * REGION_MULT[region] * CHANNEL_MULT[channel] * float(rng.lognormal(0, 0.55))
            day = int(rng.integers(1, months[m].days_in_month + 1))
            date = months[m].start_time + pd.Timedelta(days=day - 1)
            rows.append((date, f"C{c + 1:04d}", region, channel, round(amount, 2)))

    df = pd.DataFrame(rows, columns=["order_date", "customer_id", "region", "channel", "revenue"])
    df = df.sort_values("order_date", kind="mergesort").reset_index(drop=True)
    df.insert(0, "order_id", [f"ORD-{i + 1:06d}" for i in range(len(df))])
    df["order_date"] = df["order_date"].dt.strftime("%Y-%m-%d")
    df["is_simulated"] = True
    return df


def sample_csv_bytes(seed: int = 2024) -> bytes:
    return make_sample_orders(seed).to_csv(index=False).encode("utf-8")
