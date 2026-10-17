"""Synthetic teaching datasets with KNOWN ground truth. EVERYTHING HERE IS SIMULATED.

Never present findings from these tables as discoveries about real data.
Run `python -m data.synthetic` to write them as CSVs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PERIOD_WINDOWS = {"Q1": ("2025-01-01", 90), "Q2": ("2025-04-01", 91)}

# (period, segment) -> (sessions, conversions). Exact counts, so results are hand-checkable.
SIMPSON_SPEC = {
    ("Q1", "Returning"): (300, 180),
    ("Q1", "New"): (700, 70),
    ("Q2", "Returning"): (600, 348),
    ("Q2", "New"): (400, 32),
}
MIX_SHIFT_SPEC = {  # segment rates unchanged; only the mix moves
    ("Q1", "A"): (200, 80),
    ("Q1", "B"): (800, 80),
    ("Q2", "A"): (600, 240),
    ("Q2", "B"): (400, 40),
}
MAGNITUDE_SPEC = {  # everything rises, mix unchanged: NOT a paradox
    ("Q1", "A"): (500, 100),
    ("Q1", "B"): (500, 50),
    ("Q2", "A"): (500, 125),
    ("Q2", "B"): (500, 55),
}

GROUND_TRUTH = {
    "simpson": {
        "aggregate_rate": {"Q1": 0.25, "Q2": 0.38},
        "segment_rate": {
            ("Q1", "Returning"): 0.60, ("Q2", "Returning"): 0.58,
            ("Q1", "New"): 0.10, ("Q2", "New"): 0.08,
        },
        "total_change": 0.13, "mix_effect": 0.15, "rate_effect": -0.02,
    },
    "mix_shift": {
        "aggregate_rate": {"Q1": 0.16, "Q2": 0.28},
        "total_change": 0.12, "mix_effect": 0.12, "rate_effect": 0.0,
    },
    "magnitude": {
        "aggregate_rate": {"Q1": 0.15, "Q2": 0.18},
        "total_change": 0.03, "mix_effect": 0.0, "rate_effect": 0.03,
    },
    "null_case": {"true_rate_all_cells": 0.15},
    "heavy_tail": {"n_outliers": 5, "outlier_value": 20000.0},
    "ab_test": {
        "n_per_arm": 10000, "control_conversions": 1000, "treatment_conversions": 1200,
        "true_effect": 0.02,
    },
}


def _finish(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    df = df.sort_values(df.columns[-1]).reset_index(drop=True) if False else df
    df.insert(0, id_col, np.arange(1, len(df) + 1))
    df["is_simulated"] = True
    return df


def _sessions(period: str, segment: str, converted: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    start, days = PERIOD_WINDOWS[period]
    offsets = rng.integers(0, days, size=len(converted))
    dates = pd.Timestamp(start) + pd.to_timedelta(offsets, unit="D")
    return pd.DataFrame(
        {"period": period, "segment": segment, "converted": converted, "session_date": dates}
    )


def _from_spec(spec: dict, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = []
    for (period, segment), (n, conversions) in spec.items():
        converted = np.zeros(n, dtype=int)
        converted[:conversions] = 1
        rng.shuffle(converted)
        frames.append(_sessions(period, segment, converted, rng))
    df = pd.concat(frames, ignore_index=True).sort_values("session_date").reset_index(drop=True)
    return _finish(df, "session_id")


def make_simpson(seed: int = 42) -> pd.DataFrame:
    return _from_spec(SIMPSON_SPEC, seed)


def make_mix_shift(seed: int = 42) -> pd.DataFrame:
    return _from_spec(MIX_SHIFT_SPEC, seed)


def make_magnitude_only(seed: int = 42) -> pd.DataFrame:
    return _from_spec(MAGNITUDE_SPEC, seed)


def make_null_case(seed: int = 42, n_per_cell: int = 1000, rate: float = 0.15) -> pd.DataFrame:
    """No real difference anywhere: every cell shares one true rate (random draws)."""
    rng = np.random.default_rng(seed)
    frames = []
    for period in ("Q1", "Q2"):
        for segment in ("A", "B"):
            converted = (rng.random(n_per_cell) < rate).astype(int)
            frames.append(_sessions(period, segment, converted, rng))
    df = pd.concat(frames, ignore_index=True).sort_values("session_date").reset_index(drop=True)
    return _finish(df, "session_id")


def make_heavy_tail(seed: int = 42, n_per_period: int = 2000) -> pd.DataFrame:
    """Same lognormal order values in both periods, plus 5 huge orders in Q2.
    The mean jumps; the median does not."""
    rng = np.random.default_rng(seed)
    frames = []
    for period in ("Q1", "Q2"):
        values = np.round(rng.lognormal(mean=4.0, sigma=0.5, size=n_per_period), 2)
        if period == "Q2":
            n_out = GROUND_TRUTH["heavy_tail"]["n_outliers"]
            values = np.concatenate([values, np.full(n_out, GROUND_TRUTH["heavy_tail"]["outlier_value"])])
        start, days = PERIOD_WINDOWS[period]
        dates = pd.Timestamp(start) + pd.to_timedelta(rng.integers(0, days, size=len(values)), unit="D")
        frames.append(pd.DataFrame({"period": period, "order_value": values, "order_date": dates}))
    df = pd.concat(frames, ignore_index=True).sort_values("order_date").reset_index(drop=True)
    return _finish(df, "order_id")


def make_ab_test(seed: int = 42) -> pd.DataFrame:
    """Randomized test with exact conversion counts: control 1000/10000 (10%),
    treatment 1200/10000 (12%). `device` is assigned independently of arm."""
    rng = np.random.default_rng(seed)
    gt = GROUND_TRUTH["ab_test"]
    n = gt["n_per_arm"]
    frames = []
    for arm, conv in (("control", gt["control_conversions"]), ("treatment", gt["treatment_conversions"])):
        converted = np.zeros(n, dtype=int)
        converted[:conv] = 1
        rng.shuffle(converted)
        device = np.where(rng.random(n) < 0.6, "mobile", "desktop")
        frames.append(pd.DataFrame({"arm": arm, "device": device, "converted": converted}))
    df = pd.concat(frames, ignore_index=True)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return _finish(df, "user_id")


def make_all(seed: int = 42) -> dict[str, pd.DataFrame]:
    return {
        "simpson": make_simpson(seed),
        "mix_shift": make_mix_shift(seed),
        "magnitude": make_magnitude_only(seed),
        "null_case": make_null_case(seed),
        "heavy_tail": make_heavy_tail(seed),
        "ab_test": make_ab_test(seed),
    }


def write_all(out_dir: str | Path = "data/processed/synthetic", seed: int = 42) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, df in make_all(seed).items():
        p = out / f"synthetic_{name}.csv"
        df.to_csv(p, index=False)
        paths.append(p)
    return paths


if __name__ == "__main__":
    for p in write_all():
        print("wrote", p)
    print("All files are SIMULATED teaching data.")