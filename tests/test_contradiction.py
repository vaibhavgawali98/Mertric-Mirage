import numpy as np
import pandas as pd
import pytest

from core.contracts import EvidenceLevel, FindingCategory
from core.contradiction import ContradictionError, MetricSpec, investigate_contradiction
from data.synthetic import (
    GROUND_TRUTH, SIMPSON_SPEC, make_magnitude_only, make_mix_shift, make_null_case, make_simpson,
)

RATE = MetricSpec("conversion rate", "converted")


def run(df, **kw):
    return investigate_contradiction(df, RATE, "period", "Q1", "Q2", "segment", **kw)


def df_from_counts(spec):
    parts = []
    for (period, seg), (n, conv) in spec.items():
        converted = np.r_[np.ones(conv), np.zeros(n - conv)]
        parts.append(pd.DataFrame({"period": period, "segment": seg, "converted": converted}))
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------- known-answer verdicts
def test_simpson_known_values():
    res = run(make_simpson(), data_source="synthetic")
    gt = GROUND_TRUTH["simpson"]
    assert res.aggregate["value_baseline"] == pytest.approx(gt["aggregate_rate"]["Q1"])
    assert res.aggregate["value_comparison"] == pytest.approx(gt["aggregate_rate"]["Q2"])
    assert res.aggregate["change"] == pytest.approx(0.13)
    assert res.mix_effect == pytest.approx(gt["mix_effect"], abs=1e-9)
    assert res.rate_effect == pytest.approx(gt["rate_effect"], abs=1e-9)
    assert res.entry_exit_effect == pytest.approx(0.0)
    assert res.verdict == "simpson_reversal"
    assert res.n_eligible == 2 and res.n_opposing == 2
    assert res.aggregate["p_value"] < 1e-6
    # At this sample size the segment-level declines are within noise:
    assert res.n_opposing_significant == 0


def test_simpson_finding_is_descriptive_and_flagged_simulated():
    f = run(make_simpson(), data_source="synthetic").to_finding()
    assert f.category == FindingCategory.AGGREGATE_VS_SEGMENT
    assert f.evidence_level == EvidenceLevel.DESCRIPTIVE
    assert f.is_simulated and f.alternatives and f.parameters["period_col"] == "period"


def test_same_structure_with_20x_data_becomes_statistical_lead():
    spec = {k: (n * 20, c * 20) for k, (n, c) in SIMPSON_SPEC.items()}
    res = run(df_from_counts(spec))
    assert res.verdict == "simpson_reversal"
    assert res.n_opposing_significant == 2
    assert res.to_finding().evidence_level == EvidenceLevel.STATISTICAL_LEAD


def test_mix_shift_only_is_mix_driven():
    res = run(make_mix_shift())
    assert res.verdict == "mix_driven"
    assert res.rate_effect == pytest.approx(0.0, abs=1e-12)
    assert res.mix_effect == pytest.approx(0.12)
    assert res.mix_share_of_change == pytest.approx(1.0)


def test_magnitude_only_is_not_flagged():
    res = run(make_magnitude_only())
    assert res.verdict == "consistent"
    assert not res.is_notable and res.to_finding() is None
    assert res.mix_effect == pytest.approx(0.0, abs=1e-12)
    assert res.rate_effect == pytest.approx(0.03)


def test_null_case_cannot_produce_a_mix_or_reversal_finding():
    res = run(make_null_case())
    # Equal cell sizes make the mix effect exactly zero, so no reversal is possible.
    assert res.mix_effect == pytest.approx(0.0, abs=1e-12)
    assert res.verdict not in {"simpson_reversal", "partial_reversal", "mix_driven"}


def test_partial_reversal():
    spec = {
        ("Q1", "A"): (200, 100), ("Q2", "A"): (600, 270),
        ("Q1", "B"): (400, 40), ("Q2", "B"): (200, 16),
        ("Q1", "C"): (400, 40), ("Q2", "C"): (200, 24),
    }
    res = run(df_from_counts(spec))
    assert res.aggregate["change"] == pytest.approx(0.13)
    assert res.mix_effect == pytest.approx(0.15) and res.rate_effect == pytest.approx(-0.02)
    assert res.verdict == "partial_reversal"
    assert res.n_opposing == 2 and res.n_eligible == 3


def test_hidden_movement_when_aggregate_is_flat():
    spec = {
        ("Q1", "A"): (5000, 1000), ("Q2", "A"): (5000, 1500),
        ("Q1", "B"): (5000, 2000), ("Q2", "B"): (5000, 1500),
    }
    res = run(df_from_counts(spec))
    assert res.aggregate["change"] == pytest.approx(0.0)
    assert res.verdict == "hidden_movement"
    assert res.n_opposing == 0


@pytest.mark.parametrize("maker", [make_simpson, make_mix_shift, make_magnitude_only, make_null_case])
def test_decomposition_always_sums_to_total_change(maker):
    res = run(maker())
    total = res.mix_effect + res.rate_effect + res.entry_exit_effect
    assert total == pytest.approx(res.aggregate["change"], abs=1e-9)


# ------------------------------------------------------------ ratio metrics
def test_ratio_metric_aggregate_and_decomposition():
    rows = []
    for period, seg, pairs in [
        ("P0", "X", [(60, 6), (40, 4)]), ("P0", "Y", [(20, 15), (30, 25)]),
        ("P1", "X", [(50, 3), (40, 2)]), ("P1", "Y", [(30, 20), (50, 25)]),
    ]:
        rows += [{"period": period, "segment": seg, "rev": r, "orders": o} for r, o in pairs]
    df = pd.DataFrame(rows)
    res = investigate_contradiction(df, MetricSpec("revenue per order", "rev", "orders"),
                                    "period", "P0", "P1", "segment", min_segment_n=2, min_coverage=0.0)
    assert res.aggregate["value_baseline"] == pytest.approx(3.0)
    assert res.aggregate["value_comparison"] == pytest.approx(3.4)
    assert res.mix_effect + res.rate_effect == pytest.approx(0.4)


# -------------------------------------------------- edge cases and honesty
def test_segment_with_zero_numerator_and_denominator_is_not_evaluated():
    rows = ([{"period": p, "segment": "A", "rev": 10.0, "den": 2.0} for p in ("P0", "P1") for _ in range(40)]
            + [{"period": p, "segment": "B", "rev": 0.0, "den": 0.0} for p in ("P0", "P1") for _ in range(40)])
    res = investigate_contradiction(pd.DataFrame(rows), MetricSpec("m", "rev", "den"),
                                    "period", "P0", "P1", "segment", min_coverage=0.0)
    b = res.segments.set_index("segment").loc["B"]
    assert not b["evaluable"] and np.isnan(b["value_baseline"])


def test_numerator_without_denominator_is_rejected():
    rows = [{"period": p, "segment": s, "rev": 5.0, "den": d}
            for p in ("P0", "P1") for s, d in (("A", 1.0), ("B", 0.0)) for _ in range(5)]
    with pytest.raises(ContradictionError):
        investigate_contradiction(pd.DataFrame(rows), MetricSpec("m", "rev", "den"),
                                  "period", "P0", "P1", "segment", min_segment_n=2)


def test_zero_total_denominator_in_a_period_is_rejected():
    df = pd.DataFrame({"period": ["P0", "P0", "P1", "P1"], "segment": list("ABAB"),
                       "rev": [1.0, 2.0, 0.0, 0.0], "den": [1.0, 1.0, 0.0, 0.0]})
    with pytest.raises(ContradictionError):
        investigate_contradiction(df, MetricSpec("m", "rev", "den"), "period", "P0", "P1", "segment")


def test_missing_segment_values_become_an_explicit_segment():
    df = make_simpson()
    df.loc[df.index[:50], "segment"] = np.nan
    res = run(df)
    assert res.missing["rows_missing_segment"] == 50
    assert "(missing)" in set(res.segments["segment"])
    assert res.mix_effect + res.rate_effect + res.entry_exit_effect == pytest.approx(
        res.aggregate["change"], abs=1e-9)


def test_segment_present_in_only_one_period_uses_entry_exit_and_stays_exact():
    df = make_simpson()
    df = df[~((df["period"] == "Q2") & (df["segment"] == "New"))]
    res = run(df)
    assert res.entry_exit_effect != 0
    assert res.verdict == "insufficient_data"
    assert res.mix_effect + res.rate_effect + res.entry_exit_effect == pytest.approx(
        res.aggregate["change"], abs=1e-9)


def test_too_few_rows_per_segment_means_insufficient_data():
    res = run(make_simpson(), min_segment_n=10_000)
    assert res.verdict == "insufficient_data" and res.n_eligible == 0
    assert res.to_finding() is None


def test_invalid_inputs_raise():
    df = make_simpson()
    with pytest.raises(ContradictionError):
        run(pd.DataFrame())
    with pytest.raises(ContradictionError):
        investigate_contradiction(df, RATE, "period", "Q1", "Q2", "nope")
    with pytest.raises(ContradictionError):
        investigate_contradiction(df, RATE, "period", "Q1", "Q1", "segment")
    with pytest.raises(ContradictionError):
        investigate_contradiction(df, RATE, "period", "Q1", "Q9", "segment")
    with pytest.raises(ContradictionError):
        investigate_contradiction(df, MetricSpec("m", "segment"), "period", "Q1", "Q2", "segment")
    with pytest.raises(ContradictionError):
        run(df, min_segment_n=1)