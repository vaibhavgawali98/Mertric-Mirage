import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from core.contracts import EvidenceLevel, FindingCategory
from core.experiments import (
    EvidenceTier, ExperimentError, adjusted_effect, balance_check, evidence_tier,
    experiment_finding, minimum_detectable_effect, practical_assessment,
    sample_size_two_means, sample_size_two_proportions, srm_check, two_mean_test,
    two_proportion_test,
)
from data.synthetic import GROUND_TRUTH, make_ab_test, make_heavy_tail


def test_known_ab_test_values():
    gt = GROUND_TRUTH["ab_test"]
    r = two_proportion_test(gt["control_conversions"], gt["n_per_arm"],
                            gt["treatment_conversions"], gt["n_per_arm"])
    assert r.risk_difference == pytest.approx(0.02)
    assert r.relative_lift == pytest.approx(0.2)
    assert r.test_used == "two_proportion_z"
    se = math.sqrt(0.11 * 0.89 * 2 / 10000)
    assert r.p_value == pytest.approx(2 * stats.norm.sf(0.02 / se), rel=1e-9)
    assert 0.010 < r.ci_low < 0.02 < r.ci_high < 0.030
    assert r.significant


def test_small_counts_use_fisher():
    r = two_proportion_test(0, 20, 5, 20)
    assert r.test_used == "fisher_exact" and 0 < r.p_value < 0.1


def test_no_events_anywhere_gives_p_one_and_undefined_lift():
    r = two_proportion_test(0, 100, 0, 100)
    assert r.p_value == 1.0 and math.isnan(r.relative_lift)


@pytest.mark.parametrize("args", [(5, 0, 1, 10), (11, 10, 1, 10), (-1, 10, 1, 10), (1.5, 10, 1, 10)])
def test_invalid_counts_raise(args):
    with pytest.raises(ExperimentError):
        two_proportion_test(*args)


def test_welch_matches_scipy_and_interval_contains_difference():
    rng = np.random.default_rng(3)
    a, b = rng.normal(10, 2, 80), rng.normal(11, 3, 120)
    r = two_mean_test(a, b, n_boot=300)
    assert r.p_value == pytest.approx(stats.ttest_ind(b, a, equal_var=False).pvalue, rel=1e-9)
    assert r.ci_low < r.mean_difference < r.ci_high
    assert r.boot_ci_low < r.mean_difference < r.boot_ci_high
    assert r.mean_difference == pytest.approx(b.mean() - a.mean())


def test_constant_and_tiny_groups():
    assert two_mean_test([5, 5, 5], [5, 5, 5], n_boot=20).p_value == 1.0
    shifted = two_mean_test([5, 5, 5], [6, 6, 6], n_boot=20)
    assert shifted.p_value == 0.0 and shifted.ci_low == shifted.ci_high == pytest.approx(1.0)
    with pytest.raises(ExperimentError):
        two_mean_test([1.0], [1.0, 2.0])


def test_heavy_tail_moves_the_mean_but_not_the_median():
    df = make_heavy_tail()
    r = two_mean_test(df.loc[df["period"] == "Q1", "order_value"], df.loc[df["period"] == "Q2", "order_value"],
                      n_boot=200)
    assert r.mean_difference > 30 and abs(r.median_difference) < 5


@pytest.mark.parametrize("lo,hi,expected", [
    (0.5, 1.5, "meaningful_positive"), (-1.5, -0.5, "meaningful_negative"),
    (-0.1, 0.1, "negligible"), (0.05, 0.25, "negligible"), (-0.2, 0.9, "inconclusive"),
])
def test_practical_assessment(lo, hi, expected):
    assert practical_assessment(lo, hi, 0.3) == expected
    with pytest.raises(ExperimentError):
        practical_assessment(0, 1, -1)


def test_power_functions_agree_with_each_other_and_textbook():
    n = sample_size_two_proportions(0.10, 0.02)
    assert 3700 < n < 4000                                   # textbook value is about 3,840
    mde = minimum_detectable_effect(0.10, n)
    assert 0.0195 < mde <= 0.02 + 1e-9
    assert sample_size_two_means(sd=10, mde=2) == pytest.approx(393, abs=3)
    for bad in [(1.2, 0.01), (0.1, 0.0), (0.1, 0.95)]:
        with pytest.raises(ExperimentError):
            sample_size_two_proportions(*bad)


def test_srm_check():
    assert not srm_check({"A": 5000, "B": 5000})["flagged"]
    assert srm_check({"A": 5000, "B": 5500})["flagged"]
    assert not srm_check({"A": 900, "B": 100}, expected_share={"A": 0.9, "B": 0.1})["flagged"]
    with pytest.raises(ExperimentError):
        srm_check({"A": 10})


def _confounded(n=4000, seed=1):
    rng = np.random.default_rng(seed)
    hi = rng.random(n) < 0.5
    treated = rng.random(n) < np.where(hi, 0.8, 0.2)
    y = np.where(hi, 10.0, 0.0) + rng.normal(0, 1, n)        # NO true treatment effect
    return pd.DataFrame({"arm": np.where(treated, "treatment", "control"),
                         "segment": np.where(hi, "hi", "lo"), "y": y})


def test_balance_check_flags_confounding_but_not_randomization():
    bal = balance_check(_confounded(), "arm", "control", "treatment", ["segment"])
    assert bal.loc[0, "flagged"] and bal.loc[0, "effect_size"] > 0.4
    ab = make_ab_test()
    ok = balance_check(ab, "arm", "control", "treatment", ["device"])
    assert not ok["flagged"].any()
    with pytest.raises(ExperimentError):
        balance_check(ab, "arm", "control", "control", ["device"])


def test_adjustment_removes_confounding_bias():
    df = _confounded()
    adj = adjusted_effect(df, "y", "arm", "control", "treatment", ["segment"])
    assert adj.naive_difference > 5
    assert abs(adj.adjusted_difference) < 0.3
    assert adj.ci_low < adj.adjusted_difference < adj.ci_high
    assert adj.shift_from_adjustment < -5


def test_adjusted_effect_input_validation():
    df = _confounded()
    with pytest.raises(ExperimentError):
        adjusted_effect(df, "segment", "arm", "control", "treatment", [])
    with pytest.raises(ExperimentError):
        adjusted_effect(df, "y", "arm", "control", "treatment", ["nope"])
    with pytest.raises(ExperimentError):
        adjusted_effect(pd.DataFrame(), "y", "arm", "control", "treatment", [])


def test_evidence_tiers():
    assert evidence_tier(True, False, False, False)[0] == EvidenceTier.CAUSAL
    assert evidence_tier(True, True, False, False)[0] == EvidenceTier.DESCRIPTIVE
    assert evidence_tier(True, False, True, True)[0] == EvidenceTier.ADJUSTED
    tier, text = evidence_tier(False, False, False, True)
    assert tier == EvidenceTier.ADJUSTED and "not declared randomized" in text


def test_findings_never_claim_causation_without_the_tier():
    ab = make_ab_test()
    c = ab.loc[ab["arm"] == "control", "converted"]
    t = ab.loc[ab["arm"] == "treatment", "converted"]
    res = two_proportion_test(int(c.sum()), len(c), int(t.sum()), len(t))

    tier, text = evidence_tier(False, False, False, False)
    f = experiment_finding(res, "conversion rate", tier, text, "meaningful_positive", 0.01,
                           data_source="synthetic")
    assert f.category == FindingCategory.EXPERIMENT_RESULT and f.is_simulated
    assert f.evidence_level == EvidenceLevel.STATISTICAL_LEAD
    assert "not established" in f.title and f.alternatives

    tier2, text2 = evidence_tier(True, False, False, False)
    f2 = experiment_finding(res, "conversion rate", tier2, text2, "meaningful_positive", 0.01)
    assert "randomized comparison" in f2.title