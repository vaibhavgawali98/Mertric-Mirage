import pandas as pd
import pytest

from core.contracts import EvidenceLevel, Finding, FindingCategory
from core.schema import (
    ONLINE_RETAIL_II,
    Analysis,
    DatasetSchema,
    Role,
    SchemaError,
)


def test_retail_schema_enables_all_analyses():
    for analysis in Analysis:
        assert ONLINE_RETAIL_II.capability(analysis).enabled


def test_revenue_is_derived_from_quantity_and_price():
    assert Role.REVENUE in ONLINE_RETAIL_II.available_roles()


def test_cohort_disabled_without_customer_id():
    schema = DatasetSchema(
        mapping={
            Role.INVOICE_ID: "inv",
            Role.QUANTITY: "q",
            Role.UNIT_PRICE: "p",
            Role.TIMESTAMP: "t",
        },
        segment_columns=["region"],
    )
    cap = schema.capability(Analysis.COHORT)
    assert not cap.enabled
    assert "customer_id" in cap.reason


def test_contradiction_needs_a_segment_column():
    schema = DatasetSchema(
        mapping={Role.REVENUE: "rev", Role.TIMESTAMP: "t"}, segment_columns=[]
    )
    assert not schema.capability(Analysis.CONTRADICTION).enabled


def test_validate_rejects_missing_columns():
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(SchemaError):
        ONLINE_RETAIL_II.validate_against(df)


def test_validate_rejects_empty_frame():
    with pytest.raises(SchemaError):
        ONLINE_RETAIL_II.validate_against(pd.DataFrame())


def _finding(**overrides):
    base = dict(
        title="t",
        category=FindingCategory.DATA_QUALITY,
        evidence_level=EvidenceLevel.DESCRIPTIVE,
        metric="m",
        comparison="c",
        evidence={"x": 1},
        method="count",
        n=10,
        alternatives=["something else"],
        next_step="look closer",
        parameters={"p": 1},
    )
    base.update(overrides)
    return Finding(**base)


def test_finding_requires_alternatives():
    with pytest.raises(ValueError):
        _finding(alternatives=[])


def test_finding_fingerprint_is_stable_and_ignores_timestamp():
    a, b = _finding(), _finding()
    a.created_at, b.created_at = "2020-01-01", "2030-01-01"
    assert a.fingerprint() == b.fingerprint()


def test_synthetic_findings_are_flagged_simulated():
    assert _finding(data_source="synthetic").is_simulated
    assert not _finding(data_source="public").is_simulated