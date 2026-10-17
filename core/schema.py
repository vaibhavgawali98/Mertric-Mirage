"""Schema contract: maps raw dataset columns to analytical roles and decides
which analyses a dataset can support."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import pandas as pd


class SchemaError(ValueError):
    """Raised when a column mapping is invalid for the given data."""


class Role(str, Enum):
    INVOICE_ID = "invoice_id"
    PRODUCT_ID = "product_id"
    DESCRIPTION = "description"
    QUANTITY = "quantity"
    UNIT_PRICE = "unit_price"
    TIMESTAMP = "timestamp"
    CUSTOMER_ID = "customer_id"
    REVENUE = "revenue"  # provided directly, or derived from quantity * unit_price


class Analysis(str, Enum):
    COMMAND_CENTER = "command_center"
    METRIC_INTEGRITY = "metric_integrity"
    CONTRADICTION = "contradiction"
    COHORT = "cohort"
    EXPERIMENT = "experiment"


@dataclass(frozen=True)
class AnalysisSpec:
    label: str
    required_roles: frozenset[Role]
    needs_segment_column: bool = False


ANALYSES: dict[Analysis, AnalysisSpec] = {
    Analysis.COMMAND_CENTER: AnalysisSpec(
        "Command Center", frozenset({Role.REVENUE})
    ),
    # The three engines below take explicit column choices, so they do not
    # depend on a fixed role (a conversion table has no revenue column).
    Analysis.METRIC_INTEGRITY: AnalysisSpec("Metric Integrity", frozenset()),
    Analysis.CONTRADICTION: AnalysisSpec(
        "Contradiction Lab", frozenset(), needs_segment_column=True
    ),
    Analysis.COHORT: AnalysisSpec(
        "Cohort Observatory",
        frozenset(
            {Role.CUSTOMER_ID, Role.TIMESTAMP, Role.INVOICE_ID, Role.REVENUE}
        ),
    ),
    Analysis.EXPERIMENT: AnalysisSpec(
        "Experiment Lab", frozenset(), needs_segment_column=True
    ),
}


@dataclass(frozen=True)
class Capability:
    enabled: bool
    reason: str  # shown in the UI when disabled


@dataclass
class DatasetSchema:
    """mapping: Role -> raw column name. segment_columns: categorical columns
    usable for grouping (e.g. Country)."""

    mapping: dict[Role, str]
    segment_columns: list[str] = field(default_factory=list)
    name: str = "dataset"
    source: str = "upload"  # "public" | "synthetic" | "upload"

    def available_roles(self) -> set[Role]:
        roles = set(self.mapping)
        # Revenue can be derived when quantity and price both exist.
        if Role.REVENUE not in roles and {Role.QUANTITY, Role.UNIT_PRICE} <= roles:
            roles.add(Role.REVENUE)
        return roles

    def capability(self, analysis: Analysis) -> Capability:
        spec = ANALYSES[analysis]
        missing = spec.required_roles - self.available_roles()
        if missing:
            names = ", ".join(sorted(r.value for r in missing))
            return Capability(False, f"Missing required columns for: {names}.")
        if spec.needs_segment_column and not self.segment_columns:
            return Capability(False, "Needs at least one categorical segment column.")
        return Capability(True, "")

    def validate_against(self, df: pd.DataFrame) -> None:
        """Check that every mapped column actually exists in the data."""
        if df is None or len(df.columns) == 0:
            raise SchemaError("Dataset is empty or has no columns.")
        wanted = list(self.mapping.values()) + list(self.segment_columns)
        missing = [c for c in wanted if c not in df.columns]
        if missing:
            raise SchemaError(f"Columns not found in data: {missing}")


# Mapping for the public UCI Online Retail II file (verify against your copy).
ONLINE_RETAIL_II = DatasetSchema(
    mapping={
        Role.INVOICE_ID: "Invoice",
        Role.PRODUCT_ID: "StockCode",
        Role.DESCRIPTION: "Description",
        Role.QUANTITY: "Quantity",
        Role.TIMESTAMP: "InvoiceDate",
        Role.UNIT_PRICE: "Price",
        Role.CUSTOMER_ID: "Customer ID",
    },
    segment_columns=["Country"],
    name="UCI Online Retail II",
    source="public",
)