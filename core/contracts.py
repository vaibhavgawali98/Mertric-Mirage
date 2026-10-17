"""Core data contract for investigation findings."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class FindingCategory(str, Enum):
    DATA_QUALITY = "data_quality"
    AGGREGATE_VS_SEGMENT = "aggregate_vs_segment"
    DISTRIBUTION_ANOMALY = "distribution_anomaly"
    COHORT_BEHAVIOR = "cohort_behavior"
    EXPERIMENT_RESULT = "experiment_result"


class EvidenceLevel(str, Enum):
    DESCRIPTIVE = "descriptive"            # a computed fact, no inference
    STATISTICAL_LEAD = "statistical_lead"  # pattern worth investigating
    CONFIRMED = "confirmed"                # verified, e.g. exact duplicate rows


@dataclass
class Finding:
    title: str
    category: FindingCategory
    evidence_level: EvidenceLevel
    metric: str
    comparison: str                      # what was compared with what
    evidence: dict[str, Any]             # exact numbers
    method: str                          # calculation / test used
    n: int                               # sample size behind the finding
    alternatives: list[str]              # alternative explanations (required)
    next_step: str
    parameters: dict[str, Any]           # filters + params for reproduction
    data_source: str = "upload"          # "public" | "synthetic" | "upload"
    affected: str = ""                   # segment or time period
    uncertainty: dict[str, Any] | None = None   # e.g. CI bounds
    limitations: list[str] = field(default_factory=list)
    chart_spec: dict[str, Any] | None = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    def __post_init__(self) -> None:
        if not self.title.strip():
            raise ValueError("A finding needs a title.")
        if self.n < 0:
            raise ValueError("Sample size cannot be negative.")
        if not self.alternatives:
            raise ValueError("A finding must list at least one alternative explanation.")

    @property
    def is_simulated(self) -> bool:
        return self.data_source == "synthetic"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["category"] = self.category.value
        d["evidence_level"] = self.evidence_level.value
        d["is_simulated"] = self.is_simulated
        return d

    def fingerprint(self) -> str:
        """Hash of the reproducible parts (not the timestamp)."""
        payload = json.dumps(
            {
                "category": self.category.value,
                "metric": self.metric,
                "comparison": self.comparison,
                "method": self.method,
                "parameters": self.parameters,
                "evidence": self.evidence,
            },
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]