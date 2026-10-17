"""Evidence store and reproducibility checks (Engine E, persistence side).

A finding is stored as strict JSON (NaN becomes null, numpy types become Python types).
Automatic re-running is implemented for aggregate-versus-segment findings, whose
parameters (metric, periods, segment column, thresholds, filters) fully define them.
Other finding types carry their parameters for a manual re-run.
"""
from __future__ import annotations

import dataclasses
import json
import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.contracts import EvidenceLevel, Finding, FindingCategory
from core.contradiction import ContradictionError, MetricSpec, investigate_contradiction

DEFAULT_EVIDENCE_DB = Path("data/processed/evidence.db")


class EvidenceError(ValueError):
    """Raised for invalid evidence-store operations."""


# ------------------------------------------------------------------- JSON safety
def sanitize(obj: Any) -> Any:
    """Nested results to strict-JSON-safe Python types (NaN and inf become None)."""
    if obj is None or obj is pd.NaT or obj is pd.NA:
        return None
    if isinstance(obj, Enum):
        return sanitize(obj.value)
    if isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, datetime):        # includes pd.Timestamp
        return obj.isoformat()
    if isinstance(obj, dict):
        return {str(k): sanitize(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        return [sanitize(v) for v in obj.tolist()]
    if isinstance(obj, (list, tuple, pd.Series)):
        return [sanitize(v) for v in obj]
    return str(obj)


def finding_to_payload(finding: Finding) -> dict[str, Any]:
    return sanitize(finding.to_dict())


def finding_from_payload(payload: dict[str, Any]) -> Finding:
    allowed = {f.name for f in dataclasses.fields(Finding)}
    data = {k: v for k, v in payload.items() if k in allowed}
    try:
        data["category"] = FindingCategory(data["category"])
        data["evidence_level"] = EvidenceLevel(data["evidence_level"])
        return Finding(**data)
    except (KeyError, TypeError, ValueError) as exc:
        raise EvidenceError(f"Stored finding is malformed: {exc}") from exc


def canonical_fingerprint(finding: Finding) -> str:
    """Fingerprint of the finding as it looks after a save/load round trip, so a live
    finding and its stored copy share one fingerprint."""
    return finding_from_payload(finding_to_payload(finding)).fingerprint()


# ------------------------------------------------------------------- filters
def apply_filters(df: pd.DataFrame, filters: dict[str, Any] | None) -> pd.DataFrame:
    """Keep rows whose column value is in the allowed list, for every filter column.
    These filters are what a finding records, so a re-run can apply the same ones."""
    out = df
    for col, allowed in (filters or {}).items():
        if col not in out.columns:
            raise EvidenceError(f"Filter column {col!r} not found.")
        values = list(allowed) if isinstance(allowed, (list, tuple, set)) else [allowed]
        out = out[out[col].isin(values)]
    return out


# ------------------------------------------------------------- reproduction
@dataclass
class ReproductionResult:
    matched: bool                 # same title and numerically equal evidence
    exact: bool                   # canonical fingerprints identical
    stored_fingerprint: str
    new_fingerprint: str | None
    message: str
    new_finding: Finding | None


def _numbers_close(a: Any, b: Any) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_numbers_close(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_numbers_close(x, y) for x, y in zip(a, b))
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)
    return a == b


_REQUIRED_PARAMS = ("metric", "numerator", "period_col", "baseline", "comparison", "segment_col",
                    "min_segment_n", "min_coverage", "alpha", "flat_rel_tol")


def reproduce_contradiction(finding: Finding, df: pd.DataFrame,
                            stored_fingerprint: str | None = None) -> ReproductionResult:
    """Re-run an aggregate-versus-segment finding from its recorded parameters (filters included)."""
    if finding.category != FindingCategory.AGGREGATE_VS_SEGMENT:
        raise EvidenceError(
            "Automatic reproduction is implemented for aggregate-versus-segment findings only; "
            "other findings carry their parameters for a manual re-run."
        )
    p = finding.parameters
    missing = [k for k in _REQUIRED_PARAMS if k not in p]
    if missing:
        raise EvidenceError(f"The finding's parameters are incomplete (missing {missing}).")
    expected = stored_fingerprint or canonical_fingerprint(finding)
    try:
        data = apply_filters(df, p.get("filters"))
        res = investigate_contradiction(
            data, MetricSpec(p["metric"], p["numerator"], p.get("denominator")),
            p["period_col"], p["baseline"], p["comparison"], p["segment_col"],
            min_segment_n=p["min_segment_n"], min_coverage=p["min_coverage"], alpha=p["alpha"],
            flat_rel_tol=p["flat_rel_tol"], filters=p.get("filters") or {},
            data_source=p.get("data_source", finding.data_source),
        )
    except (ContradictionError, EvidenceError) as exc:
        return ReproductionResult(False, False, expected, None, f"Re-running the analysis failed: {exc}", None)
    new = res.to_finding()
    if new is None:
        return ReproductionResult(False, False, expected, None,
                                  f"Re-running now gives the verdict '{res.verdict}', which is not a reportable finding.", None)
    if "data_prep" in p:                      # how the frame was built; part of the record
        new.parameters["data_prep"] = p["data_prep"]
    new_fp = canonical_fingerprint(new)
    matched = new.title == finding.title and _numbers_close(sanitize(new.evidence), sanitize(finding.evidence))
    message = ("Re-run reproduces the stored numbers." if matched else
               "Re-run gives different numbers: the data, the filters or the code changed since this finding was saved.")
    return ReproductionResult(matched, new_fp == expected, expected, new_fp, message, new)


# ---------------------------------------------------------------------- store
class FindingStore:
    """SQLite-backed Evidence Room storage. Saving an identical analysis twice keeps one copy."""

    def __init__(self, path: str | Path = DEFAULT_EVIDENCE_DB) -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS findings ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, fingerprint TEXT NOT NULL UNIQUE, "
            "created_at TEXT NOT NULL, category TEXT NOT NULL, evidence_level TEXT NOT NULL, "
            "data_source TEXT NOT NULL, title TEXT NOT NULL, note TEXT NOT NULL DEFAULT '', "
            "payload TEXT NOT NULL)"
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def save(self, finding: Finding, note: str = "") -> tuple[int, bool]:
        """Returns (id, is_new). An identical analysis is not stored twice."""
        payload = finding_to_payload(finding)
        fingerprint = finding_from_payload(payload).fingerprint()
        cur = self._conn.execute(
            "INSERT OR IGNORE INTO findings (fingerprint, created_at, category, evidence_level, "
            "data_source, title, note, payload) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (fingerprint, payload["created_at"], payload["category"], payload["evidence_level"],
             payload["data_source"], payload["title"], note, json.dumps(payload, allow_nan=False)),
        )
        self._conn.commit()
        if cur.rowcount == 1:
            return int(cur.lastrowid), True
        row = self._conn.execute("SELECT id FROM findings WHERE fingerprint = ?", (fingerprint,)).fetchone()
        return int(row[0]), False

    def _row(self, finding_id: int) -> tuple:
        row = self._conn.execute(
            "SELECT payload, fingerprint, note FROM findings WHERE id = ?", (int(finding_id),)
        ).fetchone()
        if row is None:
            raise EvidenceError(f"No finding with id {finding_id}.")
        return row

    def get(self, finding_id: int) -> Finding:
        return finding_from_payload(json.loads(self._row(finding_id)[0]))

    def get_many(self, ids: list[int]) -> list[Finding]:
        return [self.get(i) for i in ids]

    def fingerprint_of(self, finding_id: int) -> str:
        return self._row(finding_id)[1]

    def note_of(self, finding_id: int) -> str:
        return self._row(finding_id)[2]

    def set_note(self, finding_id: int, note: str) -> None:
        self._row(finding_id)
        self._conn.execute("UPDATE findings SET note = ? WHERE id = ?", (note, int(finding_id)))
        self._conn.commit()

    def delete(self, finding_id: int) -> None:
        self._row(finding_id)
        self._conn.execute("DELETE FROM findings WHERE id = ?", (int(finding_id),))
        self._conn.commit()

    def count(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM findings").fetchone()[0])

    def list_findings(self) -> pd.DataFrame:
        df = pd.read_sql_query(
            "SELECT id, created_at, category, evidence_level, data_source, title, note "
            "FROM findings ORDER BY id", self._conn)
        df["is_simulated"] = df["data_source"] == "synthetic"
        return df

    def verify(self, finding_id: int, df: pd.DataFrame) -> ReproductionResult:
        """Re-run a stored finding on `df` (pass the same unfiltered data it came from)."""
        return reproduce_contradiction(self.get(finding_id), df, stored_fingerprint=self.fingerprint_of(finding_id))