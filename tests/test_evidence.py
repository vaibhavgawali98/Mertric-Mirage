import json

import numpy as np
import pandas as pd
import pytest

from core.contracts import EvidenceLevel, Finding, FindingCategory
from core.contradiction import MetricSpec, investigate_contradiction
from core.evidence import (
    EvidenceError, FindingStore, apply_filters, canonical_fingerprint, finding_from_payload,
    finding_to_payload, reproduce_contradiction, sanitize,
)
from core.integrity import duplicate_finding, duplicate_summary
from core.reports import DISCLAIMER, build_html_report, build_markdown_report, write_report
from data.synthetic import SIMPSON_SPEC, make_simpson
from viz.charts import build_figure, decomposition_spec

RATE = MetricSpec("conversion rate", "converted")


def _counts_df(spec):
    parts = []
    for (period, seg), (n, conv) in spec.items():
        converted = np.r_[np.ones(conv), np.zeros(n - conv)]
        parts.append(pd.DataFrame({"period": period, "segment": seg, "converted": converted}))
    return pd.concat(parts, ignore_index=True)


def _simpson_finding(df=None, **kw):
    df = make_simpson() if df is None else df
    res = investigate_contradiction(df, RATE, "period", "Q1", "Q2", "segment", data_source="synthetic", **kw)
    f = res.to_finding()
    f.chart_spec = decomposition_spec(res)
    return f


@pytest.fixture
def store(tmp_path):
    s = FindingStore(tmp_path / "evidence.db")
    yield s
    s.close()


def test_sanitize_makes_strict_json():
    out = sanitize({"a": np.float64("nan"), "b": (1, 2), "c": np.int64(3), "d": pd.Timestamp("2020-01-02"),
                    "e": float("inf"), 5: np.bool_(True), "f": FindingCategory.DATA_QUALITY})
    assert out == {"a": None, "b": [1, 2], "c": 3, "d": "2020-01-02T00:00:00", "e": None,
                   "5": True, "f": "data_quality"}
    json.dumps(out, allow_nan=False)


def test_payload_round_trip_preserves_the_finding():
    f = _simpson_finding()
    g = finding_from_payload(finding_to_payload(f))
    assert (g.title, g.category, g.evidence_level, g.created_at) == (f.title, f.category, f.evidence_level, f.created_at)
    assert g.parameters["segment_col"] == "segment" and g.is_simulated
    assert canonical_fingerprint(f) == canonical_fingerprint(g)
    with pytest.raises(EvidenceError):
        finding_from_payload({"title": "x"})


def test_store_saves_once_and_reloads_with_its_chart(store):
    f = _simpson_finding()
    fid, is_new = store.save(f, note="teaching example")
    assert is_new and store.count() == 1
    assert store.save(f) == (fid, False) and store.count() == 1
    loaded = store.get(fid)
    assert loaded.title == f.title and store.note_of(fid) == "teaching example"
    assert build_figure(loaded.chart_spec).data[0].type == "waterfall"
    listing = store.list_findings()
    assert listing.loc[0, "is_simulated"] and listing.loc[0, "category"] == "aggregate_vs_segment"
    store.set_note(fid, "changed")
    assert store.note_of(fid) == "changed"
    store.delete(fid)
    assert store.count() == 0
    with pytest.raises(EvidenceError):
        store.get(fid)


def test_verify_reproduces_the_stored_numbers(store):
    fid, _ = store.save(_simpson_finding())
    out = store.verify(fid, make_simpson())
    assert out.matched and out.exact and out.new_finding is not None


def test_verify_detects_changed_or_broken_data(store):
    fid, _ = store.save(_simpson_finding())
    changed = store.verify(fid, make_simpson().iloc[:-300])
    assert not changed.matched and changed.message
    broken = store.verify(fid, make_simpson().drop(columns=["converted"]))
    assert not broken.matched and "failed" in broken.message


def test_filters_are_part_of_the_reproducible_record(store):
    df = _counts_df(SIMPSON_SPEC)
    df["region"] = np.where(np.arange(len(df)) % 2 == 0, "north", "south")
    filters = {"region": ["north"]}
    res = investigate_contradiction(apply_filters(df, filters), RATE, "period", "Q1", "Q2", "segment",
                                    filters=filters)
    assert res.verdict == "simpson_reversal"
    fid, _ = store.save(res.to_finding())
    assert store.get(fid).parameters["filters"] == filters
    out = store.verify(fid, df)       # unfiltered data: the stored filters are re-applied
    assert out.matched and out.exact


def test_apply_filters_validation():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "x"]})
    assert len(apply_filters(df, {"b": ["x"]})) == 2
    assert len(apply_filters(df, {"a": 2})) == 1
    assert len(apply_filters(df, None)) == 3
    with pytest.raises(EvidenceError):
        apply_filters(df, {"nope": [1]})


def test_other_finding_types_are_not_auto_reproduced(store):
    df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
    f = duplicate_finding(duplicate_summary(df), data_source="public")
    fid, _ = store.save(f)
    with pytest.raises(EvidenceError, match="manual"):
        store.verify(fid, df)


def test_reports_separate_simulated_findings_and_carry_the_disclaimer():
    real = duplicate_finding(duplicate_summary(pd.DataFrame({"a": [1, 1, 2]})), data_source="public")
    sim = _simpson_finding()
    md = build_markdown_report([sim, real])
    assert DISCLAIMER in md and "SIMULATED" in md
    assert md.index("Findings from the loaded data") < md.index("Simulated teaching findings")
    assert "fingerprint" in md and "```json" in md
    page = build_html_report([sim, real])
    assert "SIMULATED" in page and "plotly-graph-div" in page and DISCLAIMER in page
    assert "No findings" in build_html_report([]) and "No findings" in build_markdown_report([])


def test_html_report_escapes_text_from_the_data():
    f = Finding(title="<script>alert(1)</script>", category=FindingCategory.DATA_QUALITY,
                evidence_level=EvidenceLevel.DESCRIPTIVE, metric="m<b>", comparison="c", evidence={"x": 1},
                method="count", n=1, alternatives=["<img src=x onerror=alert(1)>"], next_step="look",
                parameters={"k": "<v>"})
    page = build_html_report([f])
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page
    assert "<script>alert(1)" not in page and "<img src=x" not in page


def test_write_report_picks_format_from_the_extension(tmp_path):
    f = _simpson_finding()
    assert write_report([f], tmp_path / "r.md").read_text(encoding="utf-8").startswith("# ")
    assert write_report([f], tmp_path / "sub" / "r.html").read_text(encoding="utf-8").startswith("<!doctype html>")
    with pytest.raises(ValueError):
        write_report([f], tmp_path / "r.pdf")


def test_data_prep_record_survives_reproduction(store):
    f = _simpson_finding()
    f.parameters["data_prep"] = {"dataset": "simpson", "freq": None, "matched_window": False}
    fid, _ = store.save(f)
    out = store.verify(fid, make_simpson())
    assert out.matched and out.exact