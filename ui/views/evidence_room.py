from __future__ import annotations

import streamlit as st

from core.contracts import FindingCategory
from core.evidence import EvidenceError
from core.reports import build_html_report, build_markdown_report
from ui import logic, state, theme


def render(ctx: logic.AppContext) -> None:
    store = state.get_store()
    theme.page_header("Evidence Room", "Saved findings, supporting charts and reproducible reports")
    listing = store.list_findings()
    if listing.empty:
        theme.callout("info", "Nothing saved yet. Open an investigation page and use 'Save to Evidence Room' on a finding.")
        return
    theme.kpi_row([("Saved findings", f"{len(listing):,}", "", "teal"),
                   ("From loaded data", f"{int((~listing['is_simulated']).sum()):,}", "", "blue"),
                   ("Simulated", f"{int(listing['is_simulated'].sum()):,}", "teaching examples", "amber")])
    st.dataframe(listing[["id", "created_at", "category", "evidence_level", "data_source", "title"]], hide_index=True)

    ids = listing["id"].tolist()
    fid = st.selectbox("Open a finding", ids, key="ev_pick",
                       format_func=lambda i: f"#{i}  {listing.loc[listing['id'] == i, 'title'].iloc[0][:80]}")
    finding = store.get(fid)
    theme.render_finding(finding, f"ev_{fid}")

    note = st.text_area("Analyst note", value=store.note_of(fid), key=f"ev_note_{fid}")
    b1, b2 = st.columns(2)
    if b1.button("Save note", key=f"ev_savenote_{fid}"):
        store.set_note(fid, note)
        st.success("Note saved.")
    confirm = b2.checkbox("Confirm delete", key=f"ev_confirm_{fid}")
    if b2.button("Delete this finding", disabled=not confirm, key=f"ev_del_{fid}"):
        store.delete(fid)
        st.rerun()

    st.markdown("### Reproduce")
    prep = finding.parameters.get("data_prep")
    if finding.category == FindingCategory.AGGREGATE_VS_SEGMENT and prep:
        if st.button("Re-run from stored parameters", key=f"ev_verify_{fid}"):
            _verify(store, fid, finding, prep)
    elif finding.category == FindingCategory.AGGREGATE_VS_SEGMENT:
        st.caption("This finding was saved without a record of how its frame was built, so it cannot be re-run here.")
    else:
        st.caption("This finding type records its parameters for a manual re-run.")

    st.markdown("### Export")
    chosen = st.multiselect("Findings to include", ids, default=ids, key="ev_export_ids")
    fmt = st.radio("Format", ["Markdown", "HTML (with charts)"], horizontal=True, key="ev_fmt")
    if st.button("Prepare report", key="ev_prepare") and chosen:
        findings = store.get_many(chosen)
        note_text = "Findings saved from Metric Mirage; findings from real and simulated data are listed separately."
        if fmt == "Markdown":
            st.session_state["ev_payload"] = (build_markdown_report(findings, dataset_note=note_text).encode("utf-8"),
                                              "metric_mirage_report.md", "text/markdown")
        else:
            st.session_state["ev_payload"] = (build_html_report(findings, dataset_note=note_text).encode("utf-8"),
                                              "metric_mirage_report.html", "text/html")
    if "ev_payload" in st.session_state:
        data, name, mime = st.session_state["ev_payload"]
        st.download_button(f"Download {name}", data, file_name=name, mime=mime, key="ev_download")


def _verify(store, fid, finding, prep) -> None:
    key = prep.get("dataset")
    info = state.info_for(key)
    if info is None:
        if logic.is_upload_key(key):
            theme.callout("warn", "This finding came from an uploaded file that is not loaded in this session. Upload the "
                                  "same file and prepare it with the same column mapping; its dataset id will then match "
                                  "and the finding can be re-run.", title="Uploaded data not available")
        else:
            st.error(f"Unknown dataset {key!r}.")
        return
    try:
        frame = logic.rebuild_frame(info, state.load_raw(info.key), prep,
                                    finding.parameters["baseline"], finding.parameters["comparison"])
        out = store.verify(fid, frame)
    except (FileNotFoundError, ValueError, EvidenceError, KeyError) as exc:
        st.error(str(exc))
        return
    theme.callout("ok" if out.matched else "warn", out.message,
                  title="Reproduced" if out.matched else "Did not reproduce")
    st.caption(f"Fingerprint of the stored finding: {out.stored_fingerprint}; of the re-run: {out.new_fingerprint}.")
