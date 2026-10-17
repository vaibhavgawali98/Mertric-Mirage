"""METRIC MIRAGE: the business metric truth engine. Run with: streamlit run app.py"""
import streamlit as st

from ui import logic, state, theme
from ui.views import (
    cohorts, command_center, contradiction, evidence_room, experiments, integrity, upload,
)

UPLOAD_PAGE = "Upload Data"
PAGES = {
    "Command Center": command_center.render,
    "Upload Data": upload.render,
    "Metric Integrity": integrity.render,
    "Contradiction Lab": contradiction.render,
    "Cohort Observatory": cohorts.render,
    "Experiment Lab": experiments.render,
    "Evidence Room": evidence_room.render,
}


def _apply_pending_navigation(infos: dict) -> None:
    """Pages cannot change a widget's value after it is drawn, so they leave a note here and
    the change is applied before the sidebar widgets are created."""
    pending_dataset = st.session_state.pop("_pending_dataset", None)
    if pending_dataset in infos:
        st.session_state["dataset_key"] = pending_dataset
    pending_page = st.session_state.pop("_pending_page", None)
    if pending_page in PAGES:
        st.session_state["page"] = pending_page
    if st.session_state.get("dataset_key") not in infos:   # e.g. an upload that was removed
        st.session_state["dataset_key"] = logic.default_dataset_key()
    if st.session_state.get("page") not in PAGES:
        st.session_state["page"] = "Command Center"


def main() -> None:
    st.set_page_config(page_title="Metric Mirage", page_icon="◐", layout="wide", initial_sidebar_state="expanded")
    theme.inject_css()
    infos = state.all_infos()
    _apply_pending_navigation(infos)
    with st.sidebar:
        st.markdown('<div class="mm-brand">METRIC MIRAGE</div><div class="mm-tag">The business metric truth engine</div>',
                    unsafe_allow_html=True)
        key = st.selectbox("Dataset", list(infos), key="dataset_key", format_func=lambda k: infos[k].label)
        page = st.radio("Investigation", list(PAGES), key="page")
        st.caption(theme.DISCLAIMER)

    if page == UPLOAD_PAGE:          # usable even when the built-in data is missing
        PAGES[page]()
        return
    info = infos[key]
    try:
        raw = state.load_raw(key)
    except FileNotFoundError:
        theme.callout("error", "The prepared retail data is missing. Run `python -m data.prepare_retail`, then "
                               "`python -m data.build_sqlite`, choose a simulated dataset in the sidebar, or upload "
                               "your own file on the Upload Data page.", title="Data not found")
        st.stop()
    PAGES[page](logic.AppContext(info, raw))


main()
