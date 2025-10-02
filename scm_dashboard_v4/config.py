"""Streamlit configuration and constants for the SCM dashboard."""

from __future__ import annotations

import streamlit as st

GSHEET_ID = "1RYjKW2UDJ2kWJLAqQH26eqx2-r9Xb0_qE_hfwu9WIj8"

CENTER_COL = {
    "태광KR": "stock2",
    "AMZUS": "fba_available_stock",
    "품고KR": "poomgo_v2_available_stock",
    "SBSPH": "shopee_ph_available_stock",
    "SBSSG": "shopee_sg_available_stock",
    "SBSMY": "shopee_my_available_stock",
    "AcrossBUS": "acrossb_available_stock",
    "어크로스비US": "acrossb_available_stock",
}

PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    "#c49c94", "#f7b6d3", "#c7c7c7", "#dbdb8d", "#9edae5",
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
    "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac",
]


def configure_page() -> None:
    """Apply the global Streamlit page configuration."""
    st.set_page_config(page_title="글로벌 대시보드 — v4", layout="wide")
    st.title("📦 SCM 재고 흐름 대시보드 — v4")


def initialize_session_state() -> None:
    """Ensure that commonly used session state keys exist."""
    if "_data_source" not in st.session_state:
        st.session_state["_data_source"] = None
    if "_snapshot_raw_cache" not in st.session_state:
        st.session_state["_snapshot_raw_cache"] = None
