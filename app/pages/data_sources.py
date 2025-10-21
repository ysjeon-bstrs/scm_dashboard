from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import streamlit as st

from app import state
from scm_dashboard_v8.application.loaders import (
    DashboardLoadResult,
    LoadError,
    load_dashboard_from_excel,
    load_dashboard_from_gsheet,
)


@dataclass
class DataSourceOutcome:
    data: state.LoadedData
    source_display: str


def _to_loaded_data(result: DashboardLoadResult) -> state.LoadedData:
    return state.LoadedData(
        moves=result.moves,
        snapshot=result.snapshot,
        snapshot_raw=result.snapshot_raw,
        wip=result.wip,
    )


def _load_from_excel_uploader() -> Optional[state.LoadedData]:
    """Return normalized data loaded from an uploaded Excel file."""

    file = st.file_uploader("엑셀 업로드 (.xlsx)", type=["xlsx"], key="v5_excel")
    if file is None:
        return None

    try:
        result = load_dashboard_from_excel(file)
    except LoadError as exc:
        st.error(str(exc))
        return None

    if result.wip_warning:
        st.warning(result.wip_warning)
    elif result.wip is not None and not result.wip.empty:
        st.success(f"WIP {len(result.wip)}건 반영 완료")

    return _to_loaded_data(result)


def _load_from_gsheet(*, show_spinner_message: str) -> Optional[state.LoadedData]:
    """Return normalized data retrieved from Google Sheets."""

    try:
        with st.spinner(show_spinner_message):
            result = load_dashboard_from_gsheet()
    except LoadError as exc:
        st.error(str(exc))
        return None

    if result.moves.empty or result.snapshot.empty:
        st.error("Google Sheets에서 데이터를 불러올 수 없습니다. 권한을 확인해주세요.")
        return None

    if result.wip_warning:
        st.warning(result.wip_warning)
    elif result.wip is not None and not result.wip.empty:
        st.success(f"WIP {len(result.wip)}건 반영 완료")

    st.success("Google Sheets 데이터가 업데이트되었습니다.")
    return _to_loaded_data(result)


def ensure_data() -> Optional[DataSourceOutcome]:
    """Load data via the available tabs and persist it in the session state."""

    data = state.get_loaded_data()
    source_label = state.get_source_label()

    st.markdown("### 데이터 소스")
    st.caption("대시보드 진입 시 Google Sheets 데이터를 자동으로 불러옵니다.")

    source_display = {
        "gsheet": "Google Sheets",
        "excel": "엑셀 업로드",
    }.get(source_label, "없음")

    source_caption = st.empty()

    refresh_clicked = st.button("Google Sheets 데이터 새로 고침", key="v5_gsheet_refresh")

    should_load_gsheet = data is None or refresh_clicked
    if should_load_gsheet:
        spinner_msg = (
            "Google Sheets 데이터 불러오는 중..."
            if data is None
            else "Google Sheets 데이터를 새로 불러오는 중..."
        )
        gsheet_data = _load_from_gsheet(show_spinner_message=spinner_msg)
        if gsheet_data is not None:
            state.set_loaded_data(gsheet_data, "gsheet")
            data = gsheet_data
            source_display = "Google Sheets"
        elif data is None:
            source_display = "없음"

    with st.expander("엑셀 파일 업로드 (선택)", expanded=False):
        st.caption("필요할 때만 수동으로 엑셀 파일을 업로드하여 데이터를 교체할 수 있습니다.")
        excel_data = _load_from_excel_uploader()
        if excel_data is not None:
            state.set_loaded_data(excel_data, "excel")
            st.success("엑셀 데이터가 로드되었습니다.")
            data = excel_data
            source_display = "엑셀 업로드"

    source_caption.caption(f"현재 데이터 소스: **{source_display}**")

    if data is None:
        return None

    return DataSourceOutcome(data=data, source_display=source_display)
