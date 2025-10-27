"""
세션 상태 관리

이 모듈은 Streamlit 세션 상태를 사용하여
로드된 데이터를 관리하고 지속합니다.
"""

from __future__ import annotations

from typing import Optional

import streamlit as st

from .excel import LoadedData, load_from_excel_uploader
from .gsheet import load_from_gsheet


def ensure_data() -> Optional[LoadedData]:
    """
    사용 가능한 탭을 통해 데이터를 로드하고 세션 상태에 저장합니다.

    이 함수는 다음 우선순위로 데이터를 로드합니다:
    1. 세션에 이미 로드된 데이터가 있으면 재사용
    2. "새로 고침" 버튼이 클릭되면 Google Sheets에서 다시 로드
    3. 엑셀 파일이 업로드되면 엑셀 데이터로 교체

    기본적으로 대시보드 진입 시 Google Sheets 데이터를
    자동으로 로드합니다.

    Returns:
        LoadedData 인스턴스. 데이터가 없으면 None.

    Session State Keys:
        - v5_data: LoadedData 인스턴스
        - _v5_source: 데이터 소스 ("gsheet" | "excel")
    """
    # ========================================
    # 1단계: 세션 상태에서 기존 데이터 확인
    # ========================================
    data: Optional[LoadedData] = st.session_state.get("v5_data")

    # ========================================
    # 2단계: 데이터 소스 UI 렌더링
    # ========================================
    st.markdown("### 데이터 소스")
    st.caption("대시보드 진입 시 Google Sheets 데이터를 자동으로 불러옵니다.")

    # 현재 소스 표시
    source_label = st.session_state.get("_v5_source")
    source_display = {
        "gsheet": "Google Sheets",
        "excel": "엑셀 업로드",
    }.get(source_label, "없음")

    source_caption = st.empty()

    # ========================================
    # 3단계: Google Sheets 새로 고침
    # ========================================
    refresh_clicked = st.button(
        "Google Sheets 데이터 새로 고침", key="v5_gsheet_refresh"
    )

    # 데이터가 없거나 새로 고침 버튼이 클릭된 경우 Google Sheets에서 로드
    should_load_gsheet = data is None or refresh_clicked

    if should_load_gsheet:
        spinner_msg = (
            "Google Sheets 데이터 불러오는 중..."
            if data is None
            else "Google Sheets 데이터를 새로 불러오는 중..."
        )

        gsheet_data = load_from_gsheet(show_spinner_message=spinner_msg)

        if gsheet_data is not None:
            st.session_state["_v5_source"] = "gsheet"
            st.session_state["v5_data"] = gsheet_data
            data = gsheet_data
            source_display = "Google Sheets"
        elif data is None:
            source_display = "없음"

    # ========================================
    # 4단계: 엑셀 파일 업로드 (선택적)
    # ========================================
    with st.expander("엑셀 파일 업로드 (선택)", expanded=False):
        st.caption(
            "필요할 때만 수동으로 엑셀 파일을 업로드하여 데이터를 교체할 수 있습니다."
        )

        excel_data = load_from_excel_uploader()

        if excel_data is not None:
            st.session_state["_v5_source"] = "excel"
            st.session_state["v5_data"] = excel_data
            st.success("엑셀 데이터가 로드되었습니다.")
            data = excel_data
            source_display = "엑셀 업로드"

    # ========================================
    # 5단계: 현재 소스 표시
    # ========================================
    source_caption.caption(f"현재 데이터 소스: **{source_display}**")

    # ========================================
    # 6단계: 데이터 반환
    # ========================================
    if data is None:
        return None

    return data
