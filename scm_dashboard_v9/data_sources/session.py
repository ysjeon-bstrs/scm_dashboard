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
    # Google Sheets 새로 고침 버튼은 사이드바로 이동됨
    refresh_clicked = st.session_state.get("_trigger_refresh", False)
    if refresh_clicked:
        st.session_state["_trigger_refresh"] = False

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

    # ========================================
    # 4단계: 데이터 반환
    # ========================================
    if data is None:
        return None

    return data
