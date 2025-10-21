"""Streamlit 세션 상태 관련 헬퍼."""

from __future__ import annotations

import streamlit as st


def ensure_session_defaults(**defaults) -> None:
    """주어진 기본값을 Streamlit session_state에 설정한다."""

    # ✅ session_state를 직접 수정하던 로직을 한 곳에서 관리한다.
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
