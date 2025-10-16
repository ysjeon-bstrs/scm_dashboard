"""
SCM Dashboard v6 — Streamlit 엔트리 (초기 스캐폴드)

- v5를 유지한 상태로, v6 구조를 점진 도입하기 위한 진입점이다.
- 실제 데이터 로딩/컨트롤/렌더링은 후속 커밋에서 연결한다.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st


def main() -> None:
    st.set_page_config(page_title="SCM Dashboard v6", layout="wide")
    st.title("SCM Dashboard v6 (스캐폴드)")
    st.caption("v6 구조 도입 준비 중 — Step 2/3에서 실제 기능 연결 예정")

    st.info("이 페이지는 v6 구조 검증용 자리 표시자입니다. v5는 그대로 사용 가능합니다.")


if __name__ == "__main__":
    main()
