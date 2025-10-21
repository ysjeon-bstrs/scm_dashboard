"""v8 모듈 구조에서도 기존 Streamlit 엔트리포인트를 재사용하기 위한 래퍼."""

from __future__ import annotations


def run_app() -> None:
    """기존 v5 Streamlit 엔트리포인트를 호출해 동일한 UI를 실행한다."""

    # ✅ v5_main.main() 호출만 수행해 UI 동작은 기존과 100% 동일하게 유지한다.
    from v5_main import main as _v5_main

    _v5_main()
