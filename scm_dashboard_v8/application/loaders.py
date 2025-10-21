"""데이터 소스 접근을 애플리케이션 레이어에서 감싸는 래퍼."""

from __future__ import annotations

from typing import Optional, Tuple

import pandas as pd

from scm_dashboard_v4.loaders import (
    load_from_excel as _load_from_excel,
    load_from_gsheet_api as _load_from_gsheet_api,
    load_snapshot_raw as _load_snapshot_raw,
)


def load_from_gsheet() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Google Sheets에서 move/정제 스냅샷/입고예정 데이터를 읽어온다.

    Returns:
        v4 구현과 동일한 순서의 DataFrame 튜플(move, snapshot_refined, incoming).
    """

    # ✅ 기존 v4 함수 호출만 수행해 동작을 100% 동일하게 유지한다.
    return _load_from_gsheet_api()


def load_from_excel_uploader(
    file: object,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """사용자가 업로드한 엑셀 파일에서 move·정제 스냅샷·입고예정·snapshot_raw를 추출한다."""

    # ✅ Streamlit uploader가 제공하는 파일 객체를 그대로 위임한다.
    return _load_from_excel(file)


def load_snapshot_raw() -> pd.DataFrame:
    """Streamlit 세션 캐시에 저장된 snapshot_raw DataFrame을 반환한다."""

    # ✅ 캐시 조회 및 fallback 로직은 v4 구현에 위임한다.
    return _load_snapshot_raw()
