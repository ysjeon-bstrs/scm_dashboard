"""
데이터 로더 (v7)

설명(한글):
- v5/v4 로더를 래핑하여 DataFrame을 반환합니다.
- 캐싱/시크릿/설정은 추후 통합 가능.
"""

from __future__ import annotations

from typing import Tuple
import pandas as pd

from scm_dashboard_v4.loaders import load_from_excel as v4_load_excel
from scm_dashboard_v4.loaders import load_from_gsheet_api as v4_load_gsheet
from scm_dashboard_v4.loaders import load_snapshot_raw as v4_load_snapshot_raw


def load_excel(file_obj) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, object]:
    """엑셀에서 (moves, snapshot_refined, incoming, extras) 튜플을 로드한다."""
    return v4_load_excel(file_obj)


def load_gsheet() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Google Sheets에서 (moves, snapshot_refined, incoming) 튜플을 로드한다."""
    return v4_load_gsheet()


def load_snapshot_raw() -> pd.DataFrame:
    """스냅샷 raw 테이블을 로드한다."""
    return v4_load_snapshot_raw()


