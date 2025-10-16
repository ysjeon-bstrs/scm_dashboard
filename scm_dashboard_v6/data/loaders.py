"""
데이터 로더 (v6)

- 초기에는 v5 로더를 호출하여 동작을 유지한다.
- 후속 단계에서 캐싱/설정/시크릿을 본 모듈로 통합한다.
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
