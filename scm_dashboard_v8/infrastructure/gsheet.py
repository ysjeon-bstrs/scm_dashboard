"""Google Sheets 연동 어댑터."""

from __future__ import annotations

from typing import Tuple

import pandas as pd

from scm_dashboard_v4.loaders import load_from_gsheet_api


def load_moves_snapshot_from_gsheet() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Google Sheets API를 호출해 move·정제 스냅샷·입고예정 데이터를 읽어온다."""

    # ✅ v4 모듈의 검증된 로직을 그대로 호출한다.
    return load_from_gsheet_api()
