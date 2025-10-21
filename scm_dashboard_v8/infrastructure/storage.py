"""세션 캐시에 보관된 snapshot_raw 접근 래퍼."""

from __future__ import annotations

import pandas as pd

from scm_dashboard_v4.loaders import load_snapshot_raw


def load_snapshot_raw_cache() -> pd.DataFrame:
    """Streamlit 세션 캐시에 저장된 snapshot_raw DataFrame을 반환한다."""

    # ✅ v4에서 관리하는 캐시 함수에 그대로 위임한다.
    return load_snapshot_raw()
