"""엑셀 업로드 어댑터."""

from __future__ import annotations

from typing import Optional, Tuple

import pandas as pd

from scm_dashboard_v4.loaders import load_from_excel


def load_moves_snapshot_from_excel(
    file: object,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """업로드된 엑셀에서 move/정제 스냅샷/입고예정/snapshot_raw를 추출한다."""

    # ✅ Streamlit uploader가 제공한 파일 객체를 그대로 전달한다.
    return load_from_excel(file)
