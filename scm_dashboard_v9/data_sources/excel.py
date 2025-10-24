"""
Excel 파일 데이터 로더

이 모듈은 사용자가 업로드한 Excel 파일(.xlsx)에서
스냅샷, 이동 원장, WIP 데이터를 로드합니다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
import streamlit as st

from scm_dashboard_v9.data_sources.loaders import (
    load_from_excel,
    load_wip_from_incoming,
    merge_wip_as_moves,
)

from scm_dashboard_v9.domain.normalization import normalize_moves, normalize_snapshot as normalize_refined_snapshot


@dataclass
class LoadedData:
    """
    로드된 데이터를 담는 컨테이너.

    Attributes:
        moves: 정규화된 이동 원장 데이터프레임 (WIP 포함)
        snapshot: 정규화된 스냅샷 데이터프레임
    """
    moves: pd.DataFrame
    snapshot: pd.DataFrame


def load_from_excel_uploader() -> Optional[LoadedData]:
    """
    Streamlit 파일 업로더를 통해 Excel 파일을 로드하고 정규화합니다.

    업로드된 Excel 파일에서 다음 시트를 읽습니다:
    - 이동 원장 시트
    - 스냅샷 시트
    - 입고 예정 시트 (WIP)

    WIP 데이터가 있으면 이동 원장에 자동으로 병합됩니다.

    Returns:
        LoadedData 인스턴스. 파일이 업로드되지 않았으면 None.

    Examples:
        >>> data = load_from_excel_uploader()
        >>> if data:
        ...     print(f"Loaded {len(data.moves)} moves, {len(data.snapshot)} snapshots")
    """
    # ========================================
    # 1단계: Streamlit 파일 업로더 렌더링
    # ========================================
    file = st.file_uploader("엑셀 업로드 (.xlsx)", type=["xlsx"], key="v5_excel")

    if file is None:
        return None

    # ========================================
    # 2단계: Excel 파일 로드 (v4 로더 사용)
    # ========================================
    df_move, df_ref, df_incoming, _ = load_from_excel(file)

    # ========================================
    # 3단계: 데이터 정규화
    # ========================================
    moves = normalize_moves(df_move)
    snapshot = normalize_refined_snapshot(df_ref)

    # ========================================
    # 4단계: WIP 데이터 병합 (있는 경우)
    # ========================================
    try:
        wip_df = load_wip_from_incoming(df_incoming)
        moves = merge_wip_as_moves(moves, wip_df)

        if wip_df is not None and not wip_df.empty:
            st.success(f"WIP {len(wip_df)}건 반영 완료")

    except Exception as exc:  # pragma: no cover - streamlit feedback
        st.warning(f"WIP 불러오기 실패: {exc}")

    return LoadedData(moves=moves, snapshot=snapshot)
