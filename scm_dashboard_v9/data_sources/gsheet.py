"""
Google Sheets 데이터 로더

이 모듈은 Google Sheets API를 통해
스냅샷, 이동 원장, WIP 데이터를 로드합니다.
"""

from __future__ import annotations

import logging
from typing import Optional

import streamlit as st

from scm_dashboard_v9.common.performance import measure_time_context

logger = logging.getLogger(__name__)

from scm_dashboard_v9.data_sources.loaders import (
    load_from_gsheet_api,
    load_wip_from_incoming,
    merge_wip_as_moves,
    normalize_tk_stock_distrib,
)
from scm_dashboard_v9.domain.normalization import (
    normalize_moves,
)
from scm_dashboard_v9.domain.normalization import (
    normalize_snapshot as normalize_refined_snapshot,
)

from .excel import LoadedData


def load_from_gsheet(*, show_spinner_message: str) -> Optional[LoadedData]:
    """
    Google Sheets API를 통해 데이터를 로드하고 정규화합니다.

    Google Sheets에서 다음 시트를 읽습니다:
    - 이동 원장 시트
    - 스냅샷 시트
    - 입고 예정 시트 (WIP)

    WIP 데이터가 있으면 이동 원장에 자동으로 병합됩니다.

    Args:
        show_spinner_message: 로딩 중 표시할 스피너 메시지

    Returns:
        LoadedData 인스턴스. 로드 실패 시 None.

    Examples:
        >>> data = load_from_gsheet(show_spinner_message="데이터 불러오는 중...")
        >>> if data:
        ...     print(f"Loaded {len(data.moves)} moves")
    """
    # ========================================
    # 1단계: Google Sheets API 호출 (스피너 표시)
    # ========================================
    logger.info("Loading data from Google Sheets")
    try:
        with st.spinner(show_spinner_message):
            with measure_time_context("Google Sheets API fetch"):
                df_move, df_ref, df_incoming, df_tk_stock = load_from_gsheet_api()
                logger.debug(
                    f"Raw data loaded: {len(df_move)} moves, {len(df_ref)} snapshots, "
                    f"{len(df_incoming)} incoming, {len(df_tk_stock)} tk_stock"
                )

    except Exception as exc:  # pragma: no cover - streamlit feedback
        logger.error(f"Failed to load from Google Sheets: {exc}", exc_info=True)
        st.error(f"Google Sheets 데이터를 불러오는 중 오류가 발생했습니다: {exc}")
        return None

    # ========================================
    # 2단계: 데이터 유효성 검증
    # ========================================
    if df_move.empty or df_ref.empty:
        logger.error("Google Sheets data is empty")
        st.error("Google Sheets에서 데이터를 불러올 수 없습니다. 권한을 확인해주세요.")
        return None

    # ========================================
    # 3단계: 데이터 정규화
    # ========================================
    logger.info("Normalizing snapshot and moves data")

    with measure_time_context("Data normalization"):
        moves = normalize_moves(df_move)
        snapshot = normalize_refined_snapshot(df_ref)
        logger.debug(f"Normalized: {len(moves)} moves, {len(snapshot)} snapshots")

    # ========================================
    # 4단계: WIP 데이터 병합 (있는 경우)
    # ========================================
    try:
        wip_df = load_wip_from_incoming(df_incoming)
        moves = merge_wip_as_moves(moves, wip_df)

        if wip_df is not None and not wip_df.empty:
            logger.info(f"WIP data merged: {len(wip_df)} rows")

    except Exception as exc:  # pragma: no cover - streamlit feedback
        logger.warning(f"Failed to load WIP data: {exc}", exc_info=True)
        st.warning(f"WIP 불러오기 실패: {exc}")

    # ========================================
    # 5단계: 로드 완료 로깅
    # ========================================
    logger.info("Google Sheets data loaded successfully")

    # 태광KR 가상창고 배분 시트를 정규화하여 대시보드에서 사용할 수 있도록 저장
    tk_stock_distrib = normalize_tk_stock_distrib(df_tk_stock)

    return LoadedData(
        moves=moves,
        snapshot=snapshot,
        tk_stock_distrib=tk_stock_distrib,
    )
