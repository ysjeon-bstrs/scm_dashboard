"""
도메인 데이터 검증 로직

이 모듈은 타임라인 빌드 전에 입력 데이터의 유효성을 검증합니다.
Streamlit 의존성이 제거되어, 순수한 도메인 로직으로 동작합니다.
"""

from __future__ import annotations

import logging

import pandas as pd

from .exceptions import ValidationError

logger = logging.getLogger(__name__)


def validate_timeline_inputs(
    snapshot: object,
    moves: object,
    start: object,
    end: object,
) -> None:
    """
    타임라인 빌드에 필요한 입력 데이터의 구조적 정합성을 검증합니다.

    검증 항목:
    1. snapshot과 moves가 DataFrame인지 확인
    2. 필수 컬럼 존재 여부 확인
       - snapshot: center, resource_code, stock_qty
       - moves: from_center, to_center, resource_code
    3. start와 end가 Timestamp인지 확인
    4. 날짜 범위가 유효한지 확인 (end >= start)

    Args:
        snapshot: 스냅샷 데이터 (pd.DataFrame이어야 함)
        moves: 이동 원장 데이터 (pd.DataFrame이어야 함)
        start: 시작 날짜 (pd.Timestamp여야 함)
        end: 종료 날짜 (pd.Timestamp여야 함)

    Raises:
        ValidationError: 검증 실패 시 발생

    Examples:
        >>> validate_timeline_inputs(snapshot_df, moves_df, start_date, end_date)
        # 검증 통과 (반환값 없음)

        >>> validate_timeline_inputs(None, moves_df, start_date, end_date)
        ValidationError: 스냅샷 데이터가 손상되었습니다...
    """
    # ========================================
    # 1단계: 데이터프레임 타입 검증
    # ========================================
    logger.debug("Validating timeline inputs")

    if not isinstance(snapshot, pd.DataFrame):
        logger.error("Snapshot is not a DataFrame")
        raise ValidationError(
            "스냅샷 데이터가 손상되었습니다. 엑셀/시트를 다시 불러와 주세요."
        )

    if not isinstance(moves, pd.DataFrame):
        logger.error("Moves is not a DataFrame")
        raise ValidationError(
            "이동 원장 데이터가 손상되었습니다. 엑셀/시트를 다시 불러와 주세요."
        )

    # ========================================
    # 2단계: 스냅샷 필수 컬럼 검증
    # ========================================
    required_snapshot_cols = {"center", "resource_code", "stock_qty"}
    missing_snapshot = [
        col for col in required_snapshot_cols if col not in snapshot.columns
    ]

    if missing_snapshot:
        logger.error(f"Missing snapshot columns: {missing_snapshot}")
        raise ValidationError(
            "스냅샷 데이터에 필요한 컬럼이 없습니다: "
            + ", ".join(sorted(missing_snapshot))
        )

    # ========================================
    # 3단계: 이동 원장 필수 컬럼 검증
    # ========================================
    required_move_cols = {"from_center", "to_center", "resource_code"}
    missing_moves = [col for col in required_move_cols if col not in moves.columns]

    if missing_moves:
        logger.error(f"Missing move columns: {missing_moves}")
        raise ValidationError(
            "이동 원장 데이터에 필요한 컬럼이 없습니다: "
            + ", ".join(sorted(missing_moves))
        )

    # ========================================
    # 4단계: 날짜 타입 검증
    # ========================================
    if not isinstance(start, pd.Timestamp) or not isinstance(end, pd.Timestamp):
        logger.error(f"Invalid date types: start={type(start)}, end={type(end)}")
        raise ValidationError(
            "기간 정보가 손상되었습니다. 기간 슬라이더를 다시 설정해 주세요."
        )

    # ========================================
    # 5단계: 날짜 범위 검증
    # ========================================
    if end < start:
        logger.error(f"Invalid date range: start={start}, end={end}")
        raise ValidationError(
            "기간의 종료일이 시작일보다 빠릅니다. 기간을 다시 선택하세요."
        )

    logger.debug("Timeline inputs validation passed")
