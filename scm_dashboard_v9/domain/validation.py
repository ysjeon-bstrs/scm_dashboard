"""
도메인 데이터 검증 로직

이 모듈은 타임라인 빌드 전에 입력 데이터의 유효성을 검증합니다.
Commit 1에서는 기존 v5와 동일하게 Streamlit 호출을 유지하고,
Commit 2에서 예외로 변경하여 Streamlit 의존성을 제거할 예정입니다.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st


def validate_timeline_inputs(
    snapshot: object,
    moves: object,
    start: object,
    end: object,
) -> bool:
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

    Returns:
        True: 모든 검증 통과
        False: 하나 이상의 검증 실패 (Streamlit 에러 메시지 표시됨)

    Notes:
        - 이 함수는 Commit 1에서 기존 v5 로직을 그대로 유지합니다.
        - Commit 2에서 st.error()를 제거하고 예외를 던지도록 변경할 예정입니다.
    """
    # ========================================
    # 1단계: 데이터프레임 타입 검증
    # ========================================
    if not isinstance(snapshot, pd.DataFrame):
        st.error("스냅샷 데이터가 손상되었습니다. 엑셀/시트를 다시 불러와 주세요.")
        return False

    if not isinstance(moves, pd.DataFrame):
        st.error("이동 원장 데이터가 손상되었습니다. 엑셀/시트를 다시 불러와 주세요.")
        return False

    # ========================================
    # 2단계: 스냅샷 필수 컬럼 검증
    # ========================================
    required_snapshot_cols = {"center", "resource_code", "stock_qty"}
    missing_snapshot = [col for col in required_snapshot_cols if col not in snapshot.columns]

    if missing_snapshot:
        st.error(
            "스냅샷 데이터에 필요한 컬럼이 없습니다: "
            + ", ".join(sorted(missing_snapshot))
        )
        return False

    # ========================================
    # 3단계: 이동 원장 필수 컬럼 검증
    # ========================================
    required_move_cols = {"from_center", "to_center", "resource_code"}
    missing_moves = [col for col in required_move_cols if col not in moves.columns]

    if missing_moves:
        st.error(
            "이동 원장 데이터에 필요한 컬럼이 없습니다: " + ", ".join(sorted(missing_moves))
        )
        return False

    # ========================================
    # 4단계: 날짜 타입 검증
    # ========================================
    if not isinstance(start, pd.Timestamp) or not isinstance(end, pd.Timestamp):
        st.error("기간 정보가 손상되었습니다. 기간 슬라이더를 다시 설정해 주세요.")
        return False

    # ========================================
    # 5단계: 날짜 범위 검증
    # ========================================
    if end < start:
        st.error("기간의 종료일이 시작일보다 빠릅니다. 기간을 다시 선택하세요.")
        return False

    return True
