"""
데이터 정규화 유틸리티

이 모듈은 SCM 대시보드에서 사용하는 데이터의 타입과 형식을
표준화하는 함수들을 제공합니다. 모든 날짜는 자정(00:00:00)으로
정규화되며, 문자열/숫자 컬럼은 일관된 타입으로 변환됩니다.
"""
from __future__ import annotations

from typing import Iterable

import pandas as pd

# 기본적으로 정규화할 날짜 컬럼 목록
DATE_COLUMNS = ("onboard_date", "arrival_date", "inbound_date", "event_date")


def normalize_dates(
    frame: pd.DataFrame,
    *,
    columns: Iterable[str] = DATE_COLUMNS
) -> pd.DataFrame:
    """
    지정된 날짜 컬럼들을 datetime64 타입으로 변환하고 자정으로 정규화합니다.

    날짜 정규화는 시간 부분을 00:00:00으로 만들어 일 단위 집계를
    정확하게 수행할 수 있도록 합니다.

    Args:
        frame: 원본 데이터프레임
        columns: 정규화할 날짜 컬럼명 목록 (기본값: DATE_COLUMNS)

    Returns:
        날짜 컬럼이 정규화된 데이터프레임 복사본.
        원본은 변경되지 않습니다.

    Examples:
        >>> df = pd.DataFrame({
        ...     "onboard_date": ["2024-01-01 14:30", "2024-01-02 09:15"],
        ...     "qty": [100, 200]
        ... })
        >>> normalized = normalize_dates(df)
        >>> normalized["onboard_date"]
        0   2024-01-01 00:00:00
        1   2024-01-02 00:00:00
    """
    out = frame.copy()

    # ========================================
    # 각 날짜 컬럼을 순회하며 정규화
    # ========================================
    for col in columns:
        if col in out.columns:
            # datetime64로 변환 (변환 실패 시 NaT)
            # dt.normalize()로 시간 부분을 00:00:00으로 설정
            out[col] = pd.to_datetime(out[col], errors="coerce").dt.normalize()

    return out


def normalize_moves(frame: pd.DataFrame) -> pd.DataFrame:
    """
    이동 원장 데이터의 핵심 컬럼들을 예측 가능한 타입으로 변환합니다.

    정규화되는 컬럼:
    - carrier_mode: 운송 방식 (문자열, 대문자 변환)
    - resource_code: SKU 코드 (문자열)
    - from_center: 출발 센터 (문자열)
    - to_center: 목적지 센터 (문자열)
    - qty_ea: 수량 (숫자, NaN은 0으로 대체)
    - 날짜 컬럼들: datetime64로 정규화

    Args:
        frame: 원본 이동 원장 데이터프레임

    Returns:
        정규화된 이동 원장 데이터프레임

    Examples:
        >>> raw_moves = pd.read_csv("moves.csv")
        >>> normalized_moves = normalize_moves(raw_moves)
    """
    # ========================================
    # 1단계: 날짜 컬럼 정규화
    # ========================================
    out = normalize_dates(frame)

    # ========================================
    # 2단계: carrier_mode (운송 방식) 정규화
    # ========================================
    # 컬럼이 없으면 빈 문자열로 채운 시리즈 생성
    carrier_src = out["carrier_mode"] if "carrier_mode" in out.columns else pd.Series("", index=out.index)
    # 문자열로 변환하고 대문자로 통일 (예: "sea" -> "SEA")
    out["carrier_mode"] = carrier_src.astype(str).str.upper()

    # ========================================
    # 3단계: resource_code (SKU) 정규화
    # ========================================
    resource_src = out["resource_code"] if "resource_code" in out.columns else pd.Series("", index=out.index)
    out["resource_code"] = resource_src.astype(str)

    # ========================================
    # 4단계: from_center (출발 센터) 정규화
    # ========================================
    from_src = out["from_center"] if "from_center" in out.columns else pd.Series("", index=out.index)
    out["from_center"] = from_src.astype(str)

    # ========================================
    # 5단계: to_center (목적지 센터) 정규화
    # ========================================
    to_src = out["to_center"] if "to_center" in out.columns else pd.Series("", index=out.index)
    out["to_center"] = to_src.astype(str)

    # ========================================
    # 6단계: qty_ea (수량) 정규화
    # ========================================
    qty_src = out["qty_ea"] if "qty_ea" in out.columns else pd.Series(0, index=out.index)
    # 숫자로 변환 (변환 실패 시 NaN -> 0으로 대체)
    out["qty_ea"] = pd.to_numeric(qty_src, errors="coerce").fillna(0)

    return out


def normalize_snapshot(frame: pd.DataFrame) -> pd.DataFrame:
    """
    스냅샷 데이터의 스키마를 표준화하여 하류 소비자가 사용할 수 있도록 합니다.

    표준 스키마:
    - date: 스냅샷 날짜 (datetime64, 자정으로 정규화)
    - center: 센터명 (문자열)
    - resource_code: SKU 코드 (문자열)
    - stock_qty: 재고 수량 (숫자)

    Args:
        frame: 원본 스냅샷 데이터프레임

    Returns:
        표준 스키마로 정규화된 스냅샷 데이터프레임.
        date, center, resource_code, stock_qty 컬럼만 포함.

    Raises:
        KeyError: 'date' 또는 'snapshot_date' 컬럼이 없을 경우

    Examples:
        >>> raw_snapshot = pd.read_csv("snapshot.csv")
        >>> normalized = normalize_snapshot(raw_snapshot)
        >>> normalized.columns.tolist()
        ['date', 'center', 'resource_code', 'stock_qty']
    """
    out = frame.copy()

    # ========================================
    # 1단계: 날짜 컬럼 찾기
    # ========================================
    # 'date' 또는 'snapshot_date' 컬럼 중 존재하는 것을 사용
    date_col = None
    for candidate in ("date", "snapshot_date"):
        if candidate in out.columns:
            date_col = candidate
            break

    if not date_col:
        raise KeyError("snapshot frame must include a 'date' or 'snapshot_date' column")

    # ========================================
    # 2단계: 날짜 정규화
    # ========================================
    out["date"] = pd.to_datetime(out[date_col], errors="coerce").dt.normalize()

    # ========================================
    # 3단계: center (센터명) 정규화
    # ========================================
    center_src = out["center"] if "center" in out.columns else pd.Series("", index=out.index)
    out["center"] = center_src.astype(str)

    # ========================================
    # 4단계: resource_code (SKU) 정규화
    # ========================================
    resource_src = out["resource_code"] if "resource_code" in out.columns else pd.Series("", index=out.index)
    out["resource_code"] = resource_src.astype(str)

    # ========================================
    # 5단계: stock_qty (재고 수량) 정규화
    # ========================================
    stock_src = out["stock_qty"] if "stock_qty" in out.columns else pd.Series(0, index=out.index)
    out["stock_qty"] = pd.to_numeric(stock_src, errors="coerce")

    # ========================================
    # 6단계: 유효하지 않은 행 제거 및 스키마 표준화
    # ========================================
    # 날짜가 NaT인 행은 제거 (필수 필드)
    out = out.dropna(subset=["date"])

    # 표준 컬럼만 반환
    return out[["date", "center", "resource_code", "stock_qty"]]
