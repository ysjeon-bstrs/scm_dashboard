"""공통 데이터 처리 유틸리티 함수 모듈.

여러 모듈에서 반복되는 데이터 처리 패턴을 제공합니다.
"""

from __future__ import annotations

import pandas as pd

# 빈 DataFrame 템플릿 상수
EMPTY_INVENTORY_COLUMNS = ["date", "center", "resource_code", "stock_qty"]
EMPTY_SALES_COLUMNS = ["date", "center", "resource_code", "sales_ea"]


def empty_inventory_frame() -> pd.DataFrame:
    """빈 재고 DataFrame을 반환합니다.

    Returns:
        빈 재고 DataFrame
    """
    return pd.DataFrame(columns=EMPTY_INVENTORY_COLUMNS)


def empty_sales_frame() -> pd.DataFrame:
    """빈 판매 DataFrame을 반환합니다.

    Returns:
        빈 판매 DataFrame
    """
    return pd.DataFrame(columns=EMPTY_SALES_COLUMNS)


def safe_normalize_dates(
    df: pd.DataFrame,
    date_column: str = "date",
    *,
    drop_invalid: bool = True,
) -> pd.DataFrame:
    """DataFrame의 날짜 컬럼을 안전하게 정규화합니다.

    Args:
        df: 처리할 DataFrame
        date_column: 날짜 컬럼명
        drop_invalid: 유효하지 않은 날짜를 제거할지 여부

    Returns:
        날짜가 정규화된 DataFrame
    """
    if df.empty or date_column not in df.columns:
        return df

    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column], errors="coerce").dt.normalize()

    if drop_invalid:
        df = df.dropna(subset=[date_column])

    return df


def filter_date_range(
    df: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    date_column: str = "date",
) -> pd.DataFrame:
    """DataFrame을 날짜 범위로 필터링합니다.

    Args:
        df: 필터링할 DataFrame
        start: 시작 날짜
        end: 종료 날짜
        date_column: 날짜 컬럼명

    Returns:
        필터링된 DataFrame
    """
    if df.empty or date_column not in df.columns:
        return df

    start_norm = pd.to_datetime(start).normalize()
    end_norm = pd.to_datetime(end).normalize()

    return df[(df[date_column] >= start_norm) & (df[date_column] <= end_norm)].copy()
