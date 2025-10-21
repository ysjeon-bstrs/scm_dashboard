"""KPI 포맷팅 및 계산 유틸리티 모듈.

숫자, 날짜, 재고 커버리지 등의 포맷팅 함수를 제공합니다.
"""

from __future__ import annotations

import pandas as pd


def escape(value: object) -> str:
    """HTML 이스케이프 처리를 수행합니다.
    
    Args:
        value: 이스케이프할 값
        
    Returns:
        이스케이프된 문자열
    """
    return str(value or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def format_number(value: float | int | None) -> str:
    """숫자를 천 단위 구분 기호가 있는 문자열로 포맷팅합니다.
    
    Args:
        value: 포맷팅할 숫자
        
    Returns:
        "1,234" 형식의 문자열 또는 "-"
    """
    if value is None:
        return "-"
    if isinstance(value, float) and pd.isna(value):
        return "-"
    try:
        return f"{int(round(float(value))):,}"
    except (TypeError, ValueError):
        return "-"


def value_font_size(value: str, *, base_size: float = 1.25, min_size: float = 0.9) -> str:
    """값의 길이에 따라 적절한 폰트 크기를 계산합니다.
    
    긴 숫자는 작은 폰트로 표시하여 카드 내에 잘 맞도록 합니다.
    
    Args:
        value: 표시할 값 문자열
        base_size: 기본 폰트 크기 (em 단위)
        min_size: 최소 폰트 크기 (em 단위)
        
    Returns:
        "1.25em" 형식의 CSS 폰트 크기
    """
    length = len(str(value))
    if length <= 6:
        return f"{base_size}em"
    elif length <= 9:
        return f"{max(min_size, base_size - 0.15)}em"
    elif length <= 12:
        return f"{max(min_size, base_size - 0.25)}em"
    else:
        return f"{min_size}em"


def format_days(value: float | None) -> str:
    """일수를 "N일" 형식으로 포맷팅합니다.
    
    Args:
        value: 일수
        
    Returns:
        "30일" 형식의 문자열 또는 "-"
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "-"
    try:
        return f"{int(round(float(value)))}일"
    except (TypeError, ValueError):
        return "-"


def format_date(value: pd.Timestamp | None) -> str:
    """날짜를 "YYYY-MM-DD" 형식으로 포맷팅합니다.
    
    Args:
        value: Timestamp 객체
        
    Returns:
        "2024-12-31" 형식의 문자열 또는 "-"
    """
    if value is None or pd.isna(value):
        return "-"
    try:
        return value.strftime("%Y-%m-%d")
    except (AttributeError, ValueError):
        return "-"


def calculate_coverage_days(current_qty: float | int | None, daily_demand: float | int | None) -> float | None:
    """현재 재고량과 일평균 수요로 재고 커버리지 일수를 계산합니다.
    
    Args:
        current_qty: 현재 재고량 (EA)
        daily_demand: 일평균 수요 (EA/day)
        
    Returns:
        재고 커버리지 일수 또는 None
    """
    try:
        qty = float(current_qty or 0)
        demand = float(daily_demand or 0)
        if demand <= 0:
            return None
        return qty / demand
    except (TypeError, ValueError):
        return None


def calculate_sellout_date(today: pd.Timestamp, coverage_days: float | None) -> pd.Timestamp | None:
    """재고 소진 예상 날짜를 계산합니다.
    
    Args:
        today: 기준 날짜
        coverage_days: 재고 커버리지 일수
        
    Returns:
        재고 소진 예상 날짜 또는 None
    """
    if coverage_days is None or coverage_days <= 0:
        return None
    try:
        return today + pd.Timedelta(days=int(round(coverage_days)))
    except (TypeError, ValueError):
        return None


def should_show_in_transit(center: str, in_transit_value: int) -> bool:
    """In-Transit 재고를 표시해야 하는지 판단합니다.
    
    Args:
        center: 센터 이름
        in_transit_value: In-Transit 재고량
        
    Returns:
        표시 여부
    """
    return bool(in_transit_value and in_transit_value > 0)
