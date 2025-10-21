"""
시간/기간 유틸 (v7)

설명(한글):
- 날짜 normalize와 기간 경계 클램프를 제공합니다.
"""

from __future__ import annotations

import pandas as pd


def normalize_date(value) -> pd.Timestamp:
    """입력 값을 timezone-naive Timestamp로 정규화한다."""
    return pd.to_datetime(value, errors="coerce").normalize()


def clamp_range(start: pd.Timestamp, end: pd.Timestamp, *, bound_min: pd.Timestamp, bound_max: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    """주어진 기간을 [bound_min, bound_max] 범위 내로 클램프한다."""
    s = pd.Timestamp(start).normalize()
    e = pd.Timestamp(end).normalize()
    s = max(min(s, bound_max), bound_min)
    e = max(min(e, bound_max), bound_min)
    if e < s:
        e = s
    return s, e


