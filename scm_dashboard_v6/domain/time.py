"""
시간/기간 유틸 (v6)

- 날짜 정규화와 기간 경계 계산을 제공한다.
"""

from __future__ import annotations

import pandas as pd


def normalize_date(value) -> pd.Timestamp:
    return pd.to_datetime(value, errors="coerce").normalize()


def clamp_range(start: pd.Timestamp, end: pd.Timestamp, *, bound_min: pd.Timestamp, bound_max: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    s = pd.Timestamp(start).normalize()
    e = pd.Timestamp(end).normalize()
    s = max(min(s, bound_max), bound_min)
    e = max(min(e, bound_max), bound_min)
    if e < s:
        e = s
    return s, e
