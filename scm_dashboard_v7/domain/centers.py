"""
센터 유틸 (v7)

설명(한글):
- v4/v5 별칭 정규화를 v7 경계에서 래핑합니다.
"""

from __future__ import annotations

from center_alias import normalize_center_value as _normalize


def normalize_center_value(value: object) -> str | None:
    """센터 명칭을 표준 별칭으로 정규화한다. 유효하지 않으면 None 반환."""
    return _normalize(value)


