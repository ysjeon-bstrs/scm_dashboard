"""
센터 유틸 (v6)

- v4/v5 별칭 정규화를 v6 경계에서 래핑한다.
"""

from __future__ import annotations

from center_alias import normalize_center_value as _normalize


def normalize_center_value(value: object) -> str | None:
    """센터 명칭을 표준 별칭으로 정규화한다.

    - 유효하지 않으면 None을 반환한다.
    """

    return _normalize(value)


