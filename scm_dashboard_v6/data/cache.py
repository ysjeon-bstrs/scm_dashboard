"""
간단 캐시 (v6)

- 초기: 프로세스 내 메모리 캐시
- 후속: TTL/키 전략/외부 캐시 연계 확장
"""

from __future__ import annotations

from typing import Any, Dict


_STORE: Dict[str, Any] = {}


def get(key: str) -> Any:
    return _STORE.get(key)


def set(key: str, value: Any) -> None:
    _STORE[key] = value


def clear() -> None:
    _STORE.clear()


