"""
도메인 계층 퍼블릭 API

이 모듈은 도메인 계층의 주요 클래스와 함수를 재수출하여
일관된 퍼블릭 API를 제공합니다.
"""
from __future__ import annotations

from .exceptions import (
    DataLoadError,
    DomainError,
    FilterError,
    TimelineError,
    ValidationError,
)
from .filters import (
    calculate_date_bounds,
    calculate_move_date_bounds,
    extract_center_and_sku_options,
    norm_center,
)
from .models import MoveTable, SnapshotTable, TimelineBundle
from .normalization import normalize_dates, normalize_moves, normalize_snapshot
from .validation import validate_timeline_inputs

__all__ = [
    # 예외
    "DomainError",
    "ValidationError",
    "DataLoadError",
    "FilterError",
    "TimelineError",
    # 모델
    "SnapshotTable",
    "MoveTable",
    "TimelineBundle",
    # 정규화
    "normalize_dates",
    "normalize_moves",
    "normalize_snapshot",
    # 검증
    "validate_timeline_inputs",
    # 필터
    "norm_center",
    "extract_center_and_sku_options",
    "calculate_move_date_bounds",
    "calculate_date_bounds",
]
