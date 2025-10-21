"""핵심 타임라인 계산 래퍼 모듈.

v8 패키지에서도 검증된 v5 타임라인 빌더를 그대로 노출해 호출부를 안전하게 이동할 수 있게 한다.
"""

from __future__ import annotations

from scm_dashboard_v5.core import build_timeline as build_timeline

__all__ = ["build_timeline"]
