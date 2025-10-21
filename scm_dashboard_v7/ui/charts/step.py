"""
v7 스텝 차트 래퍼

설명(한글):
- v5의 스텝 차트 렌더 함수를 그대로 호출하여 시각화 동등성을 보장합니다.
"""

from __future__ import annotations

from typing import Any

from scm_dashboard_v5.ui.charts import render_step_chart as _v5_render


def render_step_chart(*args: Any, **kwargs: Any) -> None:
    """스텝 차트를 렌더링한다 (v7: v5 래핑)."""
    _v5_render(*args, **kwargs)


