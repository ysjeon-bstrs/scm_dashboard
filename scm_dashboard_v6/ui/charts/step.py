"""
v6 스텝 차트 (초기: v5 구현을 래핑)

- 초기 단계에서는 v5 `scm_dashboard_v5.ui.charts.render_step_chart`
  를 그대로 호출하여 동작을 유지한다.
"""

from __future__ import annotations

from typing import Any

from scm_dashboard_v5.ui.charts import render_step_chart as _v5_render


def render_step_chart(*args: Any, **kwargs: Any) -> None:
    """스텝 차트를 렌더링한다 (v5 래핑)."""

    _v5_render(*args, **kwargs)




