"""
v6 아마존 판매/재고 차트 (초기: v5 구현을 래핑)

- 초기 단계에서는 v5 `scm_dashboard_v5.ui.charts.render_amazon_sales_vs_inventory`
  를 그대로 호출하여 동작을 유지한다.
- 차트 옵션/스타일/데이터 전처리는 점진적으로 분리/이전한다.
"""

from __future__ import annotations

from typing import Any

from scm_dashboard_v5.ui.charts import render_amazon_sales_vs_inventory as _v5_render


def render_amazon_sales_vs_inventory(ctx: Any, **kwargs: Any) -> None:
    """아마존 판매/재고 차트를 렌더링한다 (v5 래핑).

    ctx: v5의 AmazonForecastContext 혹은 동등한 v6 컨텍스트
    kwargs: 향후 v6 옵션을 전달하기 위한 확장 포인트
    """

    _v5_render(ctx, **kwargs)


