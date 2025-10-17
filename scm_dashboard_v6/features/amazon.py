"""
아마존 섹션 유스케이스 (v6)

- 초기에는 v5 컨텍스트 빌더를 호출하여 동작을 유지하고,
  v6 차트 래퍼로 렌더링한다.
"""

from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd

from scm_dashboard_v5.forecast import build_amazon_forecast_context as v5_build_amz_ctx
from scm_dashboard_v6.ui.charts.amazon import render_amazon_sales_vs_inventory


def render_amazon_panel(
    *,
    snapshot_long: pd.DataFrame,
    moves: pd.DataFrame,
    snapshot_raw: Optional[pd.DataFrame],
    centers: Iterable[str],
    skus: Iterable[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    today: pd.Timestamp,
    lookback_days: int,
    promotion_events: Optional[list[dict]] = None,
    use_consumption_forecast: bool = True,
    inv_actual: pd.DataFrame | None = None,
    inv_forecast: pd.DataFrame | None = None,
) -> None:
    """아마존 패널을 렌더링한다 (v6).

    - v5 컨텍스트 빌더를 사용하여 예측 번들을 만들고,
      v6 차트 래퍼로 렌더링한다.
    """

    ctx = v5_build_amz_ctx(
        snap_long=snapshot_long,
        moves=moves,
        snapshot_raw=snapshot_raw,
        centers=list(centers),
        skus=list(skus),
        start=start,
        end=end,
        today=today,
        lookback_days=int(lookback_days),
        promotion_events=promotion_events or [],
        use_consumption_forecast=bool(use_consumption_forecast),
    )

    # v5 차트 내부에서 moves_df의 event_date 가공을 기대하는 경로가 있어
    # 클라우드 환경 스키마 차이로 오류가 발생하는 경우를 회피: moves 사용 비활성화
    try:
        setattr(ctx, "moves", pd.DataFrame())
    except Exception:
        pass

    extra: dict = {}
    if inv_actual is not None:
        extra["inv_actual"] = inv_actual
    if inv_forecast is not None:
        extra["inv_forecast"] = inv_forecast

    render_amazon_sales_vs_inventory(ctx, **extra)



