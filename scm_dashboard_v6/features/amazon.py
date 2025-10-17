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
    use_inventory_for_sales: bool = True,
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

    # v5 컨텍스트는 내부에서 promotion_events의 'uplift'를 읽어 멀티플라이어를 구성한다.
    # 별도 힌트는 불필요하나, 디버그용으로 첫 이벤트의 상승률을 남겨둔다.
    if promotion_events:
        try:
            first = promotion_events[0]
            uplift = float(first.get("uplift", 0.0))
            setattr(ctx, "promotion_uplift_debug", uplift)
        except Exception:
            pass

    # v5 컨텍스트가 입고예정(inbound)을 반영해 판매 예측을 클램프하도록 실제 moves를 전달
    try:
        mv = moves.copy() if moves is not None else pd.DataFrame()
        if not mv.empty:
            if "event_date" in mv.columns:
                mv["event_date"] = pd.to_datetime(mv["event_date"], errors="coerce").dt.normalize()
            for col in ("to_center", "resource_code"):
                if col in mv.columns:
                    mv[col] = mv[col].astype(str)
            if "qty_ea" in mv.columns:
                mv["qty_ea"] = pd.to_numeric(mv["qty_ea"], errors="coerce").fillna(0)
        setattr(ctx, "moves", mv)
    except Exception:
        pass

    # 프로모션이 있어도 인벤토리 기반 재고 추세는 유지하되,
    # 판매 예측은 v5 컨텍스트의 uplift 적용 소비예측이 우선하도록 한다.
    extra: dict = {"use_inventory_for_sales": False}
    if inv_actual is not None:
        extra["inv_actual"] = inv_actual
    if inv_forecast is not None:
        extra["inv_forecast"] = inv_forecast

    render_amazon_sales_vs_inventory(ctx, **extra)



