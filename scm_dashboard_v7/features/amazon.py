"""
아마존 섹션 유스케이스 (v7)

설명(한글):
- v5 컨텍스트 빌더/렌더러를 래핑하여 동일 동작을 보장합니다.
- 필요 시 v7 차트로 전환 가능하나, 초기에는 v5 렌더를 그대로 호출합니다.
"""

from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd

from scm_dashboard_v5.forecast import build_amazon_forecast_context as v5_build_amz_ctx
from scm_dashboard_v5.ui.charts import (
    render_amazon_sales_vs_inventory as v5_render_amazon_sales_vs_inventory,
)
from center_alias import normalize_center_value


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
    lag_days: int | None = None,
    inv_actual: pd.DataFrame | None = None,
    inv_forecast: pd.DataFrame | None = None,
    use_inventory_for_sales: bool = True,
    sales_forecast_from_inventory: Optional[pd.DataFrame] = None,
) -> None:
    """
    아마존 패널 렌더링 (v7)

    - v5 컨텍스트를 구성한 뒤 동일 렌더를 호출하여 동작을 보장합니다.
    - 필요 시 inv_actual/inv_forecast를 주입해 라인 표시를 강제할 수 있습니다.
    """

    processed_snapshot = snapshot_long.copy() if snapshot_long is not None else pd.DataFrame()
    if not processed_snapshot.empty and "center" in processed_snapshot.columns:
        try:
            processed_snapshot["center"] = processed_snapshot["center"].apply(normalize_center_value)
        except Exception:
            pass

    ctx = v5_build_amz_ctx(
        snap_long=processed_snapshot,
        moves=moves.copy() if moves is not None else pd.DataFrame(),
        snapshot_raw=snapshot_raw,
        centers=list(centers),
        skus=list(skus),
        start=start,
        end=end,
        today=today,
        lookback_days=int(lookback_days),
        promotion_events=promotion_events or [],
        use_consumption_forecast=True,
    )

    # 판매 예측을 재고 궤적 기반으로 강제 주입하고 싶은 경우(옵션)
    if sales_forecast_from_inventory is not None and not sales_forecast_from_inventory.empty:
        try:
            forced = sales_forecast_from_inventory.copy()
            forced["date"] = pd.to_datetime(forced["date"], errors="coerce").dt.normalize()
            forced = forced.dropna(subset=["date"])  # 날짜 정합 보장
            setattr(ctx, "sales_forecast", forced)
        except Exception:
            pass

    # v5 렌더러 호출(동일 동작)
    v5_render_amazon_sales_vs_inventory(
        ctx,
        inv_actual=inv_actual,
        inv_forecast=inv_forecast,
        use_inventory_for_sales=bool(use_inventory_for_sales),
    )


