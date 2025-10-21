"""
아마존 섹션 유스케이스 (v6)

- 초기에는 v5 컨텍스트 빌더를 호출하여 동작을 유지하고,
  v6 차트 래퍼로 렌더링한다.
"""

from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd
import streamlit as st

from scm_dashboard_v5.forecast import build_amazon_forecast_context as v5_build_amz_ctx
from scm_dashboard_v6.ui.charts.amazon import render_amazon_sales_vs_inventory
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
    use_consumption_forecast: bool = True,
    lag_days: int | None = None,
    inv_actual: pd.DataFrame | None = None,
    inv_forecast: pd.DataFrame | None = None,
    use_inventory_for_sales: bool = True,
    sales_forecast_from_inventory: Optional[pd.DataFrame] = None,
) -> None:
    """아마존 패널을 렌더링한다 (v6).

    - v5 컨텍스트 빌더를 사용하여 예측 번들을 만들고,
      v6 차트 래퍼로 렌더링한다.
    """

    # v6 안전장치: 센터 별칭 정규화 (스냅샷/이동 모두)
    processed_snapshot = snapshot_long.copy() if snapshot_long is not None else pd.DataFrame()
    if not processed_snapshot.empty and "center" in processed_snapshot.columns:
        try:
            processed_snapshot["center"] = processed_snapshot["center"].apply(normalize_center_value)
        except Exception:
            pass

    # Backfill missing event_date for moves using pred_inbound_date / inbound_date / arrival_date
    processed_moves = moves.copy() if moves is not None else pd.DataFrame()
    if not processed_moves.empty:
        # ... (coalescing logic for event_date, pred_inbound_date, inbound_date, arrival_date)
        cols = {str(c).strip().lower(): c for c in processed_moves.columns}
        col_event = cols.get("event_date")
        col_pred = cols.get("pred_inbound_date")
        col_inb = cols.get("inbound_date")
        col_arr = cols.get("arrival_date")

        if col_event is None:
            processed_moves["event_date"] = pd.NaT
            col_event = "event_date"

        if col_pred and col_pred in processed_moves.columns:
            processed_moves[col_event] = processed_moves[col_event].where(processed_moves[col_event].notna(), processed_moves[col_pred])
        if col_inb and col_inb in processed_moves.columns:
            processed_moves[col_event] = processed_moves[col_event].where(processed_moves[col_event].notna(), processed_moves[col_inb])
        if col_arr and col_arr in processed_moves.columns:
            arr = pd.to_datetime(processed_moves[col_arr], errors="coerce")
            try:
                lag = int(lag_days) if lag_days is not None else 0
            except Exception:
                lag = 0
            today_norm = pd.Timestamp.today().normalize()
            past_mask = arr.notna() & (arr <= today_norm)
            fut_mask = arr.notna() & (arr > today_norm)
            est_from_arr = pd.Series(pd.NaT, index=processed_moves.index, dtype="datetime64[ns]")
            if past_mask.any():
                est_from_arr.loc[past_mask] = (arr.loc[past_mask] + pd.Timedelta(days=lag)).dt.normalize()
            if fut_mask.any():
                est_from_arr.loc[fut_mask] = arr.loc[fut_mask].dt.normalize()
            processed_moves[col_event] = processed_moves[col_event].where(processed_moves[col_event].notna(), est_from_arr)

        if "event_date" in processed_moves.columns:
            processed_moves["event_date"] = pd.to_datetime(processed_moves["event_date"], errors="coerce").dt.normalize()
        for col in ("to_center", "resource_code"):
            if col in processed_moves.columns:
                processed_moves[col] = processed_moves[col].astype(str)
        # 센터 별칭 정규화 (inbound 필터 호환)
        if "to_center" in processed_moves.columns:
            try:
                processed_moves["to_center"] = processed_moves["to_center"].apply(normalize_center_value)
            except Exception:
                pass
        if "qty_ea" in processed_moves.columns:
            processed_moves["qty_ea"] = pd.to_numeric(processed_moves["qty_ea"], errors="coerce").fillna(0)

    ctx = v5_build_amz_ctx(
        snap_long=processed_snapshot,
        moves=processed_moves,
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
            # v6 렌더러에서 일자별 가중치를 재현할 수 있도록 이벤트 자체도 보관
            setattr(ctx, "promotion_events", promotion_events or [])
        except Exception:
            pass
    else:
        try:
            setattr(ctx, "promotion_events", [])
        except Exception:
            pass

    # v5 컨텍스트에도 가공된 moves를 반영 (방어적 재할당)
    try:
        setattr(ctx, "moves", processed_moves)
    except Exception:
        pass

    # 프로모션이 있어도 인벤토리 기반 재고 추세는 유지하되,
    # 판매 예측은 v5 컨텍스트의 uplift 적용 소비예측이 우선하도록 한다.
    extra: dict = {"use_inventory_for_sales": False}
    if inv_actual is not None:
        extra["inv_actual"] = inv_actual
    # 예측 재고 프레임이 선택한 모든 SKU를 포함하지 않으면 v5 내부 추정에 맡긴다
    if inv_forecast is not None and not inv_forecast.empty:
        try:
            present = set(inv_forecast.get("resource_code", pd.Series([], dtype=str)).astype(str).unique())
            expected = set(str(s) for s in skus)
            if expected and expected.issubset(present):
                extra["inv_forecast"] = inv_forecast
        except Exception:
            # 안전하게 내부 추정 사용
            pass

    # v6: 오늘 이후 판매 예측을 inv_forecast 감소량으로 강제 동기화하고 싶을 때 사용
    if sales_forecast_from_inventory is not None and not sales_forecast_from_inventory.empty:
        try:
            # ctx.sales_forecast 형식과 동일 컬럼(date, resource_code, sales_qty)
            forced = sales_forecast_from_inventory.copy()
            forced["date"] = pd.to_datetime(forced["date"], errors="coerce").dt.normalize()
            forced = forced.dropna(subset=["date"]) 
            setattr(ctx, "sales_forecast", forced)
        except Exception:
            pass

    # v5 렌더러를 직접 호출하여 v5와 동일 동작 확보
    v5_render_amazon_sales_vs_inventory(
        ctx,
        inv_actual=extra.get("inv_actual"),
        inv_forecast=extra.get("inv_forecast"),
        use_inventory_for_sales=bool(extra.get("use_inventory_for_sales", True)),
    )

    # 디버그 마커 제거됨



