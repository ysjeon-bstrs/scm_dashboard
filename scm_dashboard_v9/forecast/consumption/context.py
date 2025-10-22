"""Amazon Forecast Context 빌더.

전체 예측 컨텍스트를 생성하는 복잡한 로직을 제공합니다.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
import streamlit as st

from center_alias import normalize_center_value
from scm_dashboard_v9.core import build_timeline

from .models import AmazonForecastContext
from .sales import load_amazon_daily_sales_from_snapshot_raw, make_forecast_sales_capped
from .inventory import forecast_sales_and_inventory
from .estimation import apply_consumption_with_events
from .context_helpers import (
    calculate_promotion_multiplier,
    normalize_inputs,
    calculate_latest_snapshot,
    process_inventory_data,
    process_sales_history,
    build_inbound_lookup,
    calculate_promotion_uplift,
    generate_sales_forecasts,
    apply_stock_depletion,
)

def build_amazon_forecast_context(
    *,
    snap_long: pd.DataFrame,
    moves: pd.DataFrame,
    snapshot_raw: pd.DataFrame | None,
    centers: Sequence[str],
    skus: Sequence[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    today: pd.Timestamp,
    lookback_days: int = 28,
    promotion_events: Optional[Iterable[dict]] = None,
    use_consumption_forecast: bool = True,
) -> AmazonForecastContext:
    """Return a fully-populated Amazon forecast bundle for charts and KPIs."""

    empty_inv = pd.DataFrame(columns=["date", "center", "resource_code", "stock_qty"])
    empty_sales = pd.DataFrame(columns=["date", "center", "resource_code", "sales_ea"])
    snapshot_raw_df = snapshot_raw.copy() if snapshot_raw is not None else pd.DataFrame()
    snapshot_long_df = snap_long.copy()
    moves_df = moves.copy() if moves is not None else pd.DataFrame()

    promo_multiplier = calculate_promotion_multiplier(promotion_events)
    center_list, sku_list, start_norm, end_norm, today_norm = normalize_inputs(
        centers, skus, start, end, today
    )

    if not center_list or not sku_list or end_norm < start_norm:
        return AmazonForecastContext(
            start=start_norm,
            end=end_norm,
            today=today_norm,
            centers=center_list,
            skus=sku_list,
            inv_actual=empty_inv.copy(),
            inv_forecast=empty_inv.copy(),
            sales_hist=empty_sales.copy(),
            sales_ma7=empty_sales.copy(),
            sales_forecast=empty_sales.copy(),
            snapshot_raw=snapshot_raw_df.copy(),
            snapshot_long=snapshot_long_df.copy(),
            moves=moves_df.copy(),
            lookback_days=int(lookback_days),
            promotion_multiplier=float(promo_multiplier),
        )

    latest_snapshot = calculate_latest_snapshot(snap_long, today_norm)

    horizon_days = int(max(0, (end_norm - latest_snapshot).days))

    timeline = build_timeline(
        snap_long,
        moves,
        centers=center_list,
        skus=sku_list,
        start=start_norm,
        end=end_norm,
        today=today_norm,
        horizon_days=horizon_days,
    )

    if timeline.empty:
        return AmazonForecastContext(
            start=start_norm,
            end=end_norm,
            today=today_norm,
            centers=center_list,
            skus=sku_list,
            inv_actual=empty_inv.copy(),
            inv_forecast=empty_inv.copy(),
            sales_hist=empty_sales.copy(),
            sales_ma7=empty_sales.copy(),
            sales_forecast=empty_sales.copy(),
            snapshot_raw=snapshot_raw_df.copy(),
            snapshot_long=snapshot_long_df.copy(),
            moves=moves_df.copy(),
            lookback_days=int(lookback_days),
            promotion_multiplier=float(promo_multiplier),
        )

    inv_actual, timeline = process_inventory_data(timeline, today_norm, center_list)

    latest_stock_lookup: dict[tuple[str, str], float] = {}
    if not inv_actual.empty:
        latest_stock_lookup = (
            inv_actual.sort_values("date")
            .groupby(["center", "resource_code"])["stock_qty"]
            .last()
            .astype(float)
            .to_dict()
        )

    inv_projected = empty_inv.copy()
    if use_consumption_forecast:
        projected = apply_consumption_with_events(
            timeline,
            snap_long,
            centers=center_list,
            skus=sku_list,
            start=start_norm,
            end=end_norm,
            lookback_days=lookback_days,
            events=promotion_events or [],
        )
        if projected is not None and not projected.empty:
            inv_projected = projected.copy()

    inv_projected = inv_projected[inv_projected["center"].isin(center_list)].copy()
    inv_projected["date"] = pd.to_datetime(inv_projected["date"], errors="coerce").dt.normalize()
    inv_projected["stock_qty"] = pd.to_numeric(
        inv_projected.get("stock_qty"), errors="coerce"
    ).fillna(0)

    inv_forecast = inv_projected[inv_projected["date"] >= today_norm].copy()

    inv_actual["stock_qty"] = inv_actual["stock_qty"].round().clip(lower=0).astype(int)
    inv_forecast["stock_qty"] = inv_forecast["stock_qty"].round().clip(lower=0).astype(int)

    sales_source = snapshot_raw_df.copy()
    sales_hist = load_amazon_daily_sales_from_snapshot_raw(
        sales_source,
        centers=tuple(center_list),
        skus=sku_list,
    )
    if not sales_hist.empty:
        sales_hist = sales_hist[(sales_hist["date"] >= start_norm) & (sales_hist["date"] <= today_norm)]
    sales_hist = process_sales_history(sales_hist, center_list, sku_list)

    future_index = (
        pd.date_range(today_norm + pd.Timedelta(days=1), end_norm, freq="D")
        if today_norm < end_norm and use_consumption_forecast
        else pd.DatetimeIndex([], dtype="datetime64[ns]")
    )

    inbound_lookup = build_inbound_lookup(moves_df, future_index, center_list, sku_list)

    uplift = calculate_promotion_uplift(promotion_events, future_index)

    sales_ma7, sales_forecast = generate_sales_forecasts(
        sales_hist=sales_hist,
        start_norm=start_norm,
        today_norm=today_norm,
        future_index=future_index,
        uplift=uplift,
        inbound_lookup=inbound_lookup,
        latest_stock_lookup=latest_stock_lookup,
        empty_sales=empty_sales,
        make_forecast_sales_capped_fn=make_forecast_sales_capped,
    )

    sales_forecast = apply_stock_depletion(
        sales_forecast=sales_forecast,
        latest_stock_lookup=latest_stock_lookup,
        inbound_lookup=inbound_lookup,
    )

    sales_forecast["sales_ea"] = (
        pd.to_numeric(sales_forecast.get("sales_ea"), errors="coerce")
        .fillna(0)
        .round()
        .clip(lower=0)
        .astype(int)
    )

    sales_ma7["sales_ea"] = pd.to_numeric(sales_ma7.get("sales_ea"), errors="coerce").fillna(0.0)

    return AmazonForecastContext(
        start=start_norm,
        end=end_norm,
        today=today_norm,
        centers=center_list,
        skus=sku_list,
        inv_actual=inv_actual.reset_index(drop=True),
        inv_forecast=inv_forecast.reset_index(drop=True),
        sales_hist=sales_hist.sort_values(["center", "resource_code", "date"]).reset_index(drop=True),
        sales_ma7=sales_ma7.sort_values(["center", "resource_code", "date"]).reset_index(drop=True),
        sales_forecast=sales_forecast.sort_values(["center", "resource_code", "date"]).reset_index(drop=True),
        snapshot_raw=snapshot_raw_df.reset_index(drop=True),
        snapshot_long=snapshot_long_df.reset_index(drop=True),
        moves=moves_df.reset_index(drop=True),
        lookback_days=int(lookback_days),
        promotion_multiplier=float(promo_multiplier),
    )


