"""Amazon Forecast Context 빌더.

전체 예측 컨텍스트를 생성하는 복잡한 로직을 제공합니다.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
import streamlit as st

from center_alias import normalize_center_value
from scm_dashboard_v4 import consumption as v4_consumption
from scm_dashboard_v9.core import build_timeline

from .models import AmazonForecastContext
from .sales import load_amazon_daily_sales_from_snapshot_raw, make_forecast_sales_capped
from .inventory import forecast_sales_and_inventory
from .estimation import apply_consumption_with_events

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

    promo_multiplier = 1.0
    if promotion_events:
        for event in promotion_events:
            try:
                uplift_val = float(event.get("uplift", 0.0))
            except (TypeError, ValueError):
                continue
            uplift_val = min(3.0, max(-1.0, uplift_val))
            promo_multiplier *= 1.0 + uplift_val
    if not np.isfinite(promo_multiplier) or promo_multiplier <= 0:
        promo_multiplier = 1.0

    start_norm = pd.to_datetime(start).normalize()
    end_norm = pd.to_datetime(end).normalize()
    today_norm = pd.to_datetime(today).normalize()

    center_list: list[str] = []
    for raw_center in centers:
        normalized = normalize_center_value(raw_center)
        if not normalized:
            continue
        if normalized not in center_list:
            center_list.append(normalized)
    sku_list = [str(sku) for sku in skus if str(sku).strip()]

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

    snapshot_cols = {c.lower(): c for c in snap_long.columns}
    snap_date_col = snapshot_cols.get("snapshot_date") or snapshot_cols.get("date")
    if snap_date_col is not None:
        snap_dates = pd.to_datetime(snap_long[snap_date_col], errors="coerce")
        latest_snapshot = snap_dates.dropna().max()
    else:
        latest_snapshot = pd.NaT

    if pd.isna(latest_snapshot):
        latest_snapshot = today_norm
    else:
        latest_snapshot = pd.to_datetime(latest_snapshot).normalize()

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

    timeline = timeline[timeline["center"].isin(center_list)].copy()
    timeline["date"] = pd.to_datetime(timeline["date"], errors="coerce").dt.normalize()
    timeline["stock_qty"] = pd.to_numeric(timeline.get("stock_qty"), errors="coerce").fillna(0)

    inv_actual = timeline[timeline["date"] <= today_norm].copy()

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
    sales_hist = sales_hist.copy()
    sales_hist["date"] = pd.to_datetime(sales_hist.get("date"), errors="coerce").dt.normalize()
    sales_hist["sales_ea"] = pd.to_numeric(sales_hist.get("sales_ea"), errors="coerce").fillna(0).astype(int)
    sales_hist = sales_hist.dropna(subset=["date"])
    sales_hist = sales_hist[(sales_hist["center"].isin(center_list)) & (sales_hist["resource_code"].isin(sku_list))]

    future_index = (
        pd.date_range(today_norm + pd.Timedelta(days=1), end_norm, freq="D")
        if today_norm < end_norm and use_consumption_forecast
        else pd.DatetimeIndex([], dtype="datetime64[ns]")
    )

    inbound_lookup: dict[tuple[str, str], pd.Series] = {}
    if not future_index.empty and not moves_df.empty:
        move_cols = {str(c).strip().lower(): c for c in moves_df.columns}
        event_col = move_cols.get("event_date")
        to_center_col = move_cols.get("to_center")
        sku_col = move_cols.get("resource_code") or move_cols.get("sku")
        qty_col = move_cols.get("qty_ea") or move_cols.get("qty")

        if event_col and to_center_col and sku_col and qty_col:
            inbound_norm = moves_df.rename(
                columns={
                    event_col: "event_date",
                    to_center_col: "to_center",
                    sku_col: "resource_code",
                    qty_col: "qty_ea",
                }
            ).copy()
            inbound_norm["event_date"] = pd.to_datetime(
                inbound_norm.get("event_date"), errors="coerce"
            ).dt.normalize()
            inbound_norm = inbound_norm.dropna(subset=["event_date"])
            inbound_norm["to_center"] = inbound_norm.get("to_center", "").apply(
                normalize_center_value
            )
            inbound_norm["resource_code"] = inbound_norm.get(
                "resource_code", ""
            ).astype(str)
            inbound_norm["qty_ea"] = pd.to_numeric(
                inbound_norm.get("qty_ea"), errors="coerce"
            ).fillna(0.0)

            inbound_norm = inbound_norm[
                inbound_norm["to_center"].isin(center_list)
                & inbound_norm["resource_code"].isin(sku_list)
            ]

            if not inbound_norm.empty:
                start_future = future_index[0]
                end_future = future_index[-1]
                inbound_norm = inbound_norm[
                    (inbound_norm["event_date"] >= start_future)
                    & (inbound_norm["event_date"] <= end_future)
                ]

                if not inbound_norm.empty:
                    inbound_grouped = (
                        inbound_norm.groupby(
                            ["to_center", "resource_code", "event_date"],
                            as_index=False,
                        )["qty_ea"].sum()
                    )

                    for (ct, sku), chunk in inbound_grouped.groupby(
                        ["to_center", "resource_code"], dropna=True
                    ):
                        series = (
                            chunk.set_index("event_date")["qty_ea"]
                            .reindex(future_index, fill_value=0.0)
                        )
                        inbound_lookup[(ct, sku)] = series

    uplift = pd.Series(1.0, index=future_index, dtype=float)
    if promotion_events and not uplift.empty:
        for event in promotion_events:
            start_evt = pd.to_datetime(event.get("start"), errors="coerce")
            end_evt = pd.to_datetime(event.get("end"), errors="coerce")
            uplift_val = float(event.get("uplift", 0.0))
            uplift_val = min(3.0, max(-1.0, uplift_val))
            if pd.notna(start_evt) and pd.notna(end_evt):
                s_norm = max(start_evt.normalize(), future_index[0])
                e_norm = min(end_evt.normalize(), future_index[-1])
                if s_norm <= e_norm:
                    uplift.loc[s_norm:e_norm] = uplift.loc[s_norm:e_norm] * (1.0 + uplift_val)

    sales_ma7_frames: list[pd.DataFrame] = []
    sales_fc_frames: list[pd.DataFrame] = []

    hist_index = pd.date_range(start_norm, today_norm, freq="D")

    for (center, sku), group in sales_hist.groupby(["center", "resource_code"], dropna=True):
        series = (
            group.sort_values("date")
            .set_index("date")["sales_ea"]
            .reindex(hist_index, fill_value=0.0)
        )
        ma7 = series.rolling(7, min_periods=1).mean()
        ma_df = (
            ma7.to_frame("sales_ea")
            .reset_index()
            .rename(columns={"index": "date"})
        )
        ma_df["center"] = center
        ma_df["resource_code"] = sku
        sales_ma7_frames.append(ma_df)

        if not future_index.empty:
            base_rate = float(ma7.iloc[-1]) if not ma7.empty else 0.0
            fc_values = pd.Series(base_rate, index=future_index, dtype=float)
            if not uplift.empty:
                fc_values = fc_values * uplift.reindex(future_index, fill_value=1.0)
            inbound_series = inbound_lookup.get((center, sku))
            latest_stock = latest_stock_lookup.get((center, sku), 0.0)
            fc_values = make_forecast_sales_capped(
                base_daily_pred=fc_values,
                latest_stock=latest_stock,
                inbound_by_day=inbound_series,
            )
            fc_df = (
                fc_values.to_frame("sales_ea")
                .reset_index()
                .rename(columns={"index": "date"})
            )
            fc_df["center"] = center
            fc_df["resource_code"] = sku
            sales_fc_frames.append(fc_df)

    sales_ma7 = (
        pd.concat(sales_ma7_frames, ignore_index=True)
        if sales_ma7_frames
        else empty_sales.copy()
    )
    sales_ma7["date"] = pd.to_datetime(sales_ma7.get("date"), errors="coerce").dt.normalize()

    sales_forecast = (
        pd.concat(sales_fc_frames, ignore_index=True)
        if sales_fc_frames
        else empty_sales.copy()
    )
    sales_forecast["date"] = pd.to_datetime(sales_forecast.get("date"), errors="coerce").dt.normalize()

    if not sales_forecast.empty:
        depletion_dates: dict[tuple[str, str], pd.Timestamp] = {}
        tol = 1e-6

        for (center, sku), grp in sales_forecast.groupby([
            "center",
            "resource_code",
        ], dropna=True):
            fc_series = (
                grp.sort_values("date")
                .set_index("date")["sales_ea"]
                .astype(float)
            )
            if fc_series.empty:
                continue

            latest_stock = float(latest_stock_lookup.get((center, sku), 0.0))
            inbound_series = inbound_lookup.get((center, sku))
            if inbound_series is None:
                inbound_series = pd.Series(0.0, index=fc_series.index, dtype=float)
            else:
                inbound_series = inbound_series.reindex(fc_series.index, fill_value=0.0).astype(float)

            remain = max(latest_stock, 0.0)
            remain_before: list[float] = []
            sale_list: list[float] = []
            for day in fc_series.index:
                remain += float(inbound_series.loc[day])
                available = max(remain, 0.0)
                remain_before.append(available)
                want = float(fc_series.loc[day])
                sale = min(want, available)
                remain -= sale
                sale_list.append(sale)

            if sale_list:
                sale_array = np.asarray(sale_list, dtype=float)
                before_array = np.asarray(remain_before, dtype=float)
                if ((sale_array <= tol) & (before_array <= tol)).any():
                    suffix_before = np.maximum.accumulate(before_array[::-1])[::-1]
                    suffix_sale = np.maximum.accumulate(sale_array[::-1])[::-1]
                    final_mask = (
                        (sale_array <= tol)
                        & (before_array <= tol)
                        & (suffix_before <= tol)
                        & (suffix_sale <= tol)
                    )
                    if final_mask.any():
                        depletion_dates[(center, sku)] = fc_series.index[int(np.argmax(final_mask))]

        if depletion_dates:
            mask_center = sales_forecast[["center", "resource_code"]].apply(tuple, axis=1)
            for key, dep_date in depletion_dates.items():
                clip_mask = (mask_center == key) & (sales_forecast["date"] > dep_date)
                if clip_mask.any():
                    sales_forecast.loc[clip_mask, "sales_ea"] = 0.0

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


