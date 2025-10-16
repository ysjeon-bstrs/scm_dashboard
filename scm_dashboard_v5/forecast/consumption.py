"""Forecast-period helpers that mirror the proven v4 logic for v5."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from center_alias import normalize_center_value
from scm_dashboard_v4 import consumption as v4_consumption
from scm_dashboard_v5.core import build_timeline


@dataclass
class AmazonForecastContext:
    """Bundle actual and forecast series required by the Amazon panel."""

    start: pd.Timestamp
    end: pd.Timestamp
    today: pd.Timestamp
    centers: list[str]
    skus: list[str]
    inv_actual: pd.DataFrame
    inv_forecast: pd.DataFrame
    sales_hist: pd.DataFrame
    sales_ma7: pd.DataFrame
    sales_forecast: pd.DataFrame
    snapshot_raw: pd.DataFrame = field(default_factory=pd.DataFrame)
    snapshot_long: pd.DataFrame = field(default_factory=pd.DataFrame)
    moves: pd.DataFrame = field(default_factory=pd.DataFrame)
    lookback_days: int = 28
    promotion_multiplier: float = 1.0


def make_forecast_sales_capped(
    base_daily_pred: pd.Series,
    latest_stock: float,
    inbound_by_day: pd.Series | None = None,
) -> pd.Series:
    """Return a forecast series clipped by available stock and inbound events."""

    if base_daily_pred is None:
        return pd.Series(dtype=float)

    base = base_daily_pred.astype(float).copy()
    if base.empty:
        return pd.Series(dtype=float, index=base.index)

    inbound: pd.Series
    if inbound_by_day is None:
        inbound = pd.Series(0.0, index=base.index, dtype=float)
    else:
        inbound = (
            inbound_by_day.astype(float)
            .reindex(base.index, fill_value=0.0)
        )

    remain = float(max(latest_stock, 0.0))
    capped: list[float] = []

    for day in base.index:
        remain += float(inbound.loc[day])
        want = float(base.loc[day])
        sale = min(want, max(remain, 0.0))
        capped.append(sale)
        remain -= sale

    return pd.Series(capped, index=base.index, dtype=float)


def load_amazon_daily_sales_from_snapshot_raw(
    snapshot_raw: pd.DataFrame,
    centers: Tuple[str, ...] = ("AMZUS",),
    skus: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Return daily FBA outbound volumes from ``snapshot_raw``.

    The legacy ``snapshot_raw`` sheet occasionally exposes the FBA outbound
    quantity column with different labels.  This helper normalises the
    structure so the downstream sales simulator can reuse the values as
    ground-truth daily sales.
    """

    if snapshot_raw is None or snapshot_raw.empty:
        return pd.DataFrame(columns=["date", "center", "resource_code", "sales_ea"])

    df = snapshot_raw.copy()

    rename_map: dict[str, str] = {}
    for col in df.columns:
        key = str(col).strip().lower()
        if key in {"snapshot_date", "date", "스냅샷일자", "스냅샷 일자"}:
            rename_map[col] = "date"
        elif key in {"center", "창고", "창고명"}:
            rename_map[col] = "center"
        elif key in {"resource_code", "sku", "상품코드", "product_code"}:
            rename_map[col] = "resource_code"
        elif "fba_output_stock" in key:
            rename_map[col] = "fba_output_stock"

    df = df.rename(columns=rename_map)

    # Allow missing center in snapshot_raw for Amazon-only sheets by inferring a default
    # centre from the requested centres (defaults to AMZUS). This keeps historical
    # bars visible in environments where the column is omitted.
    if "center" not in df.columns:
        inferred_center = None
        if centers:
            # pick the first requested centre after normalisation
            for ct in centers:
                norm = normalize_center_value(ct)
                if norm:
                    inferred_center = norm
                    break
        if inferred_center is None:
            inferred_center = "AMZUS"
        df["center"] = inferred_center

    required = {"date", "resource_code", "fba_output_stock", "center"}
    missing = required - set(df.columns)
    if missing:
        return pd.DataFrame(columns=["date", "center", "resource_code", "sales_ea"])

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["fba_output_stock"] = pd.to_numeric(df["fba_output_stock"], errors="coerce").fillna(0)
    df = df[df["fba_output_stock"] >= 0]

    if "center" in df.columns:
        df["center"] = df["center"].apply(normalize_center_value)
        df = df[df["center"].notna()]

    if centers:
        center_set = {
            normalized
            for ct in centers
            for normalized in [normalize_center_value(ct)]
            if normalized
        }
        if not center_set:
            return pd.DataFrame(columns=["date", "center", "resource_code", "sales_ea"])
        df = df[df["center"].astype(str).isin(center_set)]

    if skus:
        sku_set = {str(sku) for sku in skus}
        df = df[df["resource_code"].astype(str).isin(sku_set)]

    grouped = (
        df.groupby(["date", "center", "resource_code"], as_index=False)["fba_output_stock"]
        .sum()
        .rename(columns={"fba_output_stock": "sales_ea"})
    )
    grouped["sales_ea"] = grouped["sales_ea"].fillna(0).astype(float)
    return grouped


def forecast_sales_and_inventory(
    daily_sales: pd.DataFrame,
    timeline_center: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    lookback_days: int = 28,
    uplift_events: Optional[Iterable[dict]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate sales and inventory simultaneously for Amazon centres.

    The simulator ensures that forecast sales never exceed the available
    inventory (including inbound events present on the centre timeline).  Once
    the simulated stock reaches zero, the sales forecast is automatically
    capped at zero, keeping the OOS timing aligned between charts and KPIs.
    """

    if timeline_center is None or timeline_center.empty:
        empty_sales = pd.DataFrame(columns=["date", "center", "resource_code", "sales_ea"])
        empty_inv = pd.DataFrame(columns=["date", "center", "resource_code", "stock_qty"])
        return empty_sales, empty_inv

    if daily_sales is None or daily_sales.empty:
        sales_history = pd.DataFrame(columns=["date", "center", "resource_code", "sales_ea"])
    else:
        sales_history = daily_sales.copy()
        rename_map = {col.lower(): col for col in sales_history.columns}
        if "date" not in sales_history.columns and "snapshot_date" in rename_map:
            sales_history = sales_history.rename(columns={rename_map["snapshot_date"]: "date"})
        if "center" not in sales_history.columns and "center" in rename_map:
            sales_history = sales_history.rename(columns={rename_map["center"]: "center"})
        if "resource_code" not in sales_history.columns and "resource_code" in rename_map:
            sales_history = sales_history.rename(columns={rename_map["resource_code"]: "resource_code"})
        sales_history["date"] = pd.to_datetime(sales_history["date"], errors="coerce").dt.normalize()
        if "sales_ea" not in sales_history.columns:
            value_col = None
            for candidate in sales_history.columns:
                if str(candidate).lower().endswith("sales_ea"):
                    value_col = candidate
                    break
            if value_col is not None:
                sales_history = sales_history.rename(columns={value_col: "sales_ea"})
        if "sales_ea" in sales_history.columns:
            coerced = pd.to_numeric(sales_history["sales_ea"], errors="coerce")
        else:
            coerced = pd.Series(0.0, index=sales_history.index, dtype=float)
        sales_history["sales_ea"] = coerced.fillna(0.0)
        sales_history = sales_history.dropna(subset=["date"])

    timeline = timeline_center.copy()
    timeline["date"] = pd.to_datetime(timeline["date"], errors="coerce").dt.normalize()
    timeline["center"] = timeline.get("center", "").astype(str)
    timeline["resource_code"] = timeline.get("resource_code", "").astype(str)
    timeline["stock_qty"] = pd.to_numeric(timeline.get("stock_qty"), errors="coerce").fillna(0)

    center_set = timeline["center"].dropna().astype(str).unique().tolist()
    sku_set = timeline["resource_code"].dropna().astype(str).unique().tolist()
    if not center_set or not sku_set:
        empty_sales = pd.DataFrame(columns=["date", "center", "resource_code", "sales_ea"])
        empty_inv = pd.DataFrame(columns=["date", "center", "resource_code", "stock_qty"])
        return empty_sales, empty_inv

    start_norm = pd.to_datetime(start).normalize()
    end_norm = pd.to_datetime(end).normalize()
    if end_norm < start_norm:
        empty_sales = pd.DataFrame(columns=["date", "center", "resource_code", "sales_ea"])
        empty_inv = pd.DataFrame(columns=["date", "center", "resource_code", "stock_qty"])
        return empty_sales, empty_inv

    index = pd.date_range(start_norm, end_norm, freq="D")

    sku_mask = sales_history.get("resource_code")
    if sku_mask is None:
        filtered_sales = pd.DataFrame(columns=["date", "center", "resource_code", "sales_ea"])
    else:
        sku_mask = sales_history["resource_code"].astype(str).isin(sku_set)
        if "center" in sales_history.columns:
            center_mask = sales_history["center"].astype(str).isin(center_set)
        else:
            center_mask = pd.Series(True, index=sales_history.index)
        filtered_sales = sales_history[sku_mask & center_mask].copy()
        if "center" not in filtered_sales.columns:
            inferred_center = center_set[0] if len(center_set) == 1 else ""
            filtered_sales["center"] = inferred_center

    if filtered_sales.empty:
        pivot = pd.DataFrame(index=index)
        pivot.index.name = "date"
        pivot.columns = pd.MultiIndex(
            levels=[[], []], codes=[[], []], names=["center", "resource_code"]
        )
    else:
        filtered_sales["date"] = pd.to_datetime(
            filtered_sales["date"], errors="coerce"
        ).dt.normalize()
        grouped_sales = (
            filtered_sales.groupby(["date", "center", "resource_code"], as_index=False)[
                "sales_ea"
            ]
            .sum()
            .sort_values("date")
        )
        pivot = (
            grouped_sales.pivot_table(
                index="date",
                columns=["center", "resource_code"],
                values="sales_ea",
                aggfunc="sum",
            )
            .sort_index()
        )
        pivot.index = pd.DatetimeIndex(pivot.index)
        pivot = pivot.asfreq("D", fill_value=0)

    if lookback_days <= 0:
        lookback_days = 1
    mean_rate = pivot.tail(int(lookback_days)).mean().clip(lower=0)

    uplift = pd.Series(1.0, index=index)
    if uplift_events:
        for event in uplift_events:
            s = pd.to_datetime(event.get("start"), errors="coerce")
            t = pd.to_datetime(event.get("end"), errors="coerce")
            u = float(event.get("uplift", 0.0))
            if pd.notna(s) and pd.notna(t):
                s_norm = max(s.normalize(), index[0])
                t_norm = min(t.normalize(), index[-1])
                if s_norm <= t_norm:
                    uplift.loc[s_norm:t_norm] = uplift.loc[s_norm:t_norm] * (1.0 + u)

    timeline_rows: list[pd.DataFrame] = []
    sales_rows: list[pd.DataFrame] = []

    for (center_value, sku), group in timeline.groupby(["center", "resource_code"]):
        group = group.sort_values("date").copy()
        group = group.set_index("date")
        series = group["stock_qty"].asfreq("D")
        series = series.ffill()
        if series.empty:
            continue

        anchor_date = min(series.index.min(), start_norm - pd.Timedelta(days=1))
        full_index = pd.date_range(anchor_date, index[-1], freq="D")
        series = series.reindex(full_index, method="ffill").fillna(0.0)

        cons_start = max(start_norm, series.index.min())
        inv_start = float(series.get(cons_start - pd.Timedelta(days=1), series.iloc[-1]))

        deltas = series.diff().fillna(0.0)
        inbound_future = deltas.loc[cons_start:].clip(lower=0)

        base_rate = float(mean_rate.get((center_value, sku), 0.0))
        if base_rate < 0:
            base_rate = 0.0

        uplift_slice = uplift.reindex(index, fill_value=1.0)
        daily_target = pd.Series(base_rate, index=index) * uplift_slice

        inventory = inv_start
        inv_curve: list[float] = []
        sales_curve: list[float] = []

        for day in index:
            inventory += float(inbound_future.get(day, 0.0))
            desired = float(daily_target.loc[day])
            sell = min(inventory, desired)
            inventory = max(0.0, inventory - sell)
            sales_curve.append(sell)
            inv_curve.append(inventory)

        df_sales = pd.DataFrame(
            {
                "date": index,
                "center": center_value,
                "resource_code": sku,
                "sales_ea": np.array(sales_curve, dtype=float),
            }
        )
        df_inv = pd.DataFrame(
            {
                "date": index,
                "center": center_value,
                "resource_code": sku,
                "stock_qty": np.array(inv_curve, dtype=float),
            }
        )

        sales_rows.append(df_sales)
        timeline_rows.append(df_inv)

    forecast_sales = (
        pd.concat(sales_rows, ignore_index=True)
        if sales_rows
        else pd.DataFrame(columns=["date", "center", "resource_code", "sales_ea"])
    )
    forecast_inv = (
        pd.concat(timeline_rows, ignore_index=True)
        if timeline_rows
        else pd.DataFrame(columns=["date", "center", "resource_code", "stock_qty"])
    )

    forecast_sales["date"] = pd.to_datetime(forecast_sales["date"], errors="coerce").dt.normalize()
    forecast_inv["date"] = pd.to_datetime(forecast_inv["date"], errors="coerce").dt.normalize()

    forecast_sales["sales_ea"] = forecast_sales["sales_ea"].fillna(0).round().clip(lower=0).astype(int)
    forecast_inv["stock_qty"] = forecast_inv["stock_qty"].fillna(0).round().clip(lower=0).astype(int)

    history = filtered_sales[["date", "center", "resource_code", "sales_ea"]].copy()
    history["sales_ea"] = pd.to_numeric(history["sales_ea"], errors="coerce").fillna(0)
    history["sales_ea"] = history["sales_ea"].round().clip(lower=0).astype(int)

    combined_sales = pd.concat([history, forecast_sales], ignore_index=True)
    combined_sales = combined_sales.sort_values(["resource_code", "date"]).drop_duplicates(
        subset=["date", "center", "resource_code"], keep="last"
    )

    return combined_sales.reset_index(drop=True), forecast_inv.reset_index(drop=True)


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
            remain_after: list[float] = []
            for day in fc_series.index:
                remain += float(inbound_series.loc[day])
                want = float(fc_series.loc[day])
                sale = min(want, max(remain, 0.0))
                remain -= sale
                remain_after.append(remain)

            if remain_after:
                remain_series = pd.Series(remain_after, index=fc_series.index, dtype=float)
                if (remain_series <= tol).any():
                    remain_array = remain_series.to_numpy()
                    suffix_max = np.maximum.accumulate(remain_array[::-1])[::-1]
                    final_mask = (remain_array <= tol) & (suffix_max <= tol)
                    if final_mask.any():
                        depletion_dates[(center, sku)] = remain_series.index[int(np.argmax(final_mask))]

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

def estimate_daily_consumption(sales: pd.DataFrame, *, window: int = 28) -> pd.DataFrame:
    """Delegate to the stable v4 estimator while keeping the new namespace."""

    return v4_consumption.estimate_daily_consumption(sales, window=window)


def apply_consumption_with_events(
    timeline: pd.DataFrame,
    snapshot: pd.DataFrame,
    *,
    centers: Sequence[str],
    skus: Sequence[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    lookback_days: int = 28,
    events: Optional[Iterable[dict]] = None,
    cons_start: pd.Timestamp | None = None,
) -> pd.DataFrame:
    centers_list = list(centers)
    skus_list = list(skus)
    start_norm = pd.to_datetime(start).normalize()
    end_norm = pd.to_datetime(end).normalize()
    events_list = list(events) if events else None

    out = timeline.copy()
    if out.empty:
        return out

    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()

    snap_cols = {c.lower(): c for c in snapshot.columns}
    date_col = snap_cols.get("date") or snap_cols.get("snapshot_date")
    if date_col is None:
        raise KeyError("snapshot must include a 'date' or 'snapshot_date' column")

    latest_vals = pd.to_datetime(snapshot[date_col], errors="coerce").dropna()
    latest_snap = latest_vals.max().normalize() if not latest_vals.empty else pd.NaT

    if cons_start is not None:
        cons_start_norm = pd.to_datetime(cons_start).normalize()
        cons_start_norm = max(cons_start_norm, start_norm)
        if not pd.isna(latest_snap):
            cons_start_norm = max(cons_start_norm, latest_snap + pd.Timedelta(days=1))
    elif pd.isna(latest_snap):
        cons_start_norm = start_norm
    else:
        cons_start_norm = max(latest_snap + pd.Timedelta(days=1), start_norm)

    if cons_start_norm > end_norm:
        return out

    idx = pd.date_range(cons_start_norm, end_norm, freq="D")
    uplift = pd.Series(1.0, index=idx)
    if events_list:
        for event in events_list:
            s = pd.to_datetime(event.get("start"), errors="coerce")
            t = pd.to_datetime(event.get("end"), errors="coerce")
            u = min(3.0, max(-1.0, float(event.get("uplift", 0.0))))
            if pd.notna(s) and pd.notna(t):
                s = s.normalize()
                t = t.normalize()
                s = max(s, idx[0])
                t = min(t, idx[-1])
                if s <= t:
                    uplift.loc[s:t] = uplift.loc[s:t] * (1.0 + u)

    rates = {}
    if not pd.isna(latest_snap):
        rates = v4_consumption.estimate_daily_consumption(
            snapshot,
            centers_list,
            skus_list,
            latest_snap,
            int(lookback_days),
        )

    chunks: list[pd.DataFrame] = []
    for (ct, sku), grp in out.groupby(["center", "resource_code"]):
        g = grp.sort_values("date").copy()
        g["stock_qty"] = pd.to_numeric(g.get("stock_qty"), errors="coerce")

        if ct in ("In-Transit", "WIP"):
            chunks.append(g)
            continue

        g["stock_qty"] = g["stock_qty"].ffill()

        rate = float(rates.get((ct, sku), 0.0)) if rates else 0.0
        if rate > 0:
            mask = g["date"] >= cons_start_norm
            if mask.any():
                daily = g.loc[mask, "date"].map(uplift).fillna(1.0).values * rate
                stk = g.loc[mask, "stock_qty"].astype(float).values
                for i in range(len(stk)):
                    dec = daily[i]
                    stk[i:] = np.maximum(0.0, stk[i:] - dec)
                g.loc[mask, "stock_qty"] = stk

        chunks.append(g)

    if not chunks:
        return out

    combined = pd.concat(chunks, ignore_index=True)
    combined = combined.sort_values(["center", "resource_code", "date"])
    combined["stock_qty"] = pd.to_numeric(combined["stock_qty"], errors="coerce")
    ffill_mask = ~combined["center"].isin(["In-Transit", "WIP"])
    combined.loc[ffill_mask, "stock_qty"] = (
        combined.loc[ffill_mask]
        .groupby(["center", "resource_code"])["stock_qty"]
        .ffill()
    )
    combined["stock_qty"] = combined["stock_qty"].fillna(0)
    combined["stock_qty"] = combined["stock_qty"].replace([np.inf, -np.inf], 0)
    combined["stock_qty"] = (
        combined["stock_qty"].round().clip(lower=0).astype(int)
    )

    return combined
