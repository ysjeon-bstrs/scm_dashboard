"""판매 분석 함수들."""

from __future__ import annotations

from typing import Iterable, NamedTuple, Optional, Sequence

import pandas as pd
import numpy as np


class AmazonSalesResult(NamedTuple):
    """Container for aggregated Amazon sales/inventory series."""

    data: pd.DataFrame
    center: str


class AmazonSeriesResult(NamedTuple):
    """Container with aligned Amazon inventory, sales and forecast series."""

    inventory: pd.Series
    sales: pd.Series
    forecast: Optional[pd.Series]
    inbound: Optional[pd.Series]


def prepare_amazon_sales_series(
    snap_long: pd.DataFrame,
    skus: Iterable[str],
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
    *,
    center: str = "AMZUS",
    rolling_window: int = 7,
) -> AmazonSalesResult:
    """아마존 판매 및 재고 시계열을 집계하여 반환합니다.

    Args:
        snap_long: 정규화된 스냅샷 테이블 (date, center, resource_code, stock_qty 컬럼 필요)
        skus: 포함할 SKU 식별자 iterable (비어있으면 빈 DataFrame 반환)
        start_dt / end_dt: 포함할 날짜 범위 (출력은 이 범위의 모든 날짜로 reindex)
        center: 아마존 재고를 나타내는 스냅샷 센터 식별자
        rolling_window: 판매의 이동 평균 윈도우 (일 단위)

    Returns:
        집계된 아마존 판매/재고 시계열 AmazonSalesResult
    """
    sku_list = [str(sku) for sku in skus if pd.notna(sku)]
    if not sku_list:
        return AmazonSalesResult(pd.DataFrame(), center)

    df = snap_long.copy()
    if "date" not in df.columns:
        raise KeyError("snap_long must contain a 'date' column for sales prep")

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df.astype({"center": str, "resource_code": str})

    df = df[(df["center"] == center) & (df["resource_code"].isin(sku_list))]
    if df.empty:
        return AmazonSalesResult(pd.DataFrame(), center)

    idx = pd.date_range(start_dt, end_dt, freq="D")

    series_frames = []
    for sku, grp in df.groupby("resource_code"):
        daily = grp.groupby("date")["stock_qty"].sum().sort_index()
        daily = daily.reindex(idx).ffill().fillna(0)

        delta = daily.diff().fillna(0)
        sales = (-delta).clip(lower=0)

        frame = pd.DataFrame({
            "date": idx,
            "resource_code": sku,
            "inventory_qty": daily.values,
            "sales_qty": sales.values,
        })
        series_frames.append(frame)

    combined = pd.concat(series_frames, ignore_index=True)
    agg = (
        combined.groupby("date", as_index=False)[["inventory_qty", "sales_qty"]]
        .sum()
        .sort_values("date")
    )

    agg["sales_roll_mean"] = (
        agg["sales_qty"].rolling(window=int(max(1, rolling_window)), min_periods=1).mean()
    )

    return AmazonSalesResult(agg, center)


def prepare_amazon_daily_sales(
    snapshot: pd.DataFrame,
    *,
    centers: Sequence[str] | None,
    skus: Iterable[str],
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
    rolling_window: int = 7,
) -> AmazonSalesResult:
    """여러 아마존 센터의 일별 판매를 집계합니다.

    여러 아마존 센터가 제공되면 레코드를 합성 "AMAZON" 센터로 정규화하여
    날짜별로 재고와 일별 판매를 합산합니다.
    """

    if "center" not in snapshot.columns:
        raise KeyError("snapshot must contain a 'center' column for Amazon prep")

    if centers:
        centers_list = [str(center) for center in centers if str(center).strip()]
    else:
        centers_list = []

    df = snapshot.copy()
    df["center"] = df["center"].astype(str)

    if centers_list:
        df = df[df["center"].isin(centers_list)]
        if df.empty:
            label = centers_list[0]
            return AmazonSalesResult(pd.DataFrame(), label)
        label = centers_list[0] if len(centers_list) == 1 else "AMAZON"
    else:
        mask = df["center"].str.contains("amazon", case=False, na=False) | (
            df["center"].str.upper().str.startswith("AMZ")
        )
        df = df[mask]
        if df.empty:
            return AmazonSalesResult(pd.DataFrame(), "AMZUS")
        label = df["center"].iloc[0]

    df = df.copy()
    df["center"] = label

    return prepare_amazon_sales_series(
        df,
        skus,
        start_dt,
        end_dt,
        center=label,
        rolling_window=rolling_window,
    )


def prepare_amazon_inventory_layers(
    timeline: pd.DataFrame,
    *,
    centers: Sequence[str],
    skus: Sequence[str],
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
    forecast_timeline: Optional[pd.DataFrame] = None,
    moves: Optional[pd.DataFrame] = None,
    latest_snapshot: Optional[pd.Timestamp] = None,
) -> AmazonSeriesResult:
    """Return aligned series for the Amazon sales vs. inventory chart."""

    start_norm = pd.to_datetime(start_dt).normalize()
    end_norm = pd.to_datetime(end_dt).normalize()
    if end_norm < start_norm:
        return AmazonSeriesResult(
            pd.Series(dtype=float),
            pd.Series(dtype=float),
            None,
            None,
        )

    sku_list = [str(sku) for sku in skus if str(sku).strip()]
    if not sku_list:
        return AmazonSeriesResult(
            pd.Series(dtype=float),
            pd.Series(dtype=float),
            None,
            None,
        )

    timeline_cols = {"date", "center", "resource_code", "stock_qty"}
    if timeline is None or timeline.empty or not timeline_cols.issubset(timeline.columns):
        return AmazonSeriesResult(
            pd.Series(dtype=float),
            pd.Series(dtype=float),
            None,
            None,
        )

    def _is_amazon_center(value: str) -> bool:
        value = str(value).strip()
        if not value:
            return False
        lowered = value.lower()
        return "amazon" in lowered or value.upper().startswith("AMZ")

    center_list = [str(center) for center in centers if str(center).strip()]
    if not center_list:
        center_list = [
            str(center)
            for center in timeline.get("center", pd.Series(dtype=str)).dropna().astype(str).unique()
            if _is_amazon_center(center)
        ]

    center_list = [center for center in center_list if _is_amazon_center(center)]
    if not center_list:
        return AmazonSeriesResult(
            pd.Series(dtype=float),
            pd.Series(dtype=float),
            None,
            None,
        )

    work = timeline.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce").dt.normalize()
    work = work.dropna(subset=["date"])
    work = work.astype({"center": str, "resource_code": str})
    work = work[
        work["center"].isin(center_list) & work["resource_code"].isin(sku_list)
    ]

    if work.empty:
        return AmazonSeriesResult(
            pd.Series(dtype=float),
            pd.Series(dtype=float),
            None,
            None,
        )

    idx = pd.date_range(start_norm, end_norm, freq="D")
    inventory = (
        work.groupby("date")["stock_qty"]
        .sum()
        .sort_index()
        .reindex(idx)
        .ffill()
        .fillna(0.0)
    )
    inventory = inventory.astype(float)
    inventory.name = "inventory_qty"

    delta = inventory.diff().fillna(0.0)
    sales = (-delta).clip(lower=0.0)
    sales = sales.astype(float)
    sales.name = "sales_qty"

    forecast_series: Optional[pd.Series] = None
    if forecast_timeline is not None and not forecast_timeline.empty:
        fc_cols = {"date", "center", "resource_code", "stock_qty"}
        if fc_cols.issubset(forecast_timeline.columns):
            fc = forecast_timeline.copy()
            fc["date"] = pd.to_datetime(fc["date"], errors="coerce").dt.normalize()
            fc = fc.dropna(subset=["date"])
            fc = fc.astype({"center": str, "resource_code": str})
            fc = fc[
                fc["center"].isin(center_list) & fc["resource_code"].isin(sku_list)
            ]
            if not fc.empty:
                forecast_series = (
                    fc.groupby("date")["stock_qty"]
                    .sum()
                    .sort_index()
                    .reindex(idx)
                    .ffill()
                    .fillna(0.0)
                )
                forecast_series = forecast_series.astype(float)
                forecast_series.name = "forecast_inventory_qty"

                if latest_snapshot is not None and pd.notna(latest_snapshot):
                    cons_start = pd.to_datetime(latest_snapshot).normalize() + pd.Timedelta(days=1)
                    mask = forecast_series.index < cons_start
                    if mask.any():
                        forecast_series.loc[mask] = np.nan
                        prev_day = cons_start - pd.Timedelta(days=1)
                        if prev_day in forecast_series.index and prev_day in inventory.index:
                            forecast_series.loc[prev_day] = inventory.loc[prev_day]

    inbound_series: Optional[pd.Series] = None
    if moves is not None and not moves.empty:
        required_move_cols = {"inbound_date", "qty_ea", "to_center", "resource_code"}
        if required_move_cols.issubset(moves.columns):
            inbound = moves.copy()
            inbound = inbound.dropna(subset=["inbound_date"])
            if not inbound.empty:
                inbound["inbound_date"] = pd.to_datetime(
                    inbound["inbound_date"], errors="coerce"
                ).dt.normalize()
                inbound = inbound.dropna(subset=["inbound_date"])
                inbound = inbound.astype({"to_center": str, "resource_code": str})
                inbound = inbound[
                    inbound["to_center"].isin(center_list)
                    & inbound["resource_code"].isin(sku_list)
                ]
                if not inbound.empty:
                    inbound_series = (
                        inbound.groupby("inbound_date")["qty_ea"]
                        .sum()
                        .sort_index()
                        .reindex(idx)
                        .fillna(0.0)
                    )
                    inbound_series = inbound_series.astype(float)
                    inbound_series.name = "inbound_qty"

    return AmazonSeriesResult(inventory, sales, forecast_series, inbound_series)


__all__ = [
    "AmazonSeriesResult",
    "prepare_amazon_sales_series",
    "prepare_amazon_daily_sales",
    "prepare_amazon_inventory_layers",
]
