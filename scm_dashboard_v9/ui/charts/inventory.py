"""재고 데이터 헬퍼 모듈.

재고 시계열 처리, 매트릭스 변환, 예측 trimming 등을 제공합니다.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from .data_utils import coerce_cols, ensure_naive_index, safe_dataframe



def total_inventory_series(
    inv_actual: pd.DataFrame,
    inv_forecast: pd.DataFrame,
    *,
    sku: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.Series:
    """Aggregate actual/forecast inventory for a SKU across centres."""

    start_norm = pd.to_datetime(start).normalize()
    end_norm = pd.to_datetime(end).normalize()
    if pd.isna(start_norm) or pd.isna(end_norm) or end_norm < start_norm:
        return pd.Series(dtype=float)

    sku_str = str(sku)
    frames: list[pd.DataFrame] = []
    for frame in (inv_actual, inv_forecast):
        if frame is None or frame.empty:
            continue
        if "resource_code" not in frame.columns or "date" not in frame.columns:
            continue
        chunk = frame.copy()
        chunk["resource_code"] = chunk.get("resource_code", "").astype(str)
        chunk = chunk[chunk["resource_code"] == sku_str]
        if chunk.empty:
            continue
        chunk["date"] = pd.to_datetime(
            chunk.get("date"), errors="coerce"
        ).dt.normalize()
        chunk = chunk.dropna(subset=["date"])
        if chunk.empty:
            continue
        qty = pd.to_numeric(chunk.get("stock_qty"), errors="coerce").fillna(0.0)
        chunk = chunk.assign(stock_qty=qty)
        frames.append(chunk[["date", "stock_qty"]])

    if not frames:
        # When no inventory data exists for the SKU we should not fabricate a
        # zero-filled timeline. Returning an empty, date-indexed series allows
        # downstream callers to treat the situation as "no data" instead of
        # "out of stock", so forecasts remain visible until real inventory
        # reaches zero.
        empty_index = pd.DatetimeIndex([], name="date")
        return pd.Series(dtype=float, index=empty_index)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.groupby("date")["stock_qty"].sum().sort_index()

    index = pd.date_range(start_norm, end_norm, freq="D")
    if index.empty:
        return pd.Series(dtype=float)

    combined = combined.reindex(index)
    if combined.notna().any():
        combined = combined.ffill()
        combined = combined.bfill()
    combined = combined.fillna(0.0)
    combined.index.name = "date"
    return combined.astype(float)


def trim_sales_forecast_to_inventory(
    forecast_df: pd.DataFrame,
    inv_actual: pd.DataFrame,
    inv_forecast: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    forecast_start: pd.Timestamp,
) -> pd.DataFrame:
    """Remove forecast rows that extend beyond the first stock-out date."""

    if forecast_df is None or forecast_df.empty:
        return forecast_df

    forecast = forecast_df.copy()
    forecast["date"] = pd.to_datetime(
        forecast.get("date"), errors="coerce"
    ).dt.normalize()
    forecast = forecast.dropna(subset=["date"])
    if forecast.empty:
        return forecast

    trimmed_frames: list[pd.DataFrame] = []
    for sku, group in forecast.groupby("resource_code"):
        inv_series = total_inventory_series(
            inv_actual,
            inv_forecast,
            sku=sku,
            start=start,
            end=end,
        )
        if inv_series.empty:
            trimmed_frames.append(group)
            continue

        zero_candidates = inv_series.loc[inv_series.index >= forecast_start]
        zero_dates = zero_candidates[zero_candidates <= 0]
        if zero_dates.empty:
            trimmed_frames.append(group)
            continue

        cutoff = zero_dates.index[0]
        trimmed = group[group["date"] <= cutoff]
        trimmed_frames.append(trimmed)

    if not trimmed_frames:
        return forecast.iloc[0:0]

    result = pd.concat(trimmed_frames, ignore_index=True)
    result = result.sort_values(["resource_code", "date"]).reset_index(drop=True)
    return result


def inventory_matrix(
    snap_long: pd.DataFrame,
    centers: List[str],
    skus: List[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """선택 센터×SKU의 재고(실측) 시계열 매트릭스. index=date, columns=sku"""
    c = coerce_cols(snap_long)
    s = snap_long.rename(
        columns={
            c["date"]: "date",
            c["center"]: "center",
            c["sku"]: "resource_code",
            c["qty"]: "stock_qty",
        }
    )[["date", "center", "resource_code", "stock_qty"]].copy()

    s["date"] = pd.to_datetime(s["date"], errors="coerce").dt.normalize()
    s = s[
        s["center"].astype(str).isin(centers)
        & s["resource_code"].astype(str).isin(skus)
    ]
    if s.empty:
        idx = pd.date_range(start, end, freq="D")
        return pd.DataFrame(0, index=idx, columns=skus)

    pv = (
        s.groupby(["date", "resource_code"])["stock_qty"]
        .sum()
        .unstack("resource_code")
        .reindex(columns=skus, fill_value=0)
        .sort_index()
    )
    pv = pv.asfreq("D").ffill()
    pv = pv.loc[(pv.index >= start) & (pv.index <= end)]
    return pv


def timeline_inventory_matrix(
    timeline: Optional[pd.DataFrame],
    centers: Sequence[str],
    skus: Sequence[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> Optional[pd.DataFrame]:
    """Pivot the step-chart timeline into a date×SKU inventory matrix."""

    if timeline is None or timeline.empty:
        return None

    required_cols = {"date", "center", "resource_code", "stock_qty"}
    if not required_cols.issubset(timeline.columns):
        return None

    df = timeline.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df[df["date"].notna()]
    df = df[df["center"].astype(str).isin(centers)]
    df = df[df["resource_code"].astype(str).isin(skus)]

    if df.empty:
        return None

    pivot = (
        df.groupby(["date", "resource_code"])["stock_qty"]
        .sum()
        .unstack("resource_code")
        .reindex(columns=list(skus), fill_value=0.0)
        .sort_index()
    )

    pivot = pivot.loc[(pivot.index >= start) & (pivot.index <= end)]
    return pivot


# ---------------- Public renderer ----------------


def clamped_forecast_series(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    base_stock: float,
    inbound_by_day: dict[pd.Timestamp, float],
    daily_demand: float,
) -> tuple[pd.Series, pd.Series]:
    """Return paired (sales, inventory) series respecting remaining stock."""

    if pd.isna(start_date) or pd.isna(end_date) or end_date < start_date:
        empty_index = pd.DatetimeIndex([], dtype="datetime64[ns]")
        return pd.Series(dtype=float, index=empty_index), pd.Series(
            dtype=float, index=empty_index
        )

    idx = pd.date_range(start_date, end_date, freq="D")
    fcst_sales = pd.Series(0.0, index=idx, dtype=float)
    inv = pd.Series(np.nan, index=idx, dtype=float)

    remain = float(base_stock)
    demand = max(0.0, float(daily_demand))

    for d in idx:
        inbound_qty = float(inbound_by_day.get(pd.to_datetime(d), 0.0))
        remain += inbound_qty

        sell = min(demand, max(remain, 0.0))
        fcst_sales.loc[d] = sell
        remain -= sell

        inv.loc[d] = max(remain, 0.0)

    return fcst_sales, inv
