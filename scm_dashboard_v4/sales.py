"""Utilities for preparing Amazon US sales and inventory series."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


import pandas as pd


@dataclass
class AmazonSalesResult:
    """Container for aggregated Amazon sales/inventory series."""

    data: pd.DataFrame
    center: str


def prepare_amazon_sales_series(
    snap_long: pd.DataFrame,
    skus: Iterable[str],
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
    center: str = "AMZUS",
    rolling_window: int = 7,
) -> AmazonSalesResult:
    """Return aggregated Amazon sales and inventory series for the requested window.

    Parameters
    ----------
    snap_long:
        Normalised snapshot table containing at least ``date``, ``center``,
        ``resource_code`` and ``stock_qty`` columns.
    skus:
        Iterable of SKU identifiers to include. When empty the function returns
        an empty dataframe.
    start_dt / end_dt:
        Inclusive date range to cover. The output is reindexed to every day in
        this window so charts never break on sparse data.
    center:
        Snapshot centre identifier representing Amazon US inventory.
    rolling_window:
        Window (in days) for the optional rolling average of sales.
    """

    sku_list = [str(sku) for sku in skus if pd.notna(sku)]
    if not sku_list:
        return AmazonSalesResult(pd.DataFrame(), center)

    df = snap_long.copy()
    if "date" not in df.columns:
        raise KeyError("snap_long must contain a 'date' column for sales prep")

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["center"] = df["center"].astype(str)
    df["resource_code"] = df["resource_code"].astype(str)

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
