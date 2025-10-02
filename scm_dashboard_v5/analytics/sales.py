"""Sales adapters extending the existing v4 series helpers for v5."""

from __future__ import annotations

from typing import Iterable, Sequence

import pandas as pd

from scm_dashboard_v4 import sales as v4_sales


def prepare_amazon_sales_series(
    snapshot: pd.DataFrame,
    skus: Iterable[str],
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
    *,
    center: str = "AMZUS",
    rolling_window: int = 7,
) -> v4_sales.AmazonSalesResult:
    """Proxy to the v4 helper for single-centre Amazon series."""

    return v4_sales.prepare_amazon_sales_series(
        snapshot,
        skus,
        start_dt,
        end_dt,
        center=center,
        rolling_window=rolling_window,
    )


def prepare_amazon_daily_sales(
    snapshot: pd.DataFrame,
    *,
    centers: Sequence[str] | None,
    skus: Iterable[str],
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
    rolling_window: int = 7,
) -> v4_sales.AmazonSalesResult:
    """Aggregate Amazon daily sales across the supplied centres.

    When multiple Amazon centres are provided the records are normalised to a
    synthetic "AMAZON" centre before delegating to the v4 helper so inventory
    and daily sales are summed per date.
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
            return v4_sales.AmazonSalesResult(pd.DataFrame(), label)
        label = centers_list[0] if len(centers_list) == 1 else "AMAZON"
    else:
        mask = df["center"].str.contains("amazon", case=False, na=False) | (
            df["center"].str.upper().str.startswith("AMZ")
        )
        df = df[mask]
        if df.empty:
            return v4_sales.AmazonSalesResult(pd.DataFrame(), "AMZUS")
        label = df["center"].iloc[0]

    df = df.copy()
    df["center"] = label

    return v4_sales.prepare_amazon_sales_series(
        df,
        skus,
        start_dt,
        end_dt,
        center=label,
        rolling_window=rolling_window,
    )


__all__ = ["prepare_amazon_sales_series", "prepare_amazon_daily_sales"]
