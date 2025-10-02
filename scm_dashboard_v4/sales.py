"""Sales analytics helpers for the SCM dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import pandas as pd


@dataclass
class SalesSeries:
    """Simple container for prepared Amazon sales data."""

    frame: pd.DataFrame

    @property
    def empty(self) -> bool:
        return self.frame.empty


def _normalize_list(values: Optional[Iterable[str]]) -> list[str]:
    if not values:
        return []
    return [str(v).strip() for v in values if str(v).strip()]


def prepare_amazon_daily_sales(
    snapshots: Optional[pd.DataFrame],
    *,
    center_keyword: str = "amazon",
    centers: Optional[Sequence[str]] = None,
    skus: Optional[Sequence[str]] = None,
    rolling_window: int = 7,
) -> SalesSeries:
    """Return daily Amazon inventory & sales derived from snapshot data."""

    empty_result = SalesSeries(
        pd.DataFrame(
            {
                "date": pd.Series(dtype="datetime64[ns]"),
                "inventory_qty": pd.Series(dtype="float"),
                "daily_sales": pd.Series(dtype="float"),
            }
        )
    )

    if snapshots is None or snapshots.empty:
        return empty_result

    df = snapshots.copy()

    if "date" not in df.columns or "center" not in df.columns or "stock_qty" not in df.columns:
        return empty_result

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["date"])
    if df.empty:
        return empty_result

    df["center"] = df["center"].astype(str)
    if "resource_code" in df.columns:
        df["resource_code"] = df["resource_code"].astype(str)
    else:
        df["resource_code"] = ""

    selected_centers = _normalize_list(centers)
    if selected_centers:
        center_mask = df["center"].isin(selected_centers)
    else:
        center_mask = df["center"].str.contains(center_keyword, case=False, na=False)

    df = df[center_mask]
    if df.empty:
        return empty_result

    selected_skus = _normalize_list(skus)
    if selected_skus:
        df = df[df["resource_code"].isin(selected_skus)]
        if df.empty:
            return empty_result

    grouped = df.groupby("date", as_index=True)["stock_qty"].sum().sort_index()
    if grouped.empty:
        return empty_result

    full_range = pd.date_range(grouped.index.min(), grouped.index.max(), freq="D")
    inventory = grouped.reindex(full_range).ffill()

    sales = (-inventory.diff()).clip(lower=0)
    sales = sales.fillna(0)

    result = pd.DataFrame(
        {
            "date": inventory.index,
            "inventory_qty": inventory.values,
            "daily_sales": sales.values,
        }
    )

    if rolling_window and rolling_window > 1:
        result[f"sales_ma_{rolling_window}"] = (
            result["daily_sales"].rolling(rolling_window, min_periods=1).mean()
        )

    return SalesSeries(result)
