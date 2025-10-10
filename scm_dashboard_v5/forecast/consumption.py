"""Forecasting adapters that reuse the proven v4 consumption logic."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import pandas as pd

from scm_dashboard_v4 import consumption as v4_consumption


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
    """Delegate to the v4 helper while controlling the consumption window."""

    centers_list = list(centers)
    skus_list = list(skus)
    start_norm = pd.to_datetime(start).normalize()
    end_norm = pd.to_datetime(end).normalize()
    events_list = list(events) if events else None

    timeline_copy = timeline.copy()
    if "date" in timeline_copy.columns:
        timeline_copy["date"] = pd.to_datetime(timeline_copy["date"], errors="coerce").dt.normalize()

    if cons_start is not None:
        cons_start_norm = pd.to_datetime(cons_start).normalize()
        before_mask = timeline_copy["date"] < cons_start_norm
        before = timeline_copy.loc[before_mask].copy()
        after = timeline_copy.loc[~before_mask].copy()
        if after.empty:
            return timeline_copy

        adjusted = v4_consumption.apply_consumption_with_events(
            after,
            snapshot,
            centers_list,
            skus_list,
            cons_start_norm,
            end_norm,
            int(lookback_days),
            events_list,
        )

        combined = pd.concat([before, adjusted], ignore_index=True, sort=False)
        sort_cols = [col for col in ["date", "center", "resource_code"] if col in combined.columns]
        if sort_cols:
            combined = combined.sort_values(sort_cols).reset_index(drop=True)
        return combined

    return v4_consumption.apply_consumption_with_events(
        timeline_copy,
        snapshot,
        centers_list,
        skus_list,
        start_norm,
        end_norm,
        int(lookback_days),
        events_list,
    )

    if cons_start is None or "date" not in result.columns:
        return result

    cons_start_norm = pd.to_datetime(cons_start).normalize()
    result = result.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.normalize()

    if "date" not in timeline_copy.columns:
        return result

    key_cols = [
        col
        for col in ["date", "center", "resource_code"]
        if col in result.columns and col in timeline_copy.columns
    ]

    if not key_cols:
        return result

    orig = (
        timeline_copy[key_cols + ["stock_qty"]]
        .copy()
        .rename(columns={"stock_qty": "_orig_stock_qty"})
    )

    # Ensure unique keys to avoid Cartesian products during alignment.
    orig = orig.drop_duplicates(subset=key_cols, keep="last")

    merged = result.merge(orig, on=key_cols, how="left")
    mask = merged["date"] < cons_start_norm
    if mask.any():
        restored = merged.loc[mask, "_orig_stock_qty"].fillna(merged.loc[mask, "stock_qty"])
        merged.loc[mask, "stock_qty"] = restored.values

    merged = merged.drop(columns=["_orig_stock_qty"], errors="ignore")

    # Reorder columns to match the original timeline when possible.
    desired_cols = list(timeline_copy.columns)
    remaining_cols = [c for c in merged.columns if c not in desired_cols]
    ordered_cols = desired_cols + remaining_cols
    merged = merged[[c for c in ordered_cols if c in merged.columns]]

    return merged
