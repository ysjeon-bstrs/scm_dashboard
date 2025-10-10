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
