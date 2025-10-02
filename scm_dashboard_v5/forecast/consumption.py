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
) -> pd.DataFrame:
    """Delegate to the v4 helper while normalising parameter names."""

    return v4_consumption.apply_consumption_with_events(
        timeline,
        snapshot,
        list(centers),
        list(skus),
        pd.to_datetime(start).normalize(),
        pd.to_datetime(end).normalize(),
        int(lookback_days),
        list(events) if events else None,
    )
