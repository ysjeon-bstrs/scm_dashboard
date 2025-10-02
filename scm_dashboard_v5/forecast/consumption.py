"""Forecasting adapters that reuse the proven v4 consumption logic."""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from scm_dashboard_v4 import consumption as v4_consumption


def estimate_daily_consumption(sales: pd.DataFrame, *, window: int = 28) -> pd.DataFrame:
    """Delegate to the stable v4 estimator while keeping the new namespace."""

    return v4_consumption.estimate_daily_consumption(sales, window=window)


def apply_consumption_with_events(
    timeline: pd.DataFrame,
    events: pd.DataFrame,
    *,
    centers: Iterable[str],
    skus: Iterable[str],
) -> pd.DataFrame:
    return v4_consumption.apply_consumption_with_events(
        timeline,
        events,
        centers=list(centers),
        skus=list(skus),
    )
