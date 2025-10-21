"""KPI adapters that map to the existing v4 KPI calculators."""

from __future__ import annotations

from typing import Sequence

import pandas as pd

from scm_dashboard_v4 import kpi as v4_kpi


def kpi_breakdown_per_sku(
    snapshot: pd.DataFrame,
    moves: pd.DataFrame,
    centers: Sequence[str],
    skus: Sequence[str],
    today: pd.Timestamp,
    snapshot_date_col: str,
    latest_snapshot: pd.Timestamp,
    lag_days: int,
) -> pd.DataFrame:
    """Delegate KPI calculations to the v4 implementation."""

    return v4_kpi.kpi_breakdown_per_sku(
        snapshot,
        moves,
        list(centers),
        list(skus),
        today,
        snapshot_date_col,
        latest_snapshot,
        lag_days,
    )
