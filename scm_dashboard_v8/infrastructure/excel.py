from __future__ import annotations

from typing import Any

from scm_dashboard_v4.loaders import load_from_excel
from scm_dashboard_v4.processing import (
    load_wip_from_incoming,
    normalize_moves,
    normalize_refined_snapshot,
)

from .models import DashboardSourceData


def load_dashboard_data(file: Any) -> DashboardSourceData:
    """Load dashboard data from an Excel file and return normalized DataFrames."""

    df_move, df_ref, df_incoming, snapshot_raw = load_from_excel(file)

    moves = normalize_moves(df_move)
    snapshot = normalize_refined_snapshot(df_ref)
    wip = load_wip_from_incoming(df_incoming)

    return DashboardSourceData(
        moves=moves,
        snapshot=snapshot,
        wip=wip,
        snapshot_raw=snapshot_raw,
    )
