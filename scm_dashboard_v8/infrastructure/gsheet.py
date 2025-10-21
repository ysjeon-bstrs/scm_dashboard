from __future__ import annotations

from scm_dashboard_v4.loaders import load_from_gsheet_api, load_snapshot_raw
from scm_dashboard_v4.processing import (
    load_wip_from_incoming,
    normalize_moves,
    normalize_refined_snapshot,
)

from .models import DashboardSourceData


def load_dashboard_data() -> DashboardSourceData:
    """Load dashboard data from Google Sheets and return normalized DataFrames."""

    df_move, df_ref, df_incoming = load_from_gsheet_api()

    moves = normalize_moves(df_move)
    snapshot = normalize_refined_snapshot(df_ref)
    wip = load_wip_from_incoming(df_incoming)
    snapshot_raw = load_snapshot_raw()

    return DashboardSourceData(
        moves=moves,
        snapshot=snapshot,
        wip=wip,
        snapshot_raw=snapshot_raw,
    )
