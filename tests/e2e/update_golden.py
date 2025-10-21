"""Utility script for refreshing golden fixtures used by the e2e tests."""

from __future__ import annotations

import pandas as pd

from scm_dashboard_v5.analytics import kpi_breakdown_per_sku
from scm_dashboard_v5.core import build_timeline as build_core_timeline
from scm_dashboard_v5.forecast import (
    apply_consumption_with_events,
    build_amazon_forecast_context,
)
from scm_dashboard_v5.planning.timeline import TimelineContext, prepare_moves

from . import fixtures
from .utils import ensure_golden_dir, normalise_dataframe


def regenerate() -> None:
    """Recompute golden CSV fixtures from the v5 pipeline."""

    snapshot = fixtures.load_sample_snapshot()
    moves = fixtures.load_sample_moves()
    snapshot_long = fixtures.load_sample_snapshot_long()
    snapshot_raw = fixtures.load_sample_snapshot_raw()

    centers = fixtures.DEFAULT_CENTERS
    skus = fixtures.DEFAULT_SKUS
    start = fixtures.DEFAULT_START
    end = fixtures.DEFAULT_END
    today = fixtures.DEFAULT_TODAY
    lookback = fixtures.DEFAULT_LOOKBACK_DAYS
    lag_days = fixtures.DEFAULT_LAG_DAYS

    timeline_actual = build_core_timeline(
        snapshot,
        moves,
        centers=centers,
        skus=skus,
        start=start,
        end=end,
        today=today,
        lag_days=lag_days,
    )

    timeline_with_consumption = apply_consumption_with_events(
        timeline_actual,
        snapshot_long,
        centers=centers,
        skus=skus,
        start=start,
        end=end,
        lookback_days=lookback,
    )

    latest_snapshot = pd.to_datetime(snapshot["snapshot_date"]).max().normalize()

    kpi_raw = kpi_breakdown_per_sku(
        snapshot,
        moves,
        centers,
        skus,
        today,
        "snapshot_date",
        latest_snapshot,
        lag_days,
    )
    kpi = kpi_raw.reset_index().melt(
        id_vars=["resource_code"], var_name="metric", value_name="value"
    )

    amazon_ctx = build_amazon_forecast_context(
        snap_long=snapshot_long,
        moves=moves,
        snapshot_raw=snapshot_raw,
        centers=("AMZUS",),
        skus=skus,
        start=start,
        end=end,
        today=today,
        lookback_days=lookback,
        promotion_events=[{"start": "2024-01-05", "end": "2024-01-06", "uplift": 0.25}],
    )

    timeline_context = TimelineContext(
        centers=centers,
        skus=skus,
        start=start,
        end=end,
        today=today,
        lag_days=lag_days,
        horizon_days=0,
    )
    move_table = prepare_moves(moves, context=timeline_context, fallback_days=1)
    inbound = (
        move_table.data[move_table.data["to_center"].isin(["AMZUS"])]
        [["resource_code", "to_center", "qty_ea", "pred_inbound_date"]]
        .rename(columns={"pred_inbound_date": "date"})
    )

    out_dir = ensure_golden_dir()
    frames = {
        "timeline_actual.csv": (timeline_actual, ["center", "resource_code", "date"], None),
        "timeline_with_consumption.csv": (
            timeline_with_consumption,
            ["center", "resource_code", "date"],
            None,
        ),
        "kpi_breakdown.csv": (kpi, ["resource_code", "metric"], ("value",)),
        "amazon_inventory_actual.csv": (
            amazon_ctx.inv_actual,
            ["center", "resource_code", "date"],
            None,
        ),
        "amazon_inventory_forecast.csv": (
            amazon_ctx.inv_forecast,
            ["center", "resource_code", "date"],
            None,
        ),
        "amazon_sales_history.csv": (
            amazon_ctx.sales_hist,
            ["center", "resource_code", "date"],
            ("sales_ea",),
        ),
        "amazon_sales_ma7.csv": (
            amazon_ctx.sales_ma7,
            ["center", "resource_code", "date"],
            ("sales_ea",),
        ),
        "amazon_sales_forecast.csv": (
            amazon_ctx.sales_forecast,
            ["center", "resource_code", "date"],
            ("sales_ea",),
        ),
        "amazon_inbound.csv": (
            inbound,
            ["resource_code", "date", "to_center"],
            ("qty_ea",),
        ),
    }

    for name, (frame, sort_cols, round_cols) in frames.items():
        normalised = normalise_dataframe(frame, sort_by=sort_cols, round_columns=round_cols)
        normalised.to_csv(out_dir / name, index=False)


if __name__ == "__main__":
    regenerate()

