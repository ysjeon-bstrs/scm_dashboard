"""End-to-end regression checks for the v8 application pipeline."""

from __future__ import annotations

import pandas as pd

from scm_dashboard_v5.analytics import kpi_breakdown_per_sku
from scm_dashboard_v5.forecast import (
    apply_consumption_with_events,
    build_amazon_forecast_context,
)
from scm_dashboard_v5.planning.timeline import TimelineContext, prepare_moves
from scm_dashboard_v8.application.pipeline import BuildInputs, build_timeline_bundle

from . import fixtures
from .utils import assert_frame_equivalent, load_golden_csv, normalise_dataframe


SORT_KEY = ["center", "resource_code", "date"]


def _build_v8_timeline_raw() -> pd.DataFrame:
    snapshot = fixtures.load_sample_snapshot()
    moves = fixtures.load_sample_moves()
    inputs = BuildInputs(snapshot=snapshot, moves=moves)
    bundle = build_timeline_bundle(
        inputs,
        centers=fixtures.DEFAULT_CENTERS,
        skus=fixtures.DEFAULT_SKUS,
        start=fixtures.DEFAULT_START,
        end=fixtures.DEFAULT_END,
        today=fixtures.DEFAULT_TODAY,
        lag_days=fixtures.DEFAULT_LAG_DAYS,
    )
    return bundle.concat()


def _build_v8_timeline() -> pd.DataFrame:
    return normalise_dataframe(_build_v8_timeline_raw(), sort_by=SORT_KEY)


def test_timeline_matches_golden() -> None:
    actual = _build_v8_timeline()
    expected = load_golden_csv("timeline_actual.csv")
    assert_frame_equivalent(
        actual,
        expected,
        sort_by=SORT_KEY,
        label="timeline_actual.csv",
    )


def test_consumption_projection_matches_golden() -> None:
    timeline = _build_v8_timeline_raw()
    snapshot_long = fixtures.load_sample_snapshot_long()
    projected = apply_consumption_with_events(
        timeline,
        snapshot_long,
        centers=fixtures.DEFAULT_CENTERS,
        skus=fixtures.DEFAULT_SKUS,
        start=fixtures.DEFAULT_START,
        end=fixtures.DEFAULT_END,
        lookback_days=fixtures.DEFAULT_LOOKBACK_DAYS,
    )
    actual = normalise_dataframe(projected, sort_by=SORT_KEY)
    expected = load_golden_csv("timeline_with_consumption.csv")
    assert_frame_equivalent(
        actual,
        expected,
        sort_by=SORT_KEY,
        label="timeline_with_consumption.csv",
    )


def test_kpi_breakdown_matches_golden() -> None:
    snapshot = fixtures.load_sample_snapshot()
    moves = fixtures.load_sample_moves()
    latest_snapshot = pd.to_datetime(snapshot["snapshot_date"]).max().normalize()
    kpi_raw = kpi_breakdown_per_sku(
        snapshot,
        moves,
        fixtures.DEFAULT_CENTERS,
        fixtures.DEFAULT_SKUS,
        fixtures.DEFAULT_TODAY,
        "snapshot_date",
        latest_snapshot,
        fixtures.DEFAULT_LAG_DAYS,
    )
    actual = kpi_raw.reset_index().melt(
        id_vars=["resource_code"], var_name="metric", value_name="value"
    )
    expected = load_golden_csv("kpi_breakdown.csv")
    assert_frame_equivalent(
        normalise_dataframe(
            actual,
            sort_by=["resource_code", "metric"],
            round_columns=("value",),
        ),
        expected,
        sort_by=["resource_code", "metric"],
        label="kpi_breakdown.csv",
    )


def test_amazon_context_matches_golden() -> None:
    context = build_amazon_forecast_context(
        snap_long=fixtures.load_sample_snapshot_long(),
        moves=fixtures.load_sample_moves(),
        snapshot_raw=fixtures.load_sample_snapshot_raw(),
        centers=("AMZUS",),
        skus=fixtures.DEFAULT_SKUS,
        start=fixtures.DEFAULT_START,
        end=fixtures.DEFAULT_END,
        today=fixtures.DEFAULT_TODAY,
        lookback_days=fixtures.DEFAULT_LOOKBACK_DAYS,
        promotion_events=[{"start": "2024-01-05", "end": "2024-01-06", "uplift": 0.25}],
    )

    comparisons = {
        "amazon_inventory_actual.csv": context.inv_actual,
        "amazon_inventory_forecast.csv": context.inv_forecast,
        "amazon_sales_history.csv": context.sales_hist,
        "amazon_sales_ma7.csv": context.sales_ma7,
        "amazon_sales_forecast.csv": context.sales_forecast,
    }

    for filename, frame in comparisons.items():
        actual = normalise_dataframe(frame, sort_by=SORT_KEY)
        expected = load_golden_csv(filename)
        assert_frame_equivalent(
            actual,
            expected,
            sort_by=SORT_KEY,
            label=filename,
        )

    timeline_context = TimelineContext(
        centers=fixtures.DEFAULT_CENTERS,
        skus=fixtures.DEFAULT_SKUS,
        start=fixtures.DEFAULT_START,
        end=fixtures.DEFAULT_END,
        today=fixtures.DEFAULT_TODAY,
        lag_days=fixtures.DEFAULT_LAG_DAYS,
        horizon_days=0,
    )
    move_table = prepare_moves(
        fixtures.load_sample_moves(),
        context=timeline_context,
        fallback_days=1,
    )
    inbound = (
        move_table.data[move_table.data["to_center"].isin(["AMZUS"])]
        [["resource_code", "to_center", "qty_ea", "pred_inbound_date"]]
        .rename(columns={"pred_inbound_date": "date"})
    )
    inbound_expected = load_golden_csv("amazon_inbound.csv")
    assert_frame_equivalent(
        normalise_dataframe(
            inbound,
            sort_by=["resource_code", "date", "to_center"],
            round_columns=("qty_ea",),
            int_columns=("qty_ea",),
        ),
        inbound_expected,
        sort_by=["resource_code", "date", "to_center"],
        label="amazon_inbound.csv",
    )

