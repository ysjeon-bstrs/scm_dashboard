"""Regression tests for in-transit inventory timeline generation."""

import pandas as pd
import pytest

from scm_dashboard_v4.timeline import (
    normalize_move_dates,
    annotate_move_schedule,
    compute_in_transit_series,
)


def test_in_transit_aggregates_multiple_shipments():
    moves = pd.DataFrame(
        {
            "resource_code": ["SKU-1", "SKU-1", "SKU-1"],
            "qty_ea": [10, 5, 3],
            "carrier_mode": ["AIR", "OCEAN", "AIR"],
            "from_center": ["FC1", "FC1", "FC2"],
            "to_center": ["C1", "C1", "C1"],
            "onboard_date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "arrival_date": ["2024-01-03", "2024-01-05", "2024-01-04"],
        }
    )

    start = pd.Timestamp("2024-01-01")
    horizon_end = pd.Timestamp("2024-01-10")
    today = pd.Timestamp("2024-01-05")

    in_transit = compute_in_transit_series(
        moves,
        centers_sel=["C1"],
        skus_sel=["SKU-1"],
        start_dt=start,
        horizon_end=horizon_end,
        today=today,
    )

    assert not in_transit.empty
    assert (in_transit["center"] == "In-Transit").all()

    ts = (
        in_transit[in_transit["resource_code"] == "SKU-1"]
        .sort_values("date")
        .set_index("date")["stock_qty"]
    )

    # Shipments accumulate on board dates
    assert ts.loc["2024-01-01"] >= 10
    assert ts.loc["2024-01-02"] >= 15
    assert ts.loc["2024-01-03"] >= 18


def test_in_transit_synchronises_with_annotate_dates():
    """Verify that in-transit decrements match pred_inbound_date from annotate_move_schedule."""
    moves = pd.DataFrame(
        {
            "resource_code": ["SKU-1", "SKU-1"],
            "qty_ea": [10, 5],
            "carrier_mode": ["AIR", "AIR"],
            "from_center": ["FC1", "FC1"],
            "to_center": ["C1", "C1"],
            "onboard_date": ["2024-01-01", "2024-01-02"],
            "arrival_date": ["2024-01-03", "2024-01-05"],
            "inbound_date": ["2024-01-04", pd.NaT],
        }
    )

    start = pd.Timestamp("2024-01-01")
    horizon_end = pd.Timestamp("2024-01-10")
    today = pd.Timestamp("2024-01-06")
    lag_days = 1

    prepared = normalize_move_dates(moves)
    prepared = annotate_move_schedule(prepared, today, lag_days, horizon_end)

    in_transit = compute_in_transit_series(
        moves, ["C1"], ["SKU-1"], start, horizon_end, today, lag_days
    )

    ts = (
        in_transit[in_transit["resource_code"] == "SKU-1"]
        .sort_values("date")
        .set_index("date")["stock_qty"]
    )

    # Check that deltas match predicted inbound dates
    deltas = ts.diff().fillna(ts.iloc[0])

    onboard_field = "_onboard_date_actual" if "_onboard_date_actual" in prepared.columns else "onboard_date"
    onboard_events = (
        prepared[prepared["to_center"].astype(str) == "C1"]
        .groupby(onboard_field)["qty_ea"].sum()
    )

    for event_date, qty in onboard_events.items():
        if event_date < start or event_date > horizon_end:
            continue
        assert deltas.loc[event_date] == qty

    carry_expected = prepared[
        (prepared[onboard_field] <= start)
        & (prepared["in_transit_end_date"] > start)
        & (prepared["to_center"].astype(str) == "C1")
    ]["qty_ea"].sum()

    assert ts.iloc[0] == carry_expected


def test_in_transit_sparse_period_covers_entire_range():
    moves = pd.DataFrame(
        {
            "resource_code": ["SKU-1", "SKU-1"],
            "qty_ea": [10, 4],
            "carrier_mode": ["AIR", "AIR"],
            "from_center": ["FC1", "FC1"],
            "to_center": ["C1", "C1"],
            "onboard_date": ["2023-12-30", "2024-01-15"],
            "arrival_date": ["2024-01-02", "2024-01-22"],
            "inbound_date": ["2024-01-04", pd.NaT],
        }
    )

    start = pd.Timestamp("2024-01-01")
    horizon_end = pd.Timestamp("2024-01-31")
    today = pd.Timestamp("2024-01-20")

    in_transit = compute_in_transit_series(
        moves,
        centers_sel=["C1"],
        skus_sel=["SKU-1"],
        start_dt=start,
        horizon_end=horizon_end,
        today=today,
    )

    assert not in_transit.empty

    ts = (
        in_transit[in_transit["resource_code"] == "SKU-1"]
        .sort_values("date")
        .reset_index(drop=True)
    )

    full_idx = pd.date_range(start, horizon_end, freq="D")

    assert len(ts) == len(full_idx)
    assert ts["date"].tolist() == list(full_idx)
    # Even when no events occur for several days the line should retain the
    # last value (including zero) instead of dropping missing dates.
    assert ts["stock_qty"].isna().sum() == 0