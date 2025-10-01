import pathlib
import sys

import pandas as pd

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from scm_dashboard_v4.timeline import (  # noqa: E402
    annotate_move_schedule,
    compute_in_transit_series,
    normalize_move_dates,
)


def test_in_transit_steps_match_center_policy():
    moves = pd.DataFrame(
        {
            "resource_code": ["SKU-1", "SKU-1", "SKU-1"],
            "qty_ea": [10, 5, 4],
            "carrier_mode": ["AIR", "AIR", "AIR"],
            "from_center": ["FC1", "FC1", "FC2"],
            "to_center": ["C1", "C1", "C1"],
            "onboard_date": ["2023-12-28", "2024-01-03", "2024-01-06"],
            "arrival_date": ["2023-12-30", "2024-01-08", "2024-01-09"],
            "inbound_date": ["2024-01-05", pd.NaT, pd.NaT],
        }
    )

    today = pd.Timestamp("2024-01-04")
    start = pd.Timestamp("2023-12-30")
    horizon_end = pd.Timestamp("2024-01-12")
    lag_days = 2

    prepared = normalize_move_dates(moves)
    prepared = annotate_move_schedule(prepared, today, lag_days, horizon_end)

    in_transit = compute_in_transit_series(
        prepared,
        centers_sel=["C1"],
        skus_sel=["SKU-1"],
        start_dt=start,
        horizon_end=horizon_end,
        today=today,
        lag_days=lag_days,
    )

    assert not in_transit.empty

    ts = (
        in_transit.set_index("date")["stock_qty"].sort_index().astype(int)
    )

    deltas = ts.diff().fillna(ts.iloc[0]).astype(int)

    center_events = (
        prepared[(prepared["to_center"].astype(str) == "C1") & (prepared["carrier_mode"] != "WIP")]
        .groupby("pred_inbound_date")["qty_ea"].sum()
    )

    for event_date, qty in center_events.items():
        if event_date < start or event_date > horizon_end:
            continue
        assert deltas.loc[event_date] == -qty

    onboard_events = (
        prepared[prepared["to_center"].astype(str) == "C1"]
        .groupby("onboard_date")["qty_ea"].sum()
    )

    for event_date, qty in onboard_events.items():
        if event_date < start or event_date > horizon_end:
            continue
        assert deltas.loc[event_date] == qty

    carry_expected = prepared[
        (prepared["onboard_date"] < start)
        & (prepared["in_transit_end_date"] > start)
        & (prepared["to_center"].astype(str) == "C1")
    ]["qty_ea"].sum()

    assert ts.iloc[0] == carry_expected
