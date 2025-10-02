import pandas as pd

from scm_dashboard_v4.timeline import build_timeline as build_timeline_v4
from scm_dashboard_v5 import BuildInputs, build_timeline_bundle


def _make_sample_inputs():
    snapshot = pd.DataFrame(
        [
            {"snapshot_date": "2024-01-01", "center": "A", "resource_code": "S1", "stock_qty": 100},
            {"snapshot_date": "2024-01-02", "center": "A", "resource_code": "S1", "stock_qty": 90},
            {"snapshot_date": "2024-01-01", "center": "B", "resource_code": "S1", "stock_qty": 50},
        ]
    )
    moves = pd.DataFrame(
        [
            {
                "resource_code": "S1",
                "qty_ea": 20,
                "from_center": "A",
                "to_center": "B",
                "onboard_date": "2024-01-04",
                "carrier_mode": "SEA",
            },
            {
                "resource_code": "S1",
                "qty_ea": 30,
                "from_center": "B",
                "to_center": "A",
                "onboard_date": "2024-01-03",
                "arrival_date": "2024-01-05",
                "carrier_mode": "AIR",
            },
            {
                "resource_code": "S1",
                "qty_ea": 10,
                "from_center": "Factory",
                "to_center": "A",
                "onboard_date": "2024-01-02",
                "event_date": "2024-01-04",
                "carrier_mode": "WIP",
            },
        ]
    )
    return BuildInputs(snapshot=snapshot, moves=moves)


def test_bundle_matches_v4_timeline_output():
    inputs = _make_sample_inputs()
    centers = ["A", "B"]
    skus = ["S1"]
    start = pd.Timestamp("2024-01-01")
    end = pd.Timestamp("2024-01-05")
    today = pd.Timestamp("2024-01-02")

    bundle = build_timeline_bundle(
        inputs,
        centers=centers,
        skus=skus,
        start=start,
        end=end,
        today=today,
        lag_days=2,
    )
    v5_timeline = bundle.concat()

    v4_timeline = build_timeline_v4(
        inputs.snapshot,
        inputs.moves,
        centers_sel=centers,
        skus_sel=skus,
        start_dt=start,
        horizon_end=end,
        today=today,
        lag_days=2,
    )

    pd.testing.assert_frame_equal(
        v5_timeline.sort_values(["center", "resource_code", "date"]).reset_index(drop=True),
        v4_timeline.sort_values(["center", "resource_code", "date"]).reset_index(drop=True),
    )


def test_bundle_exposes_component_frames():
    inputs = _make_sample_inputs()
    bundle = build_timeline_bundle(
        inputs,
        centers=["A", "B"],
        skus=["S1"],
        start=pd.Timestamp("2024-01-01"),
        end=pd.Timestamp("2024-01-05"),
        today=pd.Timestamp("2024-01-02"),
        lag_days=2,
    )

    assert set(bundle.center_lines["center"].unique()) == {"A", "B"}
    assert set(bundle.in_transit_lines["center"].unique()) == {"In-Transit"}
    assert set(bundle.wip_lines["center"].unique()) == {"WIP"}
