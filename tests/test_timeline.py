import pandas as pd

from scm_dashboard_v4.timeline import build_timeline


def _timeline_to_series(timeline: pd.DataFrame, center: str, sku: str) -> pd.Series:
    filtered = timeline[(timeline["center"] == center) & (timeline["resource_code"] == sku)].copy()
    if filtered.empty:
        return pd.Series(dtype=int)
    return filtered.set_index("date")["stock_qty"].sort_index()


def test_build_timeline_combines_center_transit_and_wip_lines():
    snap_long = pd.DataFrame(
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

    timeline = build_timeline(
        snap_long,
        moves,
        centers_sel=["A", "B"],
        skus_sel=["S1"],
        start_dt=pd.Timestamp("2024-01-01"),
        horizon_end=pd.Timestamp("2024-01-05"),
        today=pd.Timestamp("2024-01-02"),
        lag_days=2,
    )

    center_series = _timeline_to_series(timeline, "A", "S1")
    assert center_series.loc[pd.Timestamp("2024-01-04")] == 80
    assert center_series.loc[pd.Timestamp("2024-01-05")] == 110

    in_transit_series = _timeline_to_series(timeline, "In-Transit", "S1")
    assert in_transit_series.loc[pd.Timestamp("2024-01-04")] == 30
    assert in_transit_series.loc[pd.Timestamp("2024-01-05")] == 0

    wip_series = _timeline_to_series(timeline, "WIP", "S1")
    assert wip_series.loc[pd.Timestamp("2024-01-03")] == 10
    assert wip_series.loc[pd.Timestamp("2024-01-04")] == 0


def test_build_timeline_uses_fallback_for_missing_arrival():
    snap_long = pd.DataFrame(
        [
            {"snapshot_date": "2024-01-01", "center": "C", "resource_code": "SKU", "stock_qty": 0},
        ]
    )

    moves = pd.DataFrame(
        [
            {
                "resource_code": "SKU",
                "qty_ea": 5,
                "from_center": "X",
                "to_center": "C",
                "onboard_date": "2024-01-01",
                "carrier_mode": "TRUCK",
            }
        ]
    )

    timeline = build_timeline(
        snap_long,
        moves,
        centers_sel=["C"],
        skus_sel=["SKU"],
        start_dt=pd.Timestamp("2024-01-01"),
        horizon_end=pd.Timestamp("2024-01-03"),
        today=pd.Timestamp("2024-01-01"),
        lag_days=1,
    )

    center_series = _timeline_to_series(timeline, "C", "SKU")
    assert center_series.loc[pd.Timestamp("2024-01-02")] == 5

    in_transit_series = _timeline_to_series(timeline, "In-Transit", "SKU")
    assert in_transit_series.loc[pd.Timestamp("2024-01-01")] == 5
    assert in_transit_series.loc[pd.Timestamp("2024-01-02")] == 0
