import pandas as pd

from scm_dashboard_v6.features.timeline import render_timeline_section
from scm_dashboard_v6.features.inventory_view import render_inventory_pivot
from scm_dashboard_v6.features.amazon import render_amazon_panel


def test_v6_timeline_smoke():
    snapshot = pd.DataFrame([
        {"date": "2025-01-01", "center": "AMZUS", "resource_code": "SKU1", "stock_qty": 10},
        {"date": "2025-01-02", "center": "AMZUS", "resource_code": "SKU1", "stock_qty": 9},
    ])
    snapshot["date"] = pd.to_datetime(snapshot["date"]).dt.normalize()
    moves = pd.DataFrame()
    out = render_timeline_section(
        snapshot=snapshot,
        moves=moves,
        centers=["AMZUS"],
        skus=["SKU1"],
        start=pd.Timestamp("2025-01-01"),
        end=pd.Timestamp("2025-01-10"),
        today=pd.Timestamp("2025-01-03"),
        lookback_days=28,
        show_production=False,
        show_in_transit=False,
    )
    assert out is not None


def test_v6_inventory_smoke():
    snapshot = pd.DataFrame([
        {"date": "2025-01-01", "center": "AMZUS", "resource_code": "S1", "stock_qty": 5},
    ])
    snapshot["date"] = pd.to_datetime(snapshot["date"]).dt.normalize()
    df = render_inventory_pivot(snapshot=snapshot, centers=["AMZUS"], latest_snapshot=pd.Timestamp("2025-01-01"))
    assert df is not None


def test_v6_amazon_smoke():
    snapshot = pd.DataFrame([
        {"date": "2025-01-01", "center": "AMZUS", "resource_code": "S1", "stock_qty": 5},
    ])
    snapshot["date"] = pd.to_datetime(snapshot["date"]).dt.normalize()
    moves = pd.DataFrame()
    render_amazon_panel(
        snapshot_long=snapshot,
        moves=moves,
        snapshot_raw=pd.DataFrame(),
        centers=["AMZUS"],
        skus=["S1"],
        start=pd.Timestamp("2025-01-01"),
        end=pd.Timestamp("2025-01-10"),
        today=pd.Timestamp("2025-01-03"),
        lookback_days=28,
        promotion_events=None,
        use_consumption_forecast=False,
    )
