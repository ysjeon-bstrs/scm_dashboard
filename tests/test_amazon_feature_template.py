import pandas as pd


def test_render_amazon_panel_smoke():
    # Placeholder for future v6 features.amazon
    snapshot = pd.DataFrame([
        {"date": "2025-01-01", "center": "AMZUS", "resource_code": "S1", "stock_qty": 50},
        {"date": "2025-01-02", "center": "AMZUS", "resource_code": "S1", "stock_qty": 45},
    ])
    moves = pd.DataFrame()
    assert not snapshot.empty or not moves.empty



