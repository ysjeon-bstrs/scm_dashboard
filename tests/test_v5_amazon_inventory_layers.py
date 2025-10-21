import pandas as pd

from scm_dashboard_v5.analytics.sales import (
    prepare_amazon_inventory_layers as prepare_amazon_inventory_layers_v5,
)
from scm_dashboard_v8.analytics.sales import prepare_amazon_inventory_layers


def test_v5_sales_module_aliases_v8():
    assert prepare_amazon_inventory_layers is prepare_amazon_inventory_layers_v5


def test_prepare_amazon_inventory_layers_returns_aligned_series():
    timeline = pd.DataFrame(
        [
            {"date": "2023-10-01", "center": "AMZUS", "resource_code": "SKU1", "stock_qty": 100},
            {"date": "2023-10-01", "center": "AMZUS", "resource_code": "SKU2", "stock_qty": 50},
            {"date": "2023-10-02", "center": "AMZUS", "resource_code": "SKU1", "stock_qty": 90},
            {"date": "2023-10-02", "center": "AMZUS", "resource_code": "SKU2", "stock_qty": 60},
            {"date": "2023-10-03", "center": "AMZUS", "resource_code": "SKU1", "stock_qty": 70},
            {"date": "2023-10-03", "center": "AMZUS", "resource_code": "SKU2", "stock_qty": 40},
        ]
    )

    forecast_timeline = pd.DataFrame(
        [
            {"date": "2023-10-01", "center": "AMZUS", "resource_code": "SKU1", "stock_qty": 100},
            {"date": "2023-10-01", "center": "AMZUS", "resource_code": "SKU2", "stock_qty": 50},
            {"date": "2023-10-04", "center": "AMZUS", "resource_code": "SKU1", "stock_qty": 60},
            {"date": "2023-10-04", "center": "AMZUS", "resource_code": "SKU2", "stock_qty": 30},
            {"date": "2023-10-05", "center": "AMZUS", "resource_code": "SKU1", "stock_qty": 40},
            {"date": "2023-10-05", "center": "AMZUS", "resource_code": "SKU2", "stock_qty": 20},
        ]
    )

    moves = pd.DataFrame(
        [
            {"inbound_date": "2023-10-02", "to_center": "AMZUS", "resource_code": "SKU1", "qty_ea": 20},
            {"inbound_date": "2023-10-05", "to_center": "AMZUS", "resource_code": "SKU2", "qty_ea": 10},
        ]
    )

    start = pd.Timestamp("2023-10-01")
    end = pd.Timestamp("2023-10-05")
    latest_snapshot = pd.Timestamp("2023-10-03")

    result = prepare_amazon_inventory_layers(
        timeline,
        centers=["AMZUS"],
        skus=["SKU1", "SKU2"],
        start_dt=start,
        end_dt=end,
        forecast_timeline=forecast_timeline,
        moves=moves,
        latest_snapshot=latest_snapshot,
    )

    expected_index = pd.date_range(start, end, freq="D")
    assert list(result.inventory.index) == list(expected_index)
    assert result.inventory.tolist() == [150.0, 150.0, 110.0, 110.0, 110.0]

    assert result.sales.tolist() == [0.0, 0.0, 40.0, 0.0, 0.0]

    assert result.forecast is not None
    forecast = result.forecast.dropna()
    assert list(forecast.index) == [pd.Timestamp("2023-10-03"), pd.Timestamp("2023-10-04"), pd.Timestamp("2023-10-05")]
    assert forecast.tolist() == [110.0, 90.0, 60.0]

    assert result.inbound is not None
    inbound = result.inbound.reindex(expected_index).fillna(0.0)
    assert inbound.tolist() == [0.0, 20.0, 0.0, 0.0, 10.0]
