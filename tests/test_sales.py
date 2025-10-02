import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scm_dashboard_v4.sales import prepare_amazon_sales_series


def test_prepare_amazon_sales_series_continuous_index():
    start = pd.Timestamp("2024-01-01")
    end = pd.Timestamp("2024-01-05")
    raw = pd.DataFrame(
        {
            "date": [
                "2024-01-01",
                "2024-01-03",
                "2024-01-01",
                "2024-01-04",
            ],
            "center": ["AMZUS", "AMZUS", "AMZUS", "AMZUS"],
            "resource_code": ["SKU1", "SKU1", "SKU2", "SKU2"],
            "stock_qty": [100, 70, 50, 40],
        }
    )

    result = prepare_amazon_sales_series(raw, ["SKU1", "SKU2"], start, end)
    series = result.data

    expected_dates = pd.date_range(start, end, freq="D")
    pd.testing.assert_index_equal(pd.Index(series["date"]), expected_dates, check_names=False)

    # SKU1 drops 30 units on Jan-03; SKU2 drops 10 units on Jan-04.
    sales_map = dict(zip(series["date"], series["sales_qty"]))
    assert sales_map[pd.Timestamp("2024-01-03")] == 30
    assert sales_map[pd.Timestamp("2024-01-04")] == 10

    # Inventory line should forward-fill across sparse days.
    inventory_map = dict(zip(series["date"], series["inventory_qty"]))
    assert inventory_map[pd.Timestamp("2024-01-02")] == 150
    assert inventory_map[pd.Timestamp("2024-01-05")] == 110

    # Rolling mean is always non-negative and defined for the first date.
    assert series.loc[series.index[0], "sales_roll_mean"] == 0
    assert (series["sales_roll_mean"] >= 0).all()
