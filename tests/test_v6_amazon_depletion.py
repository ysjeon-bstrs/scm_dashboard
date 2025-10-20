import pandas as pd

from scm_dashboard_v6.forecast.sales_from_inventory import derive_future_sales_from_inventory


def test_depletion_clipping_stops_sales_after_zero():
    # 재고가 3일차에 0이 되는 경우, 그 이후 판매는 0으로 잘리는지 확인
    idx = pd.date_range("2025-01-01", "2025-01-06", freq="D")
    inv_actual = pd.DataFrame({
        "date": [pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-02")],
        "center": ["AMZUS", "AMZUS"],
        "resource_code": ["S1", "S1"],
        "stock_qty": [5, 4],
    })
    inv_forecast = pd.DataFrame({
        "date": idx,
        "center": ["AMZUS"] * len(idx),
        "resource_code": ["S1"] * len(idx),
        "stock_qty": [3, 2, 0, 0, 0, 0],
    })

    sales = derive_future_sales_from_inventory(
        inv_actual,
        inv_forecast,
        centers=["AMZUS"],
        skus=["S1"],
        start=idx[0],
        end=idx[-1],
        today=pd.Timestamp("2025-01-01"),
    )

    assert not sales.empty
    after_zero = sales[sales["date"] > pd.Timestamp("2025-01-03")]
    assert (after_zero["sales_ea"] == 0).all()


