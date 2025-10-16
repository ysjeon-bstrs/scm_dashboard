import pandas as pd

from scm_dashboard_v5.forecast import apply_consumption_with_events


def test_consumption_applies_after_cons_start():
    # cons_start 이후 재고가 단조 감소(클램프 0)하는지 확인
    timeline = pd.DataFrame([
        {"date": "2025-01-01", "center": "AMZUS", "resource_code": "S1", "stock_qty": 10},
        {"date": "2025-01-02", "center": "AMZUS", "resource_code": "S1", "stock_qty": 10},
        {"date": "2025-01-03", "center": "AMZUS", "resource_code": "S1", "stock_qty": 10},
        {"date": "2025-01-04", "center": "AMZUS", "resource_code": "S1", "stock_qty": 10},
    ])
    timeline["date"] = pd.to_datetime(timeline["date"]).dt.normalize()

    snapshot = pd.DataFrame([
        {"date": "2024-12-31", "center": "AMZUS", "resource_code": "S1", "stock_qty": 10}
    ])

    out = apply_consumption_with_events(
        timeline,
        snapshot,
        centers=["AMZUS"],
        skus=["S1"],
        start=pd.Timestamp("2025-01-01"),
        end=pd.Timestamp("2025-01-04"),
        lookback_days=7,
        events=[],
        cons_start=pd.Timestamp("2025-01-02"),
    )

    s = out[out["resource_code"] == "S1"].sort_values("date")["stock_qty"].tolist()
    assert s[1] >= s[2] >= s[3]  # 1일차 이후 단조 감소
