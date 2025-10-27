"""재고 및 판매 예측 함수.

재고 소진을 고려한 판매 예측 로직을 제공합니다.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .inventory_helpers import (
    calculate_baseline_rates,
    combine_forecast_results,
    simulate_inventory_sales_for_group,
    validate_and_prepare_forecast_inputs,
)
from .sales import make_forecast_sales_capped


def forecast_sales_and_inventory(
    daily_sales: pd.DataFrame,
    timeline_center: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    lookback_days: int = 28,
    uplift_events: Optional[Iterable[dict]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate sales and inventory simultaneously for Amazon centres.

    The simulator ensures that forecast sales never exceed the available
    inventory (including inbound events present on the centre timeline).  Once
    the simulated stock reaches zero, the sales forecast is automatically
    capped at zero, keeping the OOS timing aligned between charts and KPIs.
    """
    # 입력 검증 및 준비
    (
        sales_history,
        timeline,
        center_set,
        sku_set,
        index,
        start_norm,
        end_norm,
    ) = validate_and_prepare_forecast_inputs(daily_sales, timeline_center, start, end)

    if not center_set or not sku_set or index.empty:
        empty_sales = pd.DataFrame(
            columns=["date", "center", "resource_code", "sales_ea"]
        )
        empty_inv = pd.DataFrame(
            columns=["date", "center", "resource_code", "stock_qty"]
        )
        return empty_sales, empty_inv

    # Baseline rate 및 uplift 계산
    mean_rate, uplift = calculate_baseline_rates(
        sales_history, center_set, sku_set, index, lookback_days, uplift_events
    )

    # 시뮬레이션 수행
    timeline_rows: list[pd.DataFrame] = []
    sales_rows: list[pd.DataFrame] = []

    for (center_value, sku), group in timeline.groupby(["center", "resource_code"]):
        df_sales, df_inv = simulate_inventory_sales_for_group(
            group, center_value, sku, index, start_norm, mean_rate, uplift
        )
        if df_sales is not None and df_inv is not None:
            sales_rows.append(df_sales)
            timeline_rows.append(df_inv)

    # Filtered sales 재계산 (combine_forecast_results에서 사용)
    sku_mask = sales_history.get("resource_code")
    if sku_mask is None:
        filtered_sales = pd.DataFrame(
            columns=["date", "center", "resource_code", "sales_ea"]
        )
    else:
        sku_mask = sales_history["resource_code"].astype(str).isin(sku_set)
        if "center" in sales_history.columns:
            center_mask = sales_history["center"].astype(str).isin(center_set)
        else:
            center_mask = pd.Series(True, index=sales_history.index)
        filtered_sales = sales_history[sku_mask & center_mask].copy()
        if "center" not in filtered_sales.columns:
            inferred_center = center_set[0] if len(center_set) == 1 else ""
            filtered_sales["center"] = inferred_center

    # 결과 결합
    return combine_forecast_results(sales_rows, timeline_rows, filtered_sales)
