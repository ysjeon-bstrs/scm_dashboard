"""Amazon 차트 렌더링 헬퍼 함수 모듈.

render_amazon_sales_vs_inventory 함수를 작은 함수들로 분해한 헬퍼들입니다.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from scm_dashboard_v9.forecast import AmazonForecastContext


def extract_forecast_parameters(
    ctx: "AmazonForecastContext",
    df: pd.DataFrame,
) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, int, float]:
    """컨텍스트에서 예측 파라미터를 추출하고 검증합니다.

    Args:
        ctx: Amazon forecast context
        df: 준비된 스냅샷 DataFrame

    Returns:
        tuple: (start, end, today, lookback_days, promo_multiplier)
    """
    start = pd.to_datetime(getattr(ctx, "start", df["date"].min())).normalize()
    end = pd.to_datetime(getattr(ctx, "end", df["date"].max())).normalize()
    today = pd.to_datetime(getattr(ctx, "today", pd.Timestamp.today())).normalize()

    lookback_days = int(getattr(ctx, "lookback_days", 28) or 28)
    lookback_days = max(1, lookback_days)

    promo_multiplier = float(getattr(ctx, "promotion_multiplier", 1.0) or 1.0)
    if not np.isfinite(promo_multiplier) or promo_multiplier <= 0:
        promo_multiplier = 1.0

    return start, end, today, lookback_days, promo_multiplier


def aggregate_actual_data(
    df: pd.DataFrame,
    today: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """실측 재고 및 판매 데이터를 집계합니다.

    Args:
        df: 'kind' 컬럼이 추가된 스냅샷 DataFrame
        today: 현재 날짜 (actual/future 구분 기준)

    Returns:
        tuple: (inv_actual_snapshot, sales_actual)
    """
    df["kind"] = np.where(df["date"] <= today, "actual", "future")

    inv_actual_snapshot = (
        df[df["kind"] == "actual"]
        .groupby(["date", "resource_code"], as_index=False)["stock_qty"]
        .sum()
    )

    sales_actual = (
        df[df["kind"] == "actual"]
        .groupby(["date", "resource_code"], as_index=False)["sales_qty"]
        .sum()
    )

    return inv_actual_snapshot, sales_actual


def calculate_sku_metrics(
    df: pd.DataFrame,
    today: pd.Timestamp,
    lookback_days: int,
) -> tuple[dict[str, float], dict[str, float]]:
    """SKU별 평균 수요 및 마지막 재고를 계산합니다.

    Args:
        df: 스냅샷 DataFrame
        today: 현재 날짜
        lookback_days: 평균 계산 시 조회할 과거 일수

    Returns:
        tuple: (avg_demand_by_sku, last_stock_by_sku)
    """
    avg_demand_by_sku: dict[str, float] = {}
    last_stock_by_sku: dict[str, float] = {}

    for sku, group in df.groupby("resource_code"):
        history = group[group["date"] <= today].sort_values("date")
        tail = history.tail(lookback_days)
        avg = float(tail["sales_qty"].mean() or 0.0)
        avg_demand_by_sku[sku] = max(0.0, avg)

        if not history.empty:
            last_stock_by_sku[sku] = float(history.iloc[-1]["stock_qty"])
        else:
            last_stock_by_sku[sku] = 0.0

    return avg_demand_by_sku, last_stock_by_sku


def calculate_moving_average(
    show_ma7: bool,
    sales_actual: pd.DataFrame,
) -> pd.DataFrame:
    """판매 데이터의 7일 이동평균을 계산합니다.

    Args:
        show_ma7: MA7 표시 여부
        sales_actual: 실측 판매 DataFrame (columns: date, resource_code, sales_qty)

    Returns:
        DataFrame: MA7 DataFrame (columns: date, resource_code, sales_ma7)
    """
    if show_ma7 and not sales_actual.empty:
        ma = (
            sales_actual.set_index("date")
            .groupby("resource_code")["sales_qty"]
            .apply(lambda s: s.rolling(7, min_periods=1).mean())
            .reset_index()
            .rename(columns={"sales_qty": "sales_ma7"})
        )
    else:
        ma = pd.DataFrame(columns=["date", "resource_code", "sales_ma7"])

    return ma
