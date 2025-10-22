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


def normalize_snapshot_data(
    snap_long: pd.DataFrame,
    target_centers: list[str],
    skus: list[str],
) -> pd.DataFrame:
    """스냅샷 데이터를 정규화하고 필터링합니다.

    Args:
        snap_long: 원본 스냅샷 DataFrame
        target_centers: 필터링할 센터 목록
        skus: 필터링할 SKU 목록

    Returns:
        DataFrame: 정규화된 스냅샷 (columns: date, center, resource_code, stock_qty, sales_qty)
    """
    cols_lower = {str(c).strip().lower(): c for c in snap_long.columns}
    date_col = cols_lower.get("date") or cols_lower.get("snapshot_date")
    center_col = cols_lower.get("center")
    sku_col = cols_lower.get("resource_code") or cols_lower.get("sku")
    stock_col = cols_lower.get("stock_qty") or cols_lower.get("qty")
    sales_col = cols_lower.get("sales_qty") or cols_lower.get("sale_qty")

    if not all([date_col, center_col, sku_col, stock_col]):
        return pd.DataFrame()

    rename_map = {
        date_col: "date",
        center_col: "center",
        sku_col: "resource_code",
        stock_col: "stock_qty",
    }
    if sales_col:
        rename_map[sales_col] = "sales_qty"

    df = snap_long.rename(columns=rename_map).copy()
    if "sales_qty" not in df.columns:
        df["sales_qty"] = 0

    df["date"] = pd.to_datetime(df.get("date"), errors="coerce").dt.normalize()
    df = df.dropna(subset=["date"])
    df["center"] = df.get("center", "").astype(str)
    df["resource_code"] = df.get("resource_code", "").astype(str)
    df["stock_qty"] = pd.to_numeric(df.get("stock_qty"), errors="coerce").fillna(0)
    df["sales_qty"] = pd.to_numeric(df.get("sales_qty"), errors="coerce").fillna(0)

    df = df[
        df["center"].isin(target_centers)
        & df["resource_code"].isin(skus)
    ].copy()

    return df


def process_moves_data(
    ctx: "AmazonForecastContext",
    target_centers: list[str],
    skus: list[str],
    today: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Moves 데이터를 처리하고 inbound를 계산합니다.

    Args:
        ctx: Amazon forecast context
        target_centers: 필터링할 센터 목록
        skus: 필터링할 SKU 목록
        today: 현재 날짜
        end: 종료 날짜

    Returns:
        tuple: (moves_df, inbound) - 정규화된 moves와 집계된 inbound
    """
    moves_df = getattr(ctx, "moves", pd.DataFrame()).copy()

    if not moves_df.empty:
        mv_cols = {str(c).lower(): c for c in moves_df.columns}
        rename_moves = {mv_cols.get("event_date", "event_date"): "event_date"}
        for name in ["to_center", "resource_code", "qty_ea"]:
            if name in mv_cols:
                rename_moves[mv_cols[name]] = name
        moves_df = moves_df.rename(columns=rename_moves)
        moves_df["event_date"] = pd.to_datetime(
            moves_df.get("event_date"), errors="coerce"
        ).dt.normalize()
        moves_df = moves_df.dropna(subset=["event_date"])
        moves_df["to_center"] = moves_df.get("to_center", "").astype(str)
        moves_df["resource_code"] = moves_df.get("resource_code", "").astype(str)
        moves_df["qty_ea"] = pd.to_numeric(moves_df.get("qty_ea"), errors="coerce").fillna(0)
        moves_df = moves_df[
            moves_df["to_center"].isin(target_centers)
            & moves_df["resource_code"].isin(skus)
            & (moves_df["event_date"] >= today + pd.Timedelta(days=1))
            & (moves_df["event_date"] <= end)
        ]
    else:
        moves_df = pd.DataFrame(columns=["event_date", "resource_code", "qty_ea"])

    inbound = (
        moves_df.groupby(["resource_code", "event_date"], as_index=False)["qty_ea"].sum()
        if not moves_df.empty
        else pd.DataFrame(columns=["resource_code", "event_date", "qty_ea"])
    )

    return moves_df, inbound


def process_inventory_forecast(
    ctx: "AmazonForecastContext",
    target_centers: list[str],
    skus: list[str],
    fcst_start: pd.Timestamp,
    end: pd.Timestamp,
    missing_inv_skus: set[str],
) -> tuple[list[pd.DataFrame], set[str]]:
    """재고 예측 데이터를 처리하고 fallback 리스트를 생성합니다.

    Args:
        ctx: Amazon forecast context
        target_centers: 필터링할 센터 목록
        skus: 필터링할 SKU 목록
        fcst_start: 예측 시작일
        end: 종료 날짜
        missing_inv_skus: 아직 처리되지 않은 SKU 집합

    Returns:
        tuple: (fallback_inv_rows, updated_missing_inv_skus)
    """
    fallback_inv_rows: list[pd.DataFrame] = []

    inv_forecast_ctx = getattr(ctx, "inv_forecast", pd.DataFrame()).copy()
    if not inv_forecast_ctx.empty:
        inv_forecast_ctx["date"] = pd.to_datetime(
            inv_forecast_ctx.get("date"), errors="coerce"
        ).dt.normalize()
        inv_forecast_ctx["center"] = inv_forecast_ctx.get("center", "").astype(str)
        inv_forecast_ctx["resource_code"] = inv_forecast_ctx.get("resource_code", "").astype(str)
        inv_forecast_ctx["stock_qty"] = pd.to_numeric(
            inv_forecast_ctx.get("stock_qty"), errors="coerce"
        ).fillna(0.0)
        inv_forecast_ctx = inv_forecast_ctx[
            inv_forecast_ctx["center"].isin(target_centers)
            & inv_forecast_ctx["resource_code"].isin(skus)
            & (inv_forecast_ctx["date"] >= fcst_start)
            & (inv_forecast_ctx["date"] <= end)
        ]
        if not inv_forecast_ctx.empty:
            grouped = (
                inv_forecast_ctx.groupby(["date", "resource_code"], as_index=False)[
                    "stock_qty"
                ].sum()
            )
            fallback_inv_rows.append(grouped)
            missing_inv_skus = missing_inv_skus - set(grouped["resource_code"].unique())

    return fallback_inv_rows, missing_inv_skus


def process_sales_forecast(
    ctx: "AmazonForecastContext",
    target_centers: list[str],
    skus: list[str],
    fcst_start: pd.Timestamp,
    end: pd.Timestamp,
    missing_sales_skus: set[str],
) -> tuple[list[pd.DataFrame], set[str]]:
    """판매 예측 데이터를 처리하고 fallback 리스트를 생성합니다.

    Args:
        ctx: Amazon forecast context
        target_centers: 필터링할 센터 목록
        skus: 필터링할 SKU 목록
        fcst_start: 예측 시작일
        end: 종료 날짜
        missing_sales_skus: 아직 처리되지 않은 SKU 집합

    Returns:
        tuple: (fallback_sales_rows, updated_missing_sales_skus)
    """
    fallback_sales_rows: list[pd.DataFrame] = []

    sales_forecast_ctx = getattr(ctx, "sales_forecast", pd.DataFrame()).copy()
    if not sales_forecast_ctx.empty:
        sales_forecast_ctx["date"] = pd.to_datetime(
            sales_forecast_ctx.get("date"), errors="coerce"
        ).dt.normalize()
        sales_forecast_ctx["center"] = sales_forecast_ctx.get("center", "").astype(str)
        sales_forecast_ctx["resource_code"] = (
            sales_forecast_ctx.get("resource_code", "").astype(str)
        )
        value_col: str | None = None
        if "sales_ea" in sales_forecast_ctx.columns:
            value_col = "sales_ea"
        elif "sales_qty" in sales_forecast_ctx.columns:
            value_col = "sales_qty"

        if value_col is not None:
            sales_forecast_ctx[value_col] = pd.to_numeric(
                sales_forecast_ctx.get(value_col), errors="coerce"
            ).fillna(0.0)
            sales_forecast_ctx = sales_forecast_ctx[
                sales_forecast_ctx["center"].isin(target_centers)
                & sales_forecast_ctx["resource_code"].isin(skus)
                & (sales_forecast_ctx["date"] >= fcst_start)
                & (sales_forecast_ctx["date"] <= end)
            ]
            if not sales_forecast_ctx.empty:
                grouped_sales = (
                    sales_forecast_ctx.groupby(["date", "resource_code"], as_index=False)[
                        value_col
                    ].sum()
                ).rename(columns={value_col: "sales_qty"})
                fallback_sales_rows.append(grouped_sales)
                missing_sales_skus = missing_sales_skus - set(
                    grouped_sales["resource_code"].unique()
                )

    return fallback_sales_rows, missing_sales_skus
