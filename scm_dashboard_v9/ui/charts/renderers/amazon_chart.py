"""Amazon 판매 vs 재고 차트 렌더러.

render_amazon_sales_vs_inventory 함수를 제공합니다.
"""

from __future__ import annotations

from typing import Dict, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd
import streamlit as st

from center_alias import normalize_center_value
from ..colors import sku_color_map
from ..data_utils import normalize_inventory_frame
from ..plotly_helpers import ensure_plotly_available, go
from ..inventory import clamped_forecast_series
from ..sales import sales_forecast_from_inventory_projection
from .amazon_chart_helpers import (
    extract_forecast_parameters,
    aggregate_actual_data,
    calculate_sku_metrics,
    calculate_moving_average,
    normalize_snapshot_data,
    process_moves_data,
    process_inventory_forecast,
    process_sales_forecast,
    generate_fallback_forecasts,
    finalize_forecast_dataframes,
)

if TYPE_CHECKING:
    from scm_dashboard_v9.forecast import AmazonForecastContext

def render_amazon_sales_vs_inventory(
    ctx: "AmazonForecastContext",
    *,
    inv_actual: Optional[pd.DataFrame] = None,
    inv_forecast: Optional[pd.DataFrame] = None,
    sku_colors: Optional[Dict[str, str]] = None,
    use_inventory_for_sales: bool = True,
) -> None:
    """Draw the Amazon US panel, optionally using a precomputed inventory timeline."""

    if not ensure_plotly_available() or go is None:
        return

    if ctx is None:
        st.info("아마존 데이터가 없습니다.")
        return

    skus = [str(sku) for sku in getattr(ctx, "skus", []) if str(sku).strip()]
    if not skus:
        st.info("SKU를 선택하세요.")
        return

    target_centers = [normalize_center_value(c) for c in getattr(ctx, "centers", [])]
    target_centers = [c for c in target_centers if c]
    if not target_centers:
        target_centers = ["AMZUS"]

    snap_long = getattr(ctx, "snapshot_long", pd.DataFrame()).copy()
    if snap_long.empty:
        st.info("AMZUS 데이터가 없습니다.")
        return

    df = normalize_snapshot_data(snap_long, target_centers, skus)
    if df.empty:
        # 컬럼 검증 실패 또는 필터링 후 데이터 없음
        cols_lower = {str(c).strip().lower(): c for c in snap_long.columns}
        date_col = cols_lower.get("date") or cols_lower.get("snapshot_date")
        center_col = cols_lower.get("center")
        sku_col = cols_lower.get("resource_code") or cols_lower.get("sku")
        stock_col = cols_lower.get("stock_qty") or cols_lower.get("qty")
        if not all([date_col, center_col, sku_col, stock_col]):
            st.warning("정제 스냅샷 형식이 예상과 다릅니다.")
        else:
            st.info("AMZUS 데이터가 없습니다.")
        return

    start, end, today, lookback_days, promo_multiplier = extract_forecast_parameters(ctx, df)

    df = df[
        (df["date"] >= start - pd.Timedelta(days=lookback_days + 2))
        & (df["date"] <= end)
    ]

    # === 날짜 경계 ===
    display_start = start
    display_end = end

    inv_actual_snapshot, sales_actual = aggregate_actual_data(df, today)

    avg_demand_by_sku, last_stock_by_sku = calculate_sku_metrics(df, today, lookback_days)

    moves_df, inbound = process_moves_data(ctx, target_centers, skus, today, end)

    fcst_start = max(today + pd.Timedelta(days=1), start)
    fallback_sales_rows: list[pd.DataFrame] = []
    fallback_inv_rows: list[pd.DataFrame] = []

    missing_sales_skus: set[str] = set(skus)
    missing_inv_skus: set[str] = set(skus)

    # Process inventory forecast from context
    inv_rows, missing_inv_skus = process_inventory_forecast(
        ctx, target_centers, skus, fcst_start, end, missing_inv_skus
    )
    fallback_inv_rows.extend(inv_rows)

    # Process sales forecast from context
    sales_rows, missing_sales_skus = process_sales_forecast(
        ctx, target_centers, skus, fcst_start, end, missing_sales_skus
    )
    fallback_sales_rows.extend(sales_rows)

    # Generate fallback forecasts for missing SKUs
    fallback_skus = sorted((missing_sales_skus | missing_inv_skus))
    if fcst_start <= end and fallback_skus:
        fb_sales, fb_inv = generate_fallback_forecasts(
            fallback_skus, fcst_start, end, last_stock_by_sku, inbound,
            avg_demand_by_sku, promo_multiplier, missing_sales_skus, missing_inv_skus
        )
        fallback_sales_rows.extend(fb_sales)
        fallback_inv_rows.extend(fb_inv)

    # Finalize forecast DataFrames
    default_center = target_centers[0] if target_centers else None
    sales_forecast_df, inv_actual_df, inv_forecast_df = finalize_forecast_dataframes(
        fallback_sales_rows, fallback_inv_rows, inv_actual_snapshot,
        inv_actual, inv_forecast, default_center, use_inventory_for_sales,
        target_centers, skus, start, end, today
    )

    show_ma7 = bool(getattr(ctx, "show_ma7", True))
    ma = calculate_moving_average(show_ma7, sales_actual)

    # --- 실측→예측 연결용 앵커 추가 ---
    if (
        display_start <= today <= display_end
        and not inv_actual_df.empty
        and not inv_forecast_df.empty
    ):
        last_actual = (
            inv_actual_df[inv_actual_df["date"] <= today]
            .sort_values(["resource_code", "date"])
            .groupby("resource_code", as_index=False)
            .last()[["resource_code", "stock_qty"]]
        )
        if not last_actual.empty:
            anchor = last_actual.assign(date=pd.to_datetime(today).normalize())
            inv_forecast_df = pd.concat(
                [anchor[["date", "resource_code", "stock_qty"]], inv_forecast_df],
                ignore_index=True,
            )
            inv_forecast_df = (
                inv_forecast_df.sort_values(["resource_code", "date"])
                .drop_duplicates(subset=["resource_code", "date"], keep="last")
            )

    # 표시용으로만 [start, end]로 슬라이스
    def _trim_range(df_in: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        if df_in is None or df_in.empty:
            return df_in
        out = df_in[
            (df_in["date"] >= display_start) & (df_in["date"] <= display_end)
        ].copy()
        return out

    inv_actual_df = _trim_range(inv_actual_df)
    inv_forecast_df = _trim_range(inv_forecast_df)
    sales_actual = _trim_range(sales_actual)
    sales_forecast_df = _trim_range(sales_forecast_df)
    ma = _trim_range(ma) if ma is not None and not ma.empty else ma

    colors = sku_colors or sku_color_map(skus)
    fig = go.Figure()

    if not sales_actual.empty:
        for sku, group in sales_actual.groupby("resource_code"):
            color = colors.get(sku, "#6BA3FF")
            fig.add_bar(
                x=group["date"],
                y=group["sales_qty"],
                name=f"{sku} 판매(실측)",
                marker_color=color,
                opacity=0.95,
                hovertemplate="날짜: %{x|%Y-%m-%d}<br>판매: %{y:,.0f} EA<br>%{fullData.name}<extra></extra>",
            )

    if not sales_forecast_df.empty:
        for sku, group in sales_forecast_df.groupby("resource_code"):
            color = colors.get(sku, "#6BA3FF")
            fig.add_bar(
                x=group["date"],
                y=group["sales_qty"],
                name=f"{sku} 판매(예측)",
                marker_color=color,
                opacity=0.45,
                hovertemplate="날짜: %{x|%Y-%m-%d}<br>판매: %{y:,.0f} EA<br>%{fullData.name}<extra></extra>",
            )

    if not inv_actual_df.empty:
        for sku, group in inv_actual_df.groupby("resource_code"):
            color = colors.get(sku, "#6BA3FF")
            fig.add_trace(
                go.Scatter(
                    x=group["date"],
                    y=group["stock_qty"],
                    mode="lines",
                    name=f"{sku} 재고(실측)",
                    line=dict(color=color, width=2),
                    yaxis="y2",
                    hovertemplate="날짜: %{x|%Y-%m-%d}<br>재고: %{y:,.0f} EA<br>%{fullData.name}<extra></extra>",
                )
            )

    if not inv_forecast_df.empty:
        for sku, group in inv_forecast_df.groupby("resource_code"):
            color = colors.get(sku, "#6BA3FF")
            fig.add_trace(
                go.Scatter(
                    x=group["date"],
                    y=group["stock_qty"],
                    mode="lines",
                    name=f"{sku} 재고(예측)",
                    line=dict(color=color, width=2, dash="dot"),
                    yaxis="y2",
                    hovertemplate="날짜: %{x|%Y-%m-%d}<br>재고: %{y:,.0f} EA<br>%{fullData.name}<extra></extra>",
                )
            )

    if show_ma7 and ma is not None and not ma.empty:
        for sku, group in ma.groupby("resource_code"):
            color = colors.get(sku, "#6BA3FF")
            fig.add_trace(
                go.Scatter(
                    x=group["date"],
                    y=group["sales_ma7"],
                    mode="lines",
                    name=f"{sku} 판매 7일 평균",
                    line=dict(color=color, dash="dash"),
                )
            )

    fig.add_vline(x=today, line_color="crimson", line_dash="dash", line_width=2)

    fig.update_layout(
         # 내부 제목 제거: 바깥에서 v5_main이 섹션 제목을 이미 표시함
        title="AMZUS",
        barmode="stack",
        legend=dict(
            orientation="h",
            x=0,
            xanchor="left",
            y=-0.25,
            yanchor="top",
            traceorder="normal",
            bgcolor="rgba(255,255,255,0.6)",
        ),
        # 상단 여백 축소해 겹침 방지
        margin=dict(l=30, r=20, t=10, b=90),
        hovermode="x unified",
        xaxis=dict(title="Date"),
        yaxis=dict(title="판매량 (EA/Day)", tickformat=",.0f"),
        yaxis2=dict(
            title="재고 (EA)",
            overlaying="y",
            side="right",
            showgrid=False,
            zeroline=False,
            showline=False,
            tickfont=dict(color="#666"),
            tickformat=",.0f",
        ),
    )

    fig.update_xaxes(range=[display_start, display_end])

    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})


