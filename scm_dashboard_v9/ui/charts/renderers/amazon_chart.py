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

    if not _ensure_plotly_available() or go is None:
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

    cols_lower = {str(c).strip().lower(): c for c in snap_long.columns}
    date_col = cols_lower.get("date") or cols_lower.get("snapshot_date")
    center_col = cols_lower.get("center")
    sku_col = cols_lower.get("resource_code") or cols_lower.get("sku")
    stock_col = cols_lower.get("stock_qty") or cols_lower.get("qty")
    sales_col = cols_lower.get("sales_qty") or cols_lower.get("sale_qty")

    if not all([date_col, center_col, sku_col, stock_col]):
        st.warning("정제 스냅샷 형식이 예상과 다릅니다.")
        return

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

    if df.empty:
        st.info("AMZUS 데이터가 없습니다.")
        return

    start = pd.to_datetime(getattr(ctx, "start", df["date"].min())).normalize()
    end = pd.to_datetime(getattr(ctx, "end", df["date"].max())).normalize()
    today = pd.to_datetime(getattr(ctx, "today", pd.Timestamp.today())).normalize()
    lookback_days = int(getattr(ctx, "lookback_days", 28) or 28)
    lookback_days = max(1, lookback_days)
    promo_multiplier = float(getattr(ctx, "promotion_multiplier", 1.0) or 1.0)
    if not np.isfinite(promo_multiplier) or promo_multiplier <= 0:
        promo_multiplier = 1.0

    df = df[
        (df["date"] >= start - pd.Timedelta(days=lookback_days + 2))
        & (df["date"] <= end)
    ]

    # === 날짜 경계 ===
    display_start = start
    display_end = end

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

    fcst_start = max(today + pd.Timedelta(days=1), start)
    fallback_sales_rows: list[pd.DataFrame] = []
    fallback_inv_rows: list[pd.DataFrame] = []

    missing_sales_skus: set[str] = set(skus)
    missing_inv_skus: set[str] = set(skus)

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

    fallback_skus = sorted((missing_sales_skus | missing_inv_skus))

    if fcst_start <= end and fallback_skus:
        for sku in fallback_skus:
            base_stock = float(last_stock_by_sku.get(sku, 0.0))
            inbound_map = {
                pd.to_datetime(day): float(qty)
                for day, qty in inbound[inbound["resource_code"] == sku][["event_date", "qty_ea"]]
                .itertuples(index=False, name=None)
            }

            daily_demand = avg_demand_by_sku.get(sku, 0.0) * promo_multiplier
            fcst_sales, inv_series = _clamped_forecast_series(
                start_date=fcst_start,
                end_date=end,
                base_stock=base_stock,
                inbound_by_day=inbound_map,
                daily_demand=daily_demand,
            )

            if sku in missing_sales_skus and not fcst_sales.empty:
                fallback_sales_rows.append(
                    pd.DataFrame(
                        {
                            "date": fcst_sales.index,
                            "resource_code": sku,
                            "sales_qty": fcst_sales.values,
                        }
                    )
                )
            if sku in missing_inv_skus and not inv_series.empty:
                fallback_inv_rows.append(
                    pd.DataFrame(
                        {
                            "date": inv_series.index,
                            "resource_code": sku,
                            "stock_qty": inv_series.values,
                        }
                    )
                )

    sales_forecast_df = (
        pd.concat(fallback_sales_rows, ignore_index=True)
        if fallback_sales_rows
        else pd.DataFrame(columns=["date", "resource_code", "sales_qty"])
    )

    inv_forecast_df = (
        pd.concat(fallback_inv_rows, ignore_index=True)
        if fallback_inv_rows
        else pd.DataFrame(columns=["date", "resource_code", "stock_qty"])
    )

    default_center = target_centers[0] if target_centers else None
    inv_actual_df = _normalize_inventory_frame(inv_actual_snapshot, default_center=default_center)
    inv_forecast_df = _normalize_inventory_frame(inv_forecast_df, default_center=default_center)

    override_inventory = False
    if inv_actual is not None:
        override_inventory = True
        inv_actual_df = _normalize_inventory_frame(inv_actual, default_center=default_center)
    if inv_forecast is not None:
        override_inventory = True
        inv_forecast_df = _normalize_inventory_frame(inv_forecast, default_center=default_center)

    if override_inventory and inv_forecast is None:
        # When only actuals are injected we should not keep stale fallback forecasts.
        inv_forecast_df = pd.DataFrame(columns=["date", "center", "resource_code", "stock_qty"])

    if use_inventory_for_sales and not inv_actual_df.empty and not inv_forecast_df.empty:
        derived_sales = _sales_forecast_from_inventory_projection(
            inv_actual_df,
            inv_forecast_df,
            centers=target_centers,
            skus=skus,
            start=start,
            end=end,
            today=today,
        )
        if not derived_sales.empty:
            sales_forecast_df = (
                derived_sales.rename(columns={"sales_ea": "sales_qty"})
                if "sales_ea" in derived_sales.columns
                else derived_sales.copy()
            )

    show_ma7 = bool(getattr(ctx, "show_ma7", True))
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

    colors = sku_colors or _sku_color_map(skus)
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


