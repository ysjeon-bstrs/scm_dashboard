"""Plotting helpers for the Streamlit dashboard."""

from __future__ import annotations

from typing import Optional, Sequence

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from scm_dashboard_v4.config import PALETTE
from scm_dashboard_v4.consumption import estimate_daily_consumption


def _apply_line_styles(fig) -> None:
    """Apply v4-compatible styling to the traces in the step chart."""

    line_colors: dict[str, str] = {}
    color_idx = 0
    for trace in fig.data:
        name = trace.name or ""
        if " @ " not in name:
            continue
        if name not in line_colors:
            line_colors[name] = PALETTE[color_idx % len(PALETTE)]
            color_idx += 1

    for idx, trace in enumerate(fig.data):
        name = trace.name or ""
        if " @ " not in name:
            continue
        _, center_label = name.split(" @ ", 1)
        color = line_colors.get(name, PALETTE[0])
        if center_label == "이동중":
            fig.data[idx].update(line=dict(color=color, dash="dot", width=1.2), opacity=0.9)
        elif center_label == "생산중":
            fig.data[idx].update(line=dict(color=color, dash="dash", width=1.0), opacity=0.8)
        else:
            fig.data[idx].update(line=dict(color=color, dash="solid", width=1.5), opacity=1.0)


def render_step_chart(
    timeline: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    centers: Sequence[str],
    skus: Sequence[str],
    show_production: bool,
    show_in_transit: bool,
    today: pd.Timestamp | None = None,
    caption: str | None = "실데이터는 실선으로, 추세 예측치는 점선으로 표시됩니다.",
) -> None:
    """Render the step chart using the proven v4 styling pipeline."""

    if timeline is None or timeline.empty:
        st.info("선택한 조건에 해당하는 타임라인 데이터가 없습니다.")
        return

    today = (pd.to_datetime(today).normalize() if today is not None else pd.Timestamp.today().normalize())
    centers = [str(center) for center in centers if str(center).strip()]
    skus = [str(sku) for sku in skus if str(sku).strip()]

    vis_df = timeline.copy()
    vis_df["date"] = pd.to_datetime(vis_df["date"], errors="coerce").dt.normalize()
    vis_df = vis_df[
        (vis_df["date"] >= pd.to_datetime(start).normalize())
        & (vis_df["date"] <= pd.to_datetime(end).normalize())
    ]

    if vis_df.empty:
        st.info("선택한 조건에 해당하는 타임라인 데이터가 없습니다.")
        return

    vis_df["center"] = vis_df["center"].astype(str)
    vis_df["center"] = vis_df["center"].str.replace(
        r"^In-Transit.*$", "In-Transit", regex=True
    )

    centers_set = set(centers)
    if "태광KR" not in centers_set:
        vis_df = vis_df[vis_df["center"] != "WIP"]
    if not show_production:
        vis_df = vis_df[vis_df["center"] != "WIP"]
    if not show_in_transit:
        vis_df = vis_df[vis_df["center"] != "In-Transit"]

    vis_df["center"] = vis_df["center"].str.replace(
        r"^In-Transit$", "이동중", regex=True
    )
    vis_df.loc[vis_df["center"] == "WIP", "center"] = "생산중"

    vis_df = vis_df[pd.to_numeric(vis_df["stock_qty"], errors="coerce").fillna(0) > 0]
    if vis_df.empty:
        st.info("표시할 재고 데이터가 없습니다.")
        return

    vis_df["resource_code"] = vis_df["resource_code"].astype(str)
    vis_df["label"] = vis_df["resource_code"] + " @ " + vis_df["center"]

    fig = px.line(
        vis_df,
        x="date",
        y="stock_qty",
        color="label",
        line_shape="hv",
        title="선택한 SKU × 센터(및 이동중/생산중) 계단식 재고 흐름",
        render_mode="svg",
    )
    fig.update_layout(
        hovermode="x unified",
        xaxis_title="날짜",
        yaxis_title="재고량(EA)",
        legend_title_text="SKU @ Center / 이동중(점선) / 생산중(점선)",
        margin=dict(l=20, r=20, t=60, b=20),
    )
    fig.update_traces(
        hovertemplate="날짜: %{x|%Y-%m-%d}<br>재고: %{y:,.0f} EA<br>%{fullData.name}<extra></extra>"
    )
    fig.update_yaxes(tickformat=",.0f")

    if pd.to_datetime(start).normalize() <= today <= pd.to_datetime(end).normalize():
        fig.add_vline(x=today, line_width=1, line_dash="solid", line_color="rgba(255, 0, 0, 0.4)")
        fig.add_annotation(
            x=today,
            y=1.02,
            xref="x",
            yref="paper",
            text="오늘",
            showarrow=False,
            font=dict(size=12, color="#555"),
            align="center",
            yanchor="bottom",
        )

    _apply_line_styles(fig)

    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
    if caption:
        st.caption(caption)


def render_amazon_sales_vs_inventory(
    snapshot: pd.DataFrame,
    *,
    moves: Optional[pd.DataFrame],
    centers: Sequence[str],
    skus: Sequence[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    lookback_days: int = 28,
    today: Optional[pd.Timestamp] = None,
    show_ma7: bool = True,
    show_inbound: bool = False,
    show_inventory_forecast: bool = True,
    caption: str | None = None,
) -> None:
    """Render the Amazon sales vs. inventory chart with improved UX."""

    today = pd.to_datetime(today).normalize() if today is not None else pd.Timestamp.today().normalize()
    start_norm = pd.to_datetime(start).normalize()
    end_norm = pd.to_datetime(end).normalize()
    if end_norm < start_norm:
        st.warning("기간 설정이 올바르지 않아 Amazon 차트를 그릴 수 없습니다.")
        return

    if snapshot is None or snapshot.empty:
        st.caption("선택된 조건에 해당하는 Amazon 스냅샷 데이터가 없습니다.")
        return

    def _is_amazon_center(label: str) -> bool:
        value = str(label or "").strip()
        if not value:
            return False
        lowered = value.lower()
        return "amazon" in lowered or value.upper().startswith("AMZ")

    center_list = [str(center).strip() for center in centers if str(center).strip()]
    if not center_list:
        center_list = [
            str(center)
            for center in snapshot.get("center", pd.Series(dtype=str)).dropna().astype(str).unique()
            if _is_amazon_center(center)
        ]

    center_list = [center for center in center_list if _is_amazon_center(center)]
    if not center_list:
        st.caption("Amazon 계열 센터가 선택되지 않았습니다.")
        return

    sku_list = [str(sku).strip() for sku in skus if str(sku).strip()]
    if not sku_list:
        st.caption("표시할 SKU가 없습니다.")
        return

    work = snapshot.copy()
    date_col = "date" if "date" in work.columns else "snapshot_date"
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce").dt.normalize()
    work = work.dropna(subset=[date_col])
    work["center"] = work["center"].astype(str)
    work["resource_code"] = work["resource_code"].astype(str)
    work = work[
        work["center"].isin(center_list) & work["resource_code"].astype(str).isin(sku_list)
    ]

    if work.empty:
        st.caption("선택된 조건에 해당하는 Amazon 스냅샷 데이터가 없습니다.")
        return

    work = work.rename(columns={date_col: "date"}).copy()
    idx = pd.date_range(start_norm, end_norm, freq="D")
    ts_inv = (
        work.groupby("date")["stock_qty"].sum()
        .sort_index()
        .reindex(idx)
        .ffill()
        .fillna(0.0)
    )
    ts_inv = ts_inv.clip(lower=0.0)
    ts_inv.name = "inventory_qty"

    if ts_inv.empty:
        st.caption("표시할 Amazon 재고 데이터가 없습니다.")
        return

    latest_snap = work["date"].max().normalize()

    inbound = pd.Series(0.0, index=ts_inv.index, name="inbound_qty")
    if moves is not None and not moves.empty:
        required_cols = {"inbound_date", "qty_ea", "to_center", "resource_code"}
        if required_cols.issubset(moves.columns):
            inbound_work = moves.dropna(subset=["inbound_date"]).copy()
            if not inbound_work.empty:
                inbound_work["inbound_date"] = pd.to_datetime(
                    inbound_work["inbound_date"], errors="coerce"
                ).dt.normalize()
                inbound_work = inbound_work.dropna(subset=["inbound_date"])
                inbound_work["to_center"] = inbound_work["to_center"].astype(str)
                inbound_work["resource_code"] = inbound_work["resource_code"].astype(str)
                inbound_work = inbound_work[
                    inbound_work["to_center"].isin(center_list)
                    & inbound_work["resource_code"].isin(sku_list)
                ]
                if not inbound_work.empty:
                    inbound = (
                        inbound_work.groupby("inbound_date")["qty_ea"].sum().sort_index().reindex(idx)
                    ).fillna(0.0)
                    inbound.name = "inbound_qty"

    prev = ts_inv.shift(1).fillna(ts_inv.iloc[0])
    sales = ((prev + inbound) - ts_inv).clip(lower=0.0)
    sales.name = "sales_qty"

    sales_ma = sales.rolling(7, min_periods=1).mean() if show_ma7 else None

    rates = estimate_daily_consumption(
        snapshot.rename(columns={"date": "snapshot_date"}),
        centers_sel=center_list,
        skus_sel=sku_list,
        asof_dt=latest_snap,
        lookback_days=int(lookback_days),
    )
    daily_rate = sum(float(rates.get((center, sku), 0.0)) for center in center_list for sku in sku_list)

    future_start = max(today + pd.Timedelta(days=1), latest_snap + pd.Timedelta(days=1))
    if future_start > end_norm:
        future_index = pd.DatetimeIndex([], name="date")
    else:
        future_index = pd.date_range(future_start, end_norm, freq="D")

    sales_forecast = (
        pd.Series(daily_rate, index=future_index, name="sales_forecast") if len(future_index) else pd.Series(dtype=float)
    )

    inv_pred: Optional[pd.Series] = None
    if show_inventory_forecast and len(future_index):
        inv_values = []
        current = float(ts_inv.loc[ts_inv.index <= latest_snap].iloc[-1]) if (ts_inv.index <= latest_snap).any() else float(ts_inv.iloc[-1])
        for _ in future_index:
            current = max(0.0, current - daily_rate)
            inv_values.append(current)
        inv_pred = pd.Series(inv_values, index=future_index, name="inventory_forecast").round()

    fig = go.Figure()

    fig.add_bar(
        x=sales.index,
        y=sales.values,
        name="판매(실측)",
        marker_color="rgba(33, 150, 243, 0.85)",
        yaxis="y",
    )

    if show_ma7 and sales_ma is not None:
        fig.add_scatter(
            x=sales_ma.index,
            y=sales_ma.values,
            mode="lines",
            name="판매 7일 평균",
            line=dict(color="rgba(33,150,243,0.6)", dash="dot", width=2),
            yaxis="y",
        )

    if len(sales_forecast):
        fig.add_bar(
            x=sales_forecast.index,
            y=sales_forecast.values,
            name="판매(예측)",
            marker_color="rgba(33,150,243,0.25)",
            marker_pattern_shape="/",
            yaxis="y",
        )

    if show_inbound and inbound.sum() > 0:
        fig.add_bar(
            x=inbound.index,
            y=inbound.values,
            name="입고(실제)",
            marker_color="rgba(200,200,200,0.45)",
            marker_pattern_shape="\\",
            opacity=0.6,
            yaxis="y2",
        )

    fig.add_scatter(
        x=ts_inv.index,
        y=ts_inv.values,
        mode="lines",
        name="재고",
        line=dict(color="#ff7f0e", width=2.2),
        yaxis="y2",
    )

    if inv_pred is not None and len(inv_pred):
        fig.add_scatter(
            x=inv_pred.index,
            y=inv_pred.values,
            mode="lines",
            name="재고",
            showlegend=False,
            line=dict(color="#ff7f0e", width=2.2, dash="dot"),
            yaxis="y2",
        )

    if start_norm <= today <= end_norm:
        fig.add_vline(x=today, line_color="crimson", line_width=2, opacity=0.8)

    fig.update_layout(
        barmode="overlay",
        hovermode="x unified",
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        xaxis=dict(title=None),
        yaxis=dict(
            title="판매량 (EA/Day)",
            rangemode="tozero",
            gridcolor="rgba(0,0,0,0.06)",
            zeroline=False,
        ),
        yaxis2=dict(
            title="재고 (EA)",
            overlaying="y",
            side="right",
            showgrid=False,
            zeroline=False,
        ),
    )

    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
    if caption:
        st.caption(caption)
