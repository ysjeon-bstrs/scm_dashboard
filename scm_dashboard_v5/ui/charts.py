"""Plotting helpers for the Streamlit dashboard."""

from __future__ import annotations

from typing import Sequence

import pandas as pd
import plotly.express as px
import streamlit as st

from scm_dashboard_v4.config import PALETTE


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
    vis_df = vis_df[(vis_df["date"] >= pd.to_datetime(start).normalize()) & (vis_df["date"] <= pd.to_datetime(end).normalize())]

    if vis_df.empty:
        st.info("선택한 조건에 해당하는 타임라인 데이터가 없습니다.")
        return

    vis_df["center"] = vis_df["center"].astype(str)
    vis_df["center"] = vis_df["center"].str.replace(r"^In-Transit.*$", "이동중", regex=True)
    vis_df.loc[vis_df["center"] == "WIP", "center"] = "생산중"

    centers_set = set(centers)
    if "태광KR" not in centers_set:
        vis_df = vis_df[vis_df["center"] != "생산중"]
    if not show_production:
        vis_df = vis_df[vis_df["center"] != "생산중"]
    if not show_in_transit:
        vis_df = vis_df[~vis_df["center"].str.startswith("이동중")]

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
