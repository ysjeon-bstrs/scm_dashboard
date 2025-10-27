"""계단식(Step) 재고 흐름 차트 렌더러.

render_step_chart 함수를 제공합니다.
"""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd
import streamlit as st

from ..colors import STEP_PALETTE, step_sku_color_map, shade_for, tint
from ..data_utils import safe_dataframe, ensure_naive_index
from ..plotly_helpers import ensure_plotly_available

try:
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError:
    px = None
    go = None

_STEP_PALETTE = [
    "#4E79A7",
    "#F28E2B",
    "#E15759",
    "#76B7B2",
    "#59A14F",
    "#EDC948",
    "#B07AA1",
    "#FF9DA7",
    "#9C755F",
    "#BAB0AC",
    "#1F77B4",
    "#FF7F0E",
    "#2CA02C",
    "#D62728",
    "#9467BD",
    "#8C564B",
    "#E377C2",
    "#7F7F7F",
    "#BCBD22",
    "#17BECF",
    "#8DD3C7",
    "#FFFFB3",
    "#BEBADA",
    "#FB8072",
    "#80B1D3",
    "#FDB462",
    "#B3DE69",
    "#FCCDE5",
    "#D9D9D9",
    "#BC80BD",
    "#CCEBC5",
    "#FFED6F",
]


def render_step_chart(
    timeline: pd.DataFrame,
    *,
    centers: list[str] | None = None,
    skus: list[str] | None = None,
    start: pd.Timestamp,
    end: pd.Timestamp,
    today: pd.Timestamp | None = None,
    horizon_days: int = 0,
    show_in_transit: bool = True,
    show_wip: bool | None = None,
    title: str = "센터/SKU별 재고 대시보드",
    snapshot: pd.DataFrame | None = None,
    **kwargs,
) -> None:
    """
    v5_main에서 그대로 호출하는 공개 API.
    timeline: columns=[date, center, resource_code, stock_qty] (apply_consumption_with_events 반영 가능)
    snapshot: 스냅샷 데이터 (오늘 날짜 호버 표시 개선용, 선택적)
    """

    if not ensure_plotly_available():
        return

    if timeline is None or timeline.empty:
        st.info("타임라인 데이터가 없습니다.")
        return

    df = timeline.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    # "WIP" 센터를 태광KR 소속 생산중 상태로 통합한다.
    df["center"] = df["center"].astype(str)
    df["resource_code"] = df["resource_code"].astype(str)
    wip_mask = df["center"] == "WIP"
    df["is_wip"] = wip_mask
    df.loc[wip_mask, "center"] = "태광KR"

    show_production = kwargs.pop("show_production", None)
    if show_wip is None:
        show_wip = True if show_production is None else bool(show_production)
    else:
        show_wip = bool(show_wip)

    # 기간 슬라이스
    df = df[
        (df["date"] >= pd.to_datetime(start).normalize())
        & (df["date"] <= pd.to_datetime(end).normalize())
    ]

    # In‑Transit / WIP 노출 옵션
    if not show_in_transit:
        df = df[df["center"] != "In-Transit"]

    if centers:
        normalized_centers = [
            "태광KR" if str(center) == "WIP" else str(center) for center in centers
        ]
        df = df[df["center"].isin(normalized_centers)]
    if skus:
        df = df[df["resource_code"].isin([str(sku) for sku in skus])]

    wip_source = pd.DataFrame()
    if show_wip:
        wip_source = df[df["is_wip"]].copy()

    base_df = df[~df["is_wip"]].copy()

    if base_df.empty and (not show_wip or wip_source.empty):
        st.info("선택 조건에 해당하는 라인이 없습니다.")
        return

    # 라벨 생성: SKU @ Center
    plot_df = base_df.copy()
    if not plot_df.empty:
        plot_df["label"] = (
            plot_df["resource_code"] + " @ " + plot_df["center"].astype(str)
        )
    else:
        plot_df = pd.DataFrame(columns=["date", "stock_qty", "label"])

    # 오늘 날짜에 대해 스냅샷 재고 조회 (호버 표시 개선용)
    snapshot_today_map = {}
    if snapshot is not None and not snapshot.empty and today is not None:
        today_norm = pd.to_datetime(today).normalize()
        snapshot_df = snapshot.copy()
        snapshot_df["date"] = pd.to_datetime(
            snapshot_df.get("date", snapshot_df.get("snapshot_date", pd.NaT)),
            errors="coerce",
        ).dt.normalize()
        snapshot_today = snapshot_df[snapshot_df["date"] == today_norm]
        if (
            not snapshot_today.empty
            and "center" in snapshot_today.columns
            and "resource_code" in snapshot_today.columns
        ):
            for _, row in snapshot_today.iterrows():
                center = str(row.get("center", ""))
                sku = str(row.get("resource_code", ""))
                stock = float(row.get("stock_qty", 0))
                key = (today_norm, center, sku)
                snapshot_today_map[key] = stock

    # customdata 추가: hover 표시용 문자열 생성
    if not plot_df.empty:
        today_norm = pd.to_datetime(today).normalize() if today is not None else None

        # 기본: 총 재고만 표시
        plot_df["hover_stock"] = plot_df["stock_qty"].apply(lambda x: f"{x:,.0f} EA")

        # 오늘 날짜: 스냅샷 + 입고 분리 표시
        if snapshot_today_map and today_norm is not None:
            for idx, row in plot_df.iterrows():
                if row["date"] == today_norm:
                    key = (today_norm, row["center"], row["resource_code"])
                    base_stock = snapshot_today_map.get(key, 0)
                    inbound_qty = max(0, row["stock_qty"] - base_stock)
                    plot_df.at[idx, "hover_stock"] = (
                        f"{base_stock:,.0f} EA + {inbound_qty:,.0f} EA"
                    )

    # 기본 step line
    if plot_df.empty:
        fig = go.Figure()
        fig.update_layout(title=title)
    else:
        # customdata 준비
        customdata_cols = ["hover_stock"]
        plot_sorted = plot_df.sort_values(["label", "date"])

        fig = px.line(
            plot_sorted,
            x="date",
            y="stock_qty",
            color="label",
            line_shape="hv",
            render_mode="svg",
            title=title,
            custom_data=customdata_cols,
        )
        # hovertemplate: customdata[0]에 포맷팅된 문자열 사용
        fig.update_traces(
            mode="lines",
            hovertemplate=(
                "날짜: %{x|%Y-%m-%d}<br>"
                "재고: %{customdata[0]}<br>"
                "%{fullData.name}"
                "<extra></extra>"
            ),
        )

    # SKU별 고정 색, 상태(In‑Transit/WIP) 별 스타일
    color_labels: list[str] = []
    if not plot_df.empty:
        color_labels.extend(plot_df["label"].unique().tolist())
    if show_wip and not wip_source.empty:
        wip_skus_for_colors = sorted(
            {str(v) for v in wip_source["resource_code"].unique()}
        )
        color_labels.extend([f"{sku} @ 태광KR" for sku in wip_skus_for_colors])

    sku_colors = step_sku_color_map(color_labels)
    sku_center_seen: Dict[str, Dict[str, int]] = {}
    for tr in fig.data:
        name = tr.name or ""
        if " @ " not in name:
            continue
        sku, center = name.split(" @ ", 1)
        base_color = sku_colors.get(sku, _STEP_PALETTE[0])

        if center == "In-Transit":
            tr.update(line=dict(color=base_color, dash="dot", width=2.0), opacity=0.85)
            continue

        centers_for_sku = sku_center_seen.setdefault(sku, {})
        if center not in centers_for_sku:
            centers_for_sku[center] = len(centers_for_sku)
        center_index = centers_for_sku[center]
        shade = shade_for(center, center_index)
        color = tint(base_color, shade)
        tr.update(line=dict(color=color, dash="solid", width=2.4), opacity=0.95)

    wip_plot: Optional[pd.DataFrame] = None
    if show_wip and not wip_source.empty:
        wip_skus = list(dict.fromkeys([str(v) for v in (skus or [])]))
        if not wip_skus:
            wip_skus = sorted({str(v) for v in wip_source["resource_code"].unique()})
        wip_pivot = (
            wip_source.groupby(["date", "resource_code"])["stock_qty"]
            .sum()
            .unstack("resource_code")
        )
        wip_pivot = wip_pivot.reindex(columns=wip_skus, fill_value=0.0).sort_index()
        if not wip_pivot.empty:
            wip_plot = safe_dataframe(wip_pivot.round(0), columns=wip_skus)
            if not wip_plot.empty:
                wip_plot.index = ensure_naive_index(wip_plot.index)

    if wip_plot is not None and not wip_plot.empty:
        for sku in wip_plot.columns:
            series = wip_plot.get(sku)
            if series is None:
                continue
            if isinstance(series, pd.DataFrame):
                if series.empty:
                    continue
                series = series.iloc[:, 0]
            if not isinstance(series, pd.Series):
                series = pd.Series(series, index=wip_plot.index)
            numeric = pd.to_numeric(series, errors="coerce").dropna()
            if numeric.empty:
                continue
            color = sku_colors.get(sku, _STEP_PALETTE[0])
            fig.add_scatter(
                x=numeric.index,
                y=numeric,
                mode="lines",
                name=f"{sku} 태광KR 생산중",
                line=dict(color=color, dash="dot", width=2.0),
                hovertemplate="날짜: %{x|%Y-%m-%d}<br>재고: %{y:,} EA<br>%{fullData.name}<extra></extra>",
            )

    # 오늘 세로선
    if today is not None:
        t = pd.to_datetime(today).normalize()
        t_dt = t.to_pydatetime()
        fig.add_shape(
            type="line",
            x0=t_dt,
            x1=t_dt,
            xref="x",
            y0=0,
            y1=1,
            yref="paper",
            line=dict(color="crimson", dash="dot", width=1.5),
        )
        fig.add_annotation(
            x=t_dt,
            xref="x",
            y=1.0,
            yref="paper",
            yshift=8,
            text="오늘",
            showarrow=False,
            font=dict(color="crimson"),
        )

    fig.update_layout(
        hovermode="x unified",
        xaxis_title="날짜",
        yaxis_title="재고 (EA)",
        legend_title_text="SKU @ Center / 생산중",
        legend=dict(
            orientation="h",
            x=0,
            xanchor="left",
            y=-0.25,
            yanchor="top",
            bgcolor="rgba(255,255,255,0.6)",
        ),
        margin=dict(l=20, r=20, t=40, b=90),
        height=520,
    )

    # 라벨 겹침 완화: 상단 캡션으로 설명 이동 (Streamlit UI에서 처리)
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
