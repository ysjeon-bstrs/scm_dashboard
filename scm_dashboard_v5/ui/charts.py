# scm_dashboard_v5/ui/charts.py
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# v5 코어/소진 로직 사용
from scm_dashboard_v5.core import build_timeline as build_core_timeline
from scm_dashboard_v5.forecast import apply_consumption_with_events

# ──────────────────────────────────────────────────────────────────────────────
# 색상 팔레트: v4/v5 step chart와 동일한 순서로 사용(동일 SKU=동일 색)
PALETTE = [
    "#4E79A7","#F28E2B","#E15759","#76B7B2","#59A14F",
    "#EDC948","#B07AA1","#FF9DA7","#9C755F","#BAB0AC",
    "#1F77B4","#FF7F0E","#2CA02C","#D62728","#9467BD",
    "#8C564B","#E377C2","#7F7F7F","#BCBD22","#17BECF"
]

def _pick_amazon_centers(all_centers: list[str], centers: list[str] | None) -> list[str]:
    if centers:
        return centers
    return [c for c in all_centers if "amazon" in str(c).lower() or str(c).upper().startswith("AMZ")]

def _sku_colors(skus: list[str]) -> dict[str, str]:
    cols = {}
    for i, sku in enumerate(sorted(skus)):
        cols[sku] = PALETTE[i % len(PALETTE)]
    return cols

def render_amazon_sales_vs_inventory(
    snap_long: pd.DataFrame,
    moves: pd.DataFrame,
    centers: list[str] | None,
    skus: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    lookback_days: int,
    today: pd.Timestamp,
    show_ma7: bool = True,
    show_inbound: bool = False,          # 더 이상 사용하지 않지만 시그니처 호환 유지
    show_inventory_forecast: bool = True # 더 이상 토글 안 씀(항상 오늘 이후 점선)
) -> None:
    """
    좌측: SKU별 일별 판매 막대(누적 스택)
    우측: SKU별 재고 계단선(오늘까지 실선, 이후 점선)
    """
    if snap_long.empty or not skus:
        st.info("아마존 차트: 표시할 데이터가 없습니다.")
        return

    # 표준화
    snap = snap_long.rename(columns={"snapshot_date": "date"}).copy()
    snap["date"] = pd.to_datetime(snap["date"], errors="coerce").dt.normalize()

    start = pd.to_datetime(start).normalize()
    end   = pd.to_datetime(end).normalize()
    today = pd.to_datetime(today).normalize()
    idx   = pd.date_range(start, end, freq="D")

    # Amazon 계열 센터 자동 추출(입력 비어 있으면)
    all_centers = (
        snap["center"].dropna().astype(str).unique().tolist()
        if "center" in snap.columns else []
    )
    amz_centers = _pick_amazon_centers(all_centers, centers)
    if not amz_centers:
        st.caption("Amazon 계열 센터가 없어 차트를 그릴 수 없습니다.")
        return

    # ──────────────────────────────────────────────────────────────────
    # ① SKU 색상 고정
    colors = _sku_colors(skus)

    # ──────────────────────────────────────────────────────────────────
    # ② 일별 재고 스냅샷 → 판매량(감소분) 산출 (입고는 판매에 포함되지 않음)
    amz_snap = snap[
        snap["center"].astype(str).isin(amz_centers)
        & snap["resource_code"].astype(str).isin(skus)
        & snap["date"].between(start, end)
    ][["date","resource_code","stock_qty"]].copy()

    # 일자×SKU wide
    stock_wide = (
        amz_snap.groupby(["date","resource_code"])["stock_qty"]
        .sum().unstack("resource_code").reindex(idx)
        .ffill().fillna(0)
    )
    # 판매(실측) = '전일 - 금일'의 감소분만
    sales_actual = (-stock_wide.diff()).clip(lower=0).fillna(0)

    # 7일 이동평균(옵션)
    if show_ma7:
        sales_ma7 = sales_actual.rolling(7, min_periods=1).mean()
    else:
        sales_ma7 = None

    # 미래 판매 예측치(기본: 최근 lookback_days의 평균 소진속도)
    rates = {}
    hist_end = min(today, end)
    for sku in skus:
        hist = sales_actual.loc[sales_actual.index <= hist_end, sku].tail(int(lookback_days))
        rates[sku] = float(hist.mean()) if not hist.empty else 0.0

    sales_series = sales_actual.copy()
    if end > today:
        fut_idx = sales_series.index[sales_series.index > today]
        for sku in skus:
            sales_series.loc[fut_idx, sku] = rates.get(sku, 0.0)

    # ──────────────────────────────────────────────────────────────────
    # ③ 재고 타임라인(코어) + 소진 적용(오늘 이후)
    #    - 코어 빌드는 입출 이벤트(온보드/인바운드)를 반영 → 계단
    #    - 오늘 이후엔 추세 소진을 적용(점선)
    horizon_days = max(0, int((end - snap["date"].max()).days))
    tl = build_core_timeline(
        snap_long, moves,
        centers_sel=amz_centers, skus_sel=skus,
        start_dt=start, end_dt=end, horizon_days=horizon_days, today=today
    )
    # 실제 센터 라인만 채택(In-Transit/WIP 제외)
    tl = tl[tl["center"].isin(amz_centers)][["date","resource_code","stock_qty"]]

    tl = apply_consumption_with_events(
        timeline=tl,
        snap_long=snap_long,
        centers_sel=amz_centers, skus_sel=skus,
        start_dt=start, end_dt=end,
        lookback_days=int(lookback_days),
        events=None
    )

    inv_wide = (
        tl.groupby(["date","resource_code"])["stock_qty"]
        .sum().unstack("resource_code").reindex(idx)
        .ffill().fillna(0)
    )

    # ──────────────────────────────────────────────────────────────────
    # ④ 그리기 (좌: 판매 스택 막대 / 우: 재고 계단선)
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # (A) 판매 막대 - SKU별 누적 스택
    for sku in skus:
        fig.add_bar(
            x=idx, y=sales_series.get(sku, pd.Series(0, index=idx)),
            name=sku,
            marker_color=colors[sku],
            hovertemplate=f"{sku}<br>%{{x|%Y-%m-%d}}<br>판매: %{{y:,.0f}} EA<extra></extra>",
            offsetgroup=sku
        )

    # (A-보조) 7일 이동평균(옵션) - 점선
    if show_ma7 and sales_ma7 is not None:
        for sku in skus:
            fig.add_trace(
                go.Scatter(
                    x=idx, y=sales_ma7.get(sku, pd.Series(0, index=idx)),
                    mode="lines",
                    name=f"{sku} 7일 MA",
                    line=dict(color=colors[sku], dash="dot", width=1.5),
                    hovertemplate=f"{sku} 7일 MA<br>%{{x|%Y-%m-%d}}<br>%{{y:,.0f}} EA<extra></extra>"
                ),
                secondary_y=False
            )

    # (B) 재고선(계단) - 오늘까지 실선, 오늘 이후 점선
    for sku in skus:
        y = inv_wide.get(sku, pd.Series(0, index=idx))
        # 실측 구간(<= today)
        past_mask = (idx <= today)
        if past_mask.any():
            fig.add_trace(
                go.Scatter(
                    x=idx[past_mask], y=y[past_mask],
                    mode="lines",
                    name=f"{sku} 재고",
                    line=dict(color=colors[sku], width=2.0, shape="hv"),
                    hovertemplate=f"{sku} 재고<br>%{{x|%Y-%m-%d}}<br>%{{y:,.0f}} EA<extra></extra>",
                    showlegend=False
                ),
                secondary_y=True
            )
        # 예측 구간(> today)
        fut_mask = (idx > today)
        if fut_mask.any():
            fig.add_trace(
                go.Scatter(
                    x=idx[fut_mask], y=y[fut_mask],
                    mode="lines",
                    name=f"{sku} 재고(예측)",
                    line=dict(color=colors[sku], width=2.0, dash="dash", shape="hv"),
                    hovertemplate=f"{sku} 재고(예측)<br>%{{x|%Y-%m-%d}}<br>%{{y:,.0f}} EA<extra></extra>",
                    showlegend=False
                ),
                secondary_y=True
            )

    # 레이아웃: 제목은 Streamlit 쪽에서, 범례는 위쪽 가로, 마진 확보
    fig.update_layout(
        barmode="stack",
        hovermode="x unified",
        margin=dict(l=10, r=10, t=20, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        # 제목은 차트 내부에 두지 않아 겹침 방지(=None)
        title=None
    )
    # 좌/우 축 라벨
    fig.update_yaxes(title_text="판매량 (EA/Day)", secondary_y=False)
    fig.update_yaxes(title_text="재고 (EA)", secondary_y=True)

    # 오늘 기준선
    fig.add_vline(
        x=today, line_width=2, line_color="#D62728", line_dash="dot",
        annotation_text="오늘", annotation_position="top right",
        annotation_font_color="#D62728"
    )

    st.plotly_chart(
        fig, use_container_width=True,
        config={"displaylogo": False}
    )
