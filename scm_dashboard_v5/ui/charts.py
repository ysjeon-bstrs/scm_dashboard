# scm_dashboard_v5/ui/charts.py
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from typing import Iterable, Optional, Sequence


PALETTE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]

def _ensure_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """snap_long의 날짜 컬럼을 'date'로 통일."""
    df = df.copy()
    cols = {c.lower(): c for c in df.columns}
    date_col = cols.get("date") or cols.get("snapshot_date")
    if not date_col:
        raise KeyError("스냅샷 데이터프레임에 'date' 또는 'snapshot_date' 컬럼이 필요합니다.")
    if date_col != "date":
        df = df.rename(columns={date_col: "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    return df

def _detect_amazon_centers(names: Sequence[str]) -> list[str]:
    out = []
    for n in names:
        s = str(n or "").strip()
        if not s:
            continue
        if "amazon" in s.lower() or s.upper().startswith("AMZ"):
            out.append(s)
    # 중복 제거
    return sorted(set(out))

def _sum_inventory_series(snap: pd.DataFrame,
                          centers: Iterable[str],
                          skus: Iterable[str],
                          start: pd.Timestamp,
                          end: pd.Timestamp) -> pd.Series:
    """선택 Amazon 계열 센터 × SKU 합계 재고(일자별) 시리즈."""
    mask = (
        snap["center"].astype(str).isin(centers) &
        snap["resource_code"].astype(str).isin(list(skus)) &
        (snap["date"] >= start) & (snap["date"] <= end)
    )
    g = (snap.loc[mask]
            .groupby("date", as_index=True)["stock_qty"]
            .sum()
            .sort_index())
    idx = pd.date_range(start, end, freq="D")
    return g.reindex(idx).ffill().fillna(0.0)

def _daily_sales_from_snapshot(snap: pd.DataFrame,
                               centers: Iterable[str],
                               skus: Iterable[str],
                               start: pd.Timestamp,
                               end: pd.Timestamp) -> pd.Series:
    """
    스냅샷을 이용한 일별 판매량(추정).
    - '증가(입고)'는 0으로 무시
    - '감소'만 판매로 집계
    - 센터별로 계산한 뒤 날짜별 합산 (내부 이동으로 인한 왜곡 최소화)
    """
    snap_f = snap[
        snap["center"].astype(str).isin(centers) &
        snap["resource_code"].astype(str).isin(list(skus))
    ].copy()
    if snap_f.empty:
        return pd.Series(dtype=float)

    sales_parts = []
    for (_sku, _ct), g in snap_f.groupby(["resource_code", "center"]):
        g = g.sort_values("date")
        s = g.set_index("date")["stock_qty"].astype(float).asfreq("D").ffill()
        # 일별 변화량 (오늘 - 어제)
        diff = s.diff()
        # 감소분만 판매로 카운트
        sale = (-diff).clip(lower=0)
        sales_parts.append(sale)

    if not sales_parts:
        return pd.Series(dtype=float)

    sales = pd.concat(sales_parts, axis=1).sum(axis=1)
    sales = sales[(sales.index >= start) & (sales.index <= end)]
    sales = sales.fillna(0.0)
    return sales


def _normalise_timeline_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a normalised copy of the timeline frame for plotting."""

    required = {"date", "center", "resource_code", "stock_qty"}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(
            "타임라인 데이터에 필요한 컬럼이 없습니다: " + ", ".join(sorted(missing))
        )

    df = frame.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["center"] = df["center"].astype(str)
    df["resource_code"] = df["resource_code"].astype(str)
    df["stock_qty"] = pd.to_numeric(df["stock_qty"], errors="coerce").fillna(0.0)
    df = df[df["date"].notna()]
    return df


def _format_center(center: str) -> str:
    mapping = {"WIP": "생산중", "In-Transit": "이동중"}
    return mapping.get(center, center)


def render_step_chart(
    timeline: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    centers: Sequence[str],
    skus: Sequence[str],
    show_production: bool = True,
    show_in_transit: bool = True,
    today: Optional[pd.Timestamp] = None,
    title: str = "선택한 SKU × 센터(및 이동중/생산중) 계단식 재고 흐름",
    caption: str = "WIP/이동중 라인은 점선으로 표시됩니다.",
) -> None:
    """Render the consolidated inventory step chart using Plotly."""

    if timeline is None or timeline.empty:
        st.info("선택한 조건에 해당하는 타임라인 데이터가 없습니다.")
        return

    try:
        df = _normalise_timeline_frame(timeline)
    except KeyError as exc:  # pragma: no cover - guardrail for misconfigured data
        st.error(str(exc))
        return

    start_dt = pd.to_datetime(start).normalize()
    end_dt = pd.to_datetime(end).normalize()
    df = df[(df["date"] >= start_dt) & (df["date"] <= end_dt)]

    if df.empty:
        st.info("선택한 기간 내에 표시할 타임라인 데이터가 없습니다.")
        return

    centers_set = {str(c) for c in centers}
    if centers_set:
        df = df[df["center"].isin(centers_set | {"WIP", "In-Transit"})]

    if skus:
        df = df[df["resource_code"].isin({str(s) for s in skus})]

    if not show_production:
        df = df[df["center"] != "WIP"]
    if not show_in_transit:
        df = df[df["center"] != "In-Transit"]

    df = df[df["stock_qty"] > 0]

    if df.empty:
        st.info("선택한 조건에서 표시할 센터/생산중 데이터가 없습니다.")
        return

    df = df.sort_values(["resource_code", "center", "date"])
    df["center_label"] = df["center"].map(_format_center)
    df["label"] = df["resource_code"] + " @ " + df["center_label"]

    fig = go.Figure()
    colour_map: dict[str, str] = {}
    palette_iter = iter(PALETTE * ((len(df["label"].unique()) // len(PALETTE)) + 1))

    for label, group in df.groupby("label", sort=False):
        center_label = group["center_label"].iloc[0]
        try:
            colour = colour_map[label]
        except KeyError:
            colour = next(palette_iter)
            colour_map[label] = colour

        line_style = {"color": colour, "width": 1.6}
        if center_label == "생산중":
            line_style.update(dash="dash", width=1.2)
        elif center_label == "이동중":
            line_style.update(dash="dot", width=1.2)

        fig.add_trace(
            go.Scatter(
                x=group["date"],
                y=group["stock_qty"],
                mode="lines",
                name=label,
                line=line_style,
                line_shape="hv",
                hovertemplate=(
                    "날짜: %{x|%Y-%m-%d}<br>재고: %{y:,.0f} EA<extra>" + label + "</extra>"
                ),
            )
        )

    if today is not None:
        today_dt = pd.to_datetime(today).normalize()
        if start_dt <= today_dt <= end_dt:
            fig.add_vline(x=today_dt, line_color="rgba(255,0,0,0.5)", line_width=1)
            fig.add_annotation(
                x=today_dt,
                y=1.02,
                xref="x",
                yref="paper",
                text="오늘",
                showarrow=False,
                font=dict(color="#d62728", size=11),
                align="center",
            )

    fig.update_layout(
        title=title,
        hovermode="x unified",
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis=dict(title="날짜", range=[start_dt, end_dt], showgrid=True),
        yaxis=dict(title="재고량 (EA)", rangemode="tozero", tickformat=",.0f"),
        legend=dict(orientation="h", y=1.1, x=0),
    )

    if caption:
        st.caption(caption)
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

def render_amazon_panel(
    snap_long: pd.DataFrame,
    moves: Optional[pd.DataFrame] = None,
    centers: Optional[Sequence[str]] = None,
    skus: Optional[Sequence[str]] = None,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
    lookback_days: int = 7,
    today: Optional[pd.Timestamp] = None,
    show_ma7: bool = True,
    title: str = "Amazon US 일별 판매 vs. 재고",
    caption: str = "판매 막대(좌측 축) / 재고 선(우측 축). 오늘 이후 재고는 추세 기반 예측(점선)입니다.",
) -> None:
    """
    - 판매량: 스냅샷 감소분만 합산하여 계산(입고 증가분은 제외)
    - 재고: 오늘까지 실측 실선, 오늘 이후는 MA 기반 소진으로 점선
    - today 수직선(red) 표시
    """
    if skus is None:
        st.info("Amazon 패널을 표시하려면 SKU를 선택하세요.")
        return

    snap = _ensure_date_column(snap_long)

    # 표시 기간 기본값
    if today is None:
        today = pd.Timestamp.today().normalize()
    if start is None:
        start = (today - pd.Timedelta(days=20)).normalize()
    if end is None:
        end = (today + pd.Timedelta(days=20)).normalize()
    start = pd.to_datetime(start).normalize()
    end = pd.to_datetime(end).normalize()

    # Amazon 계열 센터 감지/필터
    all_centers = sorted(snap["center"].dropna().astype(str).unique().tolist())
    candidates = _detect_amazon_centers(centers or all_centers)
    if not candidates:
        st.caption("Amazon 계열 센터(AMZ*/amazon*)를 찾지 못했습니다.")
        return

    # 스냅샷 기반 재고/판매 시리즈
    inv_series = _sum_inventory_series(snap, candidates, skus, start, end)
    sales_series = _daily_sales_from_snapshot(snap, candidates, skus, start, end)

    # 오늘 기준 이전/이후 분리
    inv_today = float(inv_series.loc[inv_series.index <= today].iloc[-1]) if not inv_series.empty else 0.0

    # 7일 MA (요청 시 표시)
    ma7 = None
    if show_ma7 and not sales_series.empty:
        ma7 = sales_series.rolling(lookback_days, min_periods=max(3, lookback_days//2)).mean()

    # 미래 재고 예측: 오늘 이후 고정률 소진(최근 lookback일 판매 평균)
    if not sales_series.empty:
        recent = sales_series.loc[sales_series.index <= today].tail(lookback_days)
        daily_rate = float(recent.mean()) if len(recent) else 0.0
    else:
        daily_rate = 0.0

    future_idx = pd.date_range(max(today + pd.Timedelta(days=1), start), end, freq="D")
    if len(future_idx):
        # 누적 소진
        steps = np.arange(1, len(future_idx) + 1, dtype=float)
        inv_future = np.maximum(0.0, inv_today - daily_rate * steps)
        inv_future = pd.Series(inv_future, index=future_idx)
    else:
        inv_future = pd.Series(dtype=float)

    # ----- Plotly -----
    fig = go.Figure()

    # 1) 판매(막대, 좌측 y축)
    if not sales_series.empty:
        s_show = sales_series.reindex(pd.date_range(start, end, freq="D")).fillna(0.0)
        fig.add_bar(
            x=s_show.index, y=s_show.values,
            name="판매량(스냅샷 감소)",
            opacity=0.55,
            yaxis="y1",
            hovertemplate="날짜: %{x|%Y-%m-%d}<br>판매: %{y:,.0f} EA<extra></extra>",
        )

    # 1-1) 7일 MA 라인(선택)
    if ma7 is not None:
        m_show = ma7.reindex(pd.date_range(start, end, freq="D")).fillna(None)
        fig.add_trace(
            go.Scatter(
                x=m_show.index, y=m_show.values,
                name=f"판매 7일 MA",
                mode="lines",
                line=dict(width=2, dash="dot"),
                yaxis="y1",
                hovertemplate="날짜: %{x|%Y-%m-%d}<br>7일MA: %{y:,.1f} EA<extra></extra>",
            )
        )

    # 2) 재고 실선(오늘까지, 우측 y2)
    if not inv_series.empty:
        inv_past = inv_series.loc[(inv_series.index >= start) & (inv_series.index <= min(today, end))]
        fig.add_trace(
            go.Scatter(
                x=inv_past.index, y=inv_past.values,
                name="재고(실측)",
                mode="lines",
                line=dict(width=2),
                yaxis="y2",
                hovertemplate="날짜: %{x|%Y-%m-%d}<br>재고: %{y:,.0f} EA<extra></extra>",
            )
        )

    # 3) 재고 점선(오늘 이후 예측, 우측 y2)
    if len(inv_future):
        fig.add_trace(
            go.Scatter(
                x=inv_future.index, y=inv_future.values,
                name="재고(예측)",
                mode="lines",
                line=dict(width=2, dash="dash"),
                yaxis="y2",
                hovertemplate="날짜: %{x|%Y-%m-%d}<br>예측 재고: %{y:,.0f} EA<extra></extra>",
            )
        )

    # 오늘 수직선
    fig.add_vline(x=today, line_color="red", line_dash="dot")
    fig.add_annotation(
        x=today, y=1.02, xref="x", yref="paper",
        text="오늘",
        showarrow=False, font=dict(color="red", size=11)
    )

    # 레이아웃(축/범례 정리)
    fig.update_layout(
        title=title,
        hovermode="x unified",
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", y=1.12, x=0),
        xaxis=dict(
            title="Date",
            range=[start, end],
            showgrid=True, zeroline=False
        ),
        yaxis=dict(  # 좌측: 판매량
            title="판매량 (EA)",
            rangemode="tozero",
            showgrid=True,
            zeroline=False
        ),
        yaxis2=dict(  # 우측: 재고
            title="재고 (EA)",
            overlaying="y",
            side="right",
            showgrid=False,  # 보조축 그리드 제거 → 덜 어지럽게
            zeroline=False
        ),
        barmode="overlay",
    )

    if caption:
        st.caption(caption)
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
