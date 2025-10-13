# scm_dashboard_v5/ui/charts.py
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------- Palette (SKU -> Color) ----------------
# 계단식 차트와 최대한 비슷한 톤(20+색)
_PALETTE = [
    "#4E79A7","#F28E2B","#E15759","#76B7B2","#59A14F",
    "#EDC948","#B07AA1","#FF9DA7","#9C755F","#BAB0AC",
    "#1F77B4","#FF7F0E","#2CA02C","#D62728","#9467BD",
    "#8C564B","#E377C2","#7F7F7F","#BCBD22","#17BECF",
]

def _sku_colors(skus: Sequence[str], base: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """SKU 별 고정 색상 매핑을 만든다. (기존 매핑을 넘기면 그대로 존중)"""
    cmap = {} if base is None else dict(base)
    i = 0
    for s in skus:
        if s not in cmap:
            cmap[s] = _PALETTE[i % len(_PALETTE)]
            i += 1
    return cmap

def _pick_amazon_centers(all_centers: Iterable[str]) -> List[str]:
    """선택 센터 중 Amazon 계열만 추출 (없으면 자동 감지에 사용)"""
    out = []
    for c in all_centers:
        if not c:
            continue
        c_up = str(c).upper()
        if c_up.startswith("AMZ") or "AMAZON" in c_up:
            out.append(str(c))
    return out

# ---------------- Core helpers ----------------
def _coerce_cols(df: pd.DataFrame) -> Dict[str, str]:
    cols = {c.lower(): c for c in df.columns}
    date_col   = cols.get("snapshot_date") or cols.get("date")
    center_col = cols.get("center")
    sku_col    = cols.get("resource_code") or cols.get("sku")
    qty_col    = cols.get("stock_qty") or cols.get("qty") or cols.get("quantity")
    return {"date": date_col, "center": center_col, "sku": sku_col, "qty": qty_col}

def _sales_from_snapshot(
    snap_long: pd.DataFrame,
    centers: List[str],
    skus: List[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """
    스냅샷의 일간 차분으로 '판매(실측)'만 계산.
    - 증가분(+)은 입고로 보고 판매에서 제외
    - 감소분(-)만 판매로 본다
    반환: index=date, columns=sku, 값=EA/Day
    """
    c = _coerce_cols(snap_long)
    s = snap_long.rename(
        columns={c["date"]: "date", c["center"]: "center", c["sku"]: "resource_code", c["qty"]: "stock_qty"}
    )[["date", "center", "resource_code", "stock_qty"]].copy()

    s["date"] = pd.to_datetime(s["date"], errors="coerce").dt.normalize()
    s = s[s["center"].astype(str).isin(centers) & s["resource_code"].astype(str).isin(skus)]
    if s.empty:
        idx = pd.date_range(start, end, freq="D")
        return pd.DataFrame(0, index=idx, columns=skus)

    pv = (s.groupby(["date", "resource_code"])["stock_qty"].sum()
            .unstack("resource_code")
            .reindex(columns=skus, fill_value=0)
            .sort_index())
    pv = pv.asfreq("D").ffill()  # D 간격 보정
    d  = pv.diff().fillna(0)
    sales = (-d).clip(lower=0)  # 감소분만 판매
    sales = sales.loc[(sales.index >= start) & (sales.index <= end)]
    return sales

def _inventory_matrix(
    snap_long: pd.DataFrame,
    centers: List[str],
    skus: List[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """선택 센터×SKU의 재고(실측) 시계열 매트릭스. index=date, columns=sku"""
    c = _coerce_cols(snap_long)
    s = snap_long.rename(
        columns={c["date"]: "date", c["center"]: "center", c["sku"]: "resource_code", c["qty"]: "stock_qty"}
    )[["date", "center", "resource_code", "stock_qty"]].copy()

    s["date"] = pd.to_datetime(s["date"], errors="coerce").dt.normalize()
    s = s[s["center"].astype(str).isin(centers) & s["resource_code"].astype(str).isin(skus)]
    if s.empty:
        idx = pd.date_range(start, end, freq="D")
        return pd.DataFrame(0, index=idx, columns=skus)

    pv = (s.groupby(["date", "resource_code"])["stock_qty"].sum()
            .unstack("resource_code")
            .reindex(columns=skus, fill_value=0)
            .sort_index())
    pv = pv.asfreq("D").ffill()
    pv = pv.loc[(pv.index >= start) & (pv.index <= end)]
    return pv

# ---------------- Public renderer ----------------
def render_amazon_panel(
    snap_long: pd.DataFrame,
    centers: Sequence[str],
    skus: Sequence[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    today: pd.Timestamp,
    *,
    color_map: Optional[Dict[str, str]] = None,
    show_ma7: bool = True,
) -> None:
    """
    Amazon US 일별 판매(누적 막대) vs. 재고(라인) 패널.
    - 판매(실측): 스냅샷 감소분만 사용
    - 판매(예측): 최근 7일(옵션) 평균을 오늘 다음 날부터 적용 (막대 색상 동일)
    - 재고(실측): 실선, 재고(예측): 오늘 이후 점선
    - SKU별 색상 고정, 계단형 라인(hv), 오늘 세로 기준선 추가
    """
    skus = [str(s) for s in skus]
    if not skus:
        st.info("선택된 SKU가 없습니다.")
        return

    amz_centers = _pick_amazon_centers(centers)
    if not amz_centers:
        # 그래도 AMZ 관련 센터가 스냅샷에 있다면 자동 감지
        amz_centers = _pick_amazon_centers(snap_long.get("center", pd.Series()).dropna().unique())

    if not amz_centers:
        st.info("Amazon/AMZ 계열 센터가 보이지 않습니다.")
        return

    # 색상 고정
    cmap = _sku_colors(skus, base=color_map)

    # 판매(실측) 및 7일 평균
    sales = _sales_from_snapshot(snap_long, amz_centers, skus, start, end)
    ma7   = sales.rolling(7, min_periods=1).mean() if show_ma7 else None

    # 재고(실측)
    inv = _inventory_matrix(snap_long, amz_centers, skus, start, end)

    # 재고 예측(오늘 이후): MA7로 일일 차감하여 선형으로 감소
    future_idx = pd.date_range(today + pd.Timedelta(days=1), end, freq="D")
    inv_future = None
    if len(future_idx) > 0:
        start_vector = inv.loc[inv.index <= today].iloc[[-1]].copy() if (inv.index <= today).any() else pd.DataFrame([np.zeros(len(skus))], columns=skus)
        if ma7 is None or ma7.empty:
            daily = pd.DataFrame(0, index=future_idx, columns=skus)
        else:
            daily = ma7.reindex(future_idx).fillna(method="ffill").fillna(0)
        # 누적 차감
        cur = start_vector.iloc[0].astype(float).values
        vals = []
        for d in future_idx:
            cur = np.maximum(0.0, cur - daily.loc[d].reindex(skus, fill_value=0).values)
            vals.append(cur.copy())
        inv_future = pd.DataFrame(vals, index=future_idx, columns=skus)

    # ---------- Figure ----------
    fig = go.Figure()

    # (1) 판매: SKU별로 누적(bar)
    past_sales = sales.loc[sales.index <= today]
    # 실측 막대
    for sku in skus:
        if sku not in past_sales.columns:
            continue
        fig.add_bar(
            name=f"{sku} 판매",
            x=past_sales.index,
            y=past_sales[sku],
            marker_color=cmap[sku],
            opacity=0.95,
            hovertemplate="날짜 %{x|%Y-%m-%d}<br>%{fullData.name}: %{y:,} EA<extra></extra>",
            yaxis="y",
        )
    # 예측 막대(색상 동일, 투명도만 낮춤)
    if len(future_idx) > 0:
        if ma7 is None:
            future_sales = pd.DataFrame(0, index=future_idx, columns=skus)
        else:
            future_sales = ma7.reindex(future_idx).fillna(method="ffill").fillna(0)
        for sku in skus:
            fig.add_bar(
                name=f"{sku} 판매(예측)",
                x=future_sales.index,
                y=future_sales[sku],
                marker_color=cmap[sku],
                opacity=0.25,     # 같은 색, 낮은 불투명도
                hovertemplate="날짜 %{x|%Y-%m-%d}<br>%{fullData.name}: %{y:,} EA<extra></extra>",
                yaxis="y",
            )

    # (2) 재고: SKU별 선(실선), 오늘 이후는 점선
    inv_past = inv.loc[inv.index <= today]
    for sku in skus:
        if sku in inv_past.columns:
            fig.add_trace(
                go.Scatter(
                    x=inv_past.index,
                    y=inv_past[sku],
                    name=f"{sku} 재고(실측)",
                    mode="lines",
                    line=dict(color=cmap[sku], width=2, shape="hv"),
                    yaxis="y2",
                    hovertemplate="날짜 %{x|%Y-%m-%d}<br>%{fullData.name}: %{y:,} EA<extra></extra>",
                )
            )
    if inv_future is not None and not inv_future.empty:
        for sku in skus:
            if sku in inv_future.columns:
                fig.add_trace(
                    go.Scatter(
                        x=inv_future.index,
                        y=inv_future[sku],
                        name=f"{sku} 재고(예측)",
                        mode="lines",
                        line=dict(color=cmap[sku], width=2, dash="dash", shape="hv"),
                        yaxis="y2",
                        hovertemplate="날짜 %{x|%Y-%m-%d}<br>%{fullData.name}: %{y:,} EA<extra></extra>",
                    )
                )

    # (3) 오늘 기준선
    fig.add_vline(x=today, line_color="red", line_dash="dot", line_width=2)

    # 레이아웃 (제목 겹침 제거용 여백 포함)
    fig.update_layout(
        barmode="stack",
        margin=dict(l=16, r=16, t=40, b=40),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(title="Date"),
        yaxis=dict(title="판매량 (EA/Day)"),
        yaxis2=dict(title="재고 (EA)", overlaying="y", side="right"),
    )

    # 안내문(상단 설명을 그림 안에 넣지 않아 제목 겹침 방지)
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
