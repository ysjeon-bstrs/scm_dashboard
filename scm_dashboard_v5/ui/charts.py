# charts.py
# -----------------------------------------------
# Amazon US 일별 판매 vs. 재고 차트 (v5용)
# -----------------------------------------------
from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ---------- 내부 유틸 ----------

def _pick_date_col(df: pd.DataFrame) -> str:
    for c in ["snapshot_date", "date"]:
        if c in df.columns:
            return c
    raise KeyError("snap_long에는 'snapshot_date' 또는 'date' 컬럼이 필요합니다.")


def _to_date(s) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def _series_daily_index(x: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Daily index로 리샘플해 forward-fill/fillna(0)하기 위한 헬퍼."""
    # x는 이미 날짜 index를 가진 Series여야 함
    idx = pd.date_range(start, end, freq="D")
    # asfreq로 missing day 생성 후 ffill
    x = x.sort_index().reindex(idx).ffill()
    return x


# ---------- A. 계산 유틸 ----------

def compute_daily_sales_from_snapshot(
    snap_long: pd.DataFrame,
    skus_sel: List[str],
    center: str = "AMZUS",
) -> pd.DataFrame:
    """
    스냅샷 재고에서 전일 대비 감소분만 '판매량'으로 계산.
    증가(입고)는 0으로 처리 → 입고가 판매로 튀는 현상 제거.
    반환: date, sales_actual, sales_roll7
    """
    dcol = _pick_date_col(snap_long)
    sub = snap_long.copy()
    sub[dcol] = _to_date(sub[dcol])
    sub["stock_qty"] = pd.to_numeric(sub["stock_qty"], errors="coerce").fillna(0)

    sub = sub[(sub["center"] == center) & (sub["resource_code"].isin(skus_sel))]
    if sub.empty:
        return pd.DataFrame(columns=["date", "sales_actual", "sales_roll7"])

    # 일자별 총 재고(선택 SKU 합계)
    inv = (sub.groupby(dcol, as_index=True)["stock_qty"]
              .sum()
              .sort_index())

    # 일일 판매량 = 전일재고 - 당일재고 (감소분만)
    sales = (inv.shift(1) - inv).clip(lower=0).fillna(0)

    # 7일 이동평균(실측 구간만)
    roll7 = sales.rolling(7, min_periods=1).mean()

    out = pd.DataFrame({
        "date": sales.index,
        "sales_actual": sales.values.astype(float),
        "sales_roll7": roll7.values.astype(float),
    })
    return out


def extract_inbound_schedule(
    moves: pd.DataFrame,
    skus_sel: List[str],
    center: str,
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
) -> pd.Series:
    """
    예측 입고 스케줄(미래)을 날짜별 합으로 반환.
    우선순위: pred_inbound_date > inbound_date > arrival_date
    """
    if moves is None or moves.empty:
        return pd.Series(dtype=float)

    df = moves.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # 날짜 후보 고르기
    date_col = None
    for cand in ["pred_inbound_date", "inbound_date", "arrival_date", "event_date"]:
        if cand in df.columns:
            date_col = cand
            break
    if date_col is None:
        return pd.Series(dtype=float)

    df[date_col] = _to_date(df[date_col])
    df["qty_ea"] = pd.to_numeric(df.get("qty_ea", 0), errors="coerce").fillna(0)

    sel = df[
        (df["to_center"].astype(str) == str(center)) &
        (df["resource_code"].astype(str).isin(skus_sel)) &
        (df["qty_ea"] > 0) &
        df[date_col].between(start_dt, end_dt)
    ]

    if sel.empty:
        return pd.Series(dtype=float)

    g = sel.groupby(date_col)["qty_ea"].sum().astype(float)
    return g


def forecast_sales_constant(
    sales_df: pd.DataFrame,
    start_forecast: pd.Timestamp,
    end_forecast: pd.Timestamp,
    w7: float = 0.7,
) -> pd.Series:
    """
    최근 7일/28일 평균을 가중(0.7/0.3)해 오늘 이후를 '상수'로 예측.
    """
    if sales_df.empty or start_forecast > end_forecast:
        return pd.Series(dtype=float)

    s = sales_df.set_index("date")["sales_actual"].astype(float)
    # 실측 구간이 없을 수 있음 → 0 처리
    mean7 = float(s.tail(7).mean()) if len(s) else 0.0
    mean28 = float(s.tail(28).mean()) if len(s) else 0.0
    rate = max(0.0, w7 * mean7 + (1 - w7) * mean28)

    idx = pd.date_range(start_forecast, end_forecast, freq="D")
    return pd.Series(rate, index=idx)


def build_inventory_series(
    snap_long: pd.DataFrame,
    skus_sel: List[str],
    center: str,
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
) -> pd.Series:
    """실측 스냅샷 재고(선택 SKU 합계)를 일자 시계열로."""
    dcol = _pick_date_col(snap_long)
    sub = snap_long.copy()
    sub[dcol] = _to_date(sub[dcol])
    sub["stock_qty"] = pd.to_numeric(sub["stock_qty"], errors="coerce").fillna(0)

    sub = sub[(sub["center"] == center) & (sub["resource_code"].isin(skus_sel))]
    if sub.empty:
        return pd.Series(dtype=float)

    inv = (sub.groupby(dcol)["stock_qty"]
              .sum()
              .sort_index())

    # 일자 전체 범위로 확장 + ffill
    inv = _series_daily_index(inv, start_dt, end_dt).fillna(0)
    return inv


def project_inventory(
    inv_today: float,
    start_next_day: pd.Timestamp,
    end_dt: pd.Timestamp,
    sales_forecast: pd.Series,   # 미래 날짜 index
    inbound_future: pd.Series,   # 미래 날짜 index
) -> pd.Series:
    """오늘 다음 날부터 예측 재고를 생성."""
    if start_next_day > end_dt:
        return pd.Series(dtype=float)

    idx = pd.date_range(start_next_day, end_dt, freq="D")
    inv = np.zeros(len(idx), dtype=float)

    cur = float(inv_today)
    sales_f = sales_forecast.reindex(idx).fillna(0.0)
    inbound_f = inbound_future.reindex(idx).fillna(0.0)

    for i, d in enumerate(idx):
        cur = max(0.0, cur - sales_f.iloc[i] + inbound_f.iloc[i])
        inv[i] = cur

    return pd.Series(inv, index=idx)


# ---------- B. 렌더러 ----------

def render_amazon_sales_vs_inventory(
    snap_long: pd.DataFrame,
    moves: pd.DataFrame,
    skus_sel: List[str],
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
    center: str = "AMZUS",
    show_roll7: bool = True,
) -> go.Figure:
    """
    Amazon US 일별 판매 vs. 재고 복합 차트.
      - 막대: 판매(실측, 오늘까지)
      - 점선(좌축): 판매(예측, 오늘 이후)
      - 실선(우축): 재고(실측, 오늘까지)
      - 점선(우축): 재고(예측, 오늘 이후)  ← 판매 예측/입고 스케줄 반영
    """
    today = pd.Timestamp.today().normalize()

    # 판매 실측/롤링
    sales_df = compute_daily_sales_from_snapshot(snap_long, skus_sel, center=center)
    # 차트 표시 범위로 슬라이싱
    sales_df = sales_df[(sales_df["date"] >= start_dt) & (sales_df["date"] <= end_dt)]

    # 재고(실측)
    inv_series = build_inventory_series(snap_long, skus_sel, center, start_dt, end_dt)

    # 판매 예측: 오늘+1 ~ end_dt
    fc_start = max(today + pd.Timedelta(days=1), start_dt)
    sales_fc = forecast_sales_constant(sales_df, fc_start, end_dt)  # 상수 추세

    # 미래 입고 스케줄 (표시는 하지 않지만 예측 재고에 반영)
    inbound_future = extract_inbound_schedule(moves, skus_sel, center, fc_start, end_dt)

    # 재고 예측(오늘 다음날부터)
    inv_today = float(inv_series.reindex([today]).ffill().iloc[-1]) if not inv_series.empty else 0.0
    inv_fc = project_inventory(inv_today, fc_start, end_dt, sales_fc, inbound_future)

    # ---- Plotly Figure ----
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    # 바: 판매(실측, 오늘까지)
    sales_actual = sales_df.set_index("date")["sales_actual"].clip(lower=0)
    sales_actual_past = sales_actual[sales_actual.index <= today]
    if not sales_actual_past.empty:
        fig.add_bar(
            x=sales_actual_past.index,
            y=sales_actual_past.values,
            name="판매(실측)",
            marker_color="#4C78A8",
            opacity=0.85,
            hovertemplate="날짜 %{x|%Y-%m-%d}<br>판매 %{y:,.0f} EA<extra></extra>",
            secondary_y=False,
        )

    # 점선: 판매(예측, 오늘 이후)
    if not sales_fc.empty:
        fig.add_scatter(
            x=sales_fc.index,
            y=sales_fc.values,
            name="판매(예측)",
            mode="lines",
            line=dict(color="#4C78A8", width=2, dash="dash"),
            hovertemplate="예측 판매 %{y:,.0f} EA<br>%{x|%Y-%m-%d}<extra></extra>",
            secondary_y=False,
        )

    # 실선: 재고(실측, 오늘까지)
    inv_past = inv_series[inv_series.index <= today]
    if not inv_past.empty:
        fig.add_scatter(
            x=inv_past.index,
            y=inv_past.values,
            name="재고(실측)",
            mode="lines",
            line=dict(color="#F58518", width=2.2),
            hovertemplate="재고 %{y:,.0f} EA<br>%{x|%Y-%m-%d}<extra></extra>",
            secondary_y=True,
        )

    # 점선: 재고(예측, 오늘 이후)
    if not inv_fc.empty:
        fig.add_scatter(
            x=inv_fc.index,
            y=inv_fc.values,
            name="재고(예측)",
            mode="lines",
            line=dict(color="#F58518", width=2.2, dash="dash"),
            hovertemplate="예측 재고 %{y:,.0f} EA<br>%{x|%Y-%m-%d}<extra></extra>",
            secondary_y=True,
        )

    # 7일 이동평균(실측 구간)
    if show_roll7:
        roll7 = sales_df.set_index("date")["sales_roll7"]
        roll7_past = roll7[roll7.index <= today]
        if not roll7_past.empty:
            fig.add_scatter(
                x=roll7_past.index,
                y=roll7_past.values,
                name="판매 7일 평균",
                mode="lines",
                line=dict(color="#72B7B2", width=2, dash="dot"),
                hovertemplate="7일 평균 %{y:,.0f} EA<br>%{x|%Y-%m-%d}<extra></extra>",
                secondary_y=False,
            )

    # 오늘 수직선
    if start_dt <= today <= end_dt:
        fig.add_vline(
            x=today,
            line_width=1.6,
            line_dash="solid",
            line_color="crimson",
            annotation_text="오늘",
            annotation_position="top",
            annotation_font_color="crimson",
        )

    # 레이아웃/축
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        bargap=0.25,
    )
    fig.update_xaxes(
        showgrid=True, gridcolor="rgba(0,0,0,0.05)",
        tickformat="%b %d", ticklabelmode="period",
    )
    fig.update_yaxes(
        title_text="판매량 (EA/Day)",
        showgrid=True, gridcolor="rgba(0,0,0,0.05)",
        secondary_y=False,
    )
    fig.update_yaxes(
        title_text="재고 (EA)",
        showgrid=False,
        secondary_y=True,
    )

    return fig
