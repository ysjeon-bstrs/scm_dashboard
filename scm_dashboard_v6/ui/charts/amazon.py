from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scm_dashboard_v6.ui.charts.utils import sku_color_map


def _build_amazon_sales_vs_inventory_fig(ctx: Any) -> go.Figure:
    """
    v6 전용 아마존 차트 렌더러.

    - snap_정제.sales_qty를 과거 실측 막대로 사용
    - lookback_days로 계산한 미래 예측(프로모션 가중치 곱) + 입고/재고 캡핑
    - 인벤토리(실측/예측) 라인은 스텝 형태로 표시
    """

    start = pd.to_datetime(getattr(ctx, "start"))
    end = pd.to_datetime(getattr(ctx, "end"))
    today = pd.to_datetime(getattr(ctx, "today"))
    lookback_days = int(getattr(ctx, "lookback_days", 28) or 28)
    # v6에서는 기간형 이벤트 가중치만 사용 (글로벌 멀티플라이어 X)
    promo_mult = 1.0
    promo_events = list(getattr(ctx, "promotion_events", []))

    df = getattr(ctx, "snapshot_long").copy()
    df["date"] = pd.to_datetime(df.get("date"), errors="coerce").dt.normalize()
    df = df.dropna(subset=["date"]) 
    centers = list(getattr(ctx, "centers") or [])
    skus = list(getattr(ctx, "skus") or [])
    df = df[df["center"].isin(centers) & df["resource_code"].isin(skus)]

    # 실측 판매
    if "sales_qty" not in df.columns:
        df["sales_qty"] = 0.0
    df["sales_qty"] = pd.to_numeric(df["sales_qty"], errors="coerce").fillna(0.0)
    display_start = start.normalize()
    display_end = end.normalize()
    sales_hist = (
        df[(df["date"] <= today) & (df["date"] >= display_start) & (df["date"] <= display_end)][["date","resource_code","sales_qty"]]
        .groupby(["date","resource_code"], as_index=False)["sales_qty"].sum()
        .sort_values("date")
    )

    # 인벤토리(실측/예측)
    inv_actual = getattr(ctx, "inv_actual", pd.DataFrame()).copy()
    inv_forecast = getattr(ctx, "inv_forecast", pd.DataFrame()).copy()
    for inv in (inv_actual, inv_forecast):
        if not inv.empty:
            inv["date"] = pd.to_datetime(inv.get("date"), errors="coerce").dt.normalize()
            inv["stock_qty"] = pd.to_numeric(inv.get("stock_qty"), errors="coerce").fillna(0.0)
            inv["resource_code"] = inv.get("resource_code", "").astype(str)
    inv_actual = inv_actual[(inv_actual["date"] <= today) & (inv_actual["resource_code"].isin(skus))]
    inv_actual = inv_actual[(inv_actual["date"] >= display_start) & (inv_actual["date"] <= display_end)]
    inv_forecast = inv_forecast[(inv_forecast["date"] > today) & (inv_forecast["resource_code"].isin(skus))]
    inv_forecast = inv_forecast[(inv_forecast["date"] >= display_start) & (inv_forecast["date"] <= display_end)]

    # 실측 마지막 값의 재고 수치를 예측의 첫 날짜에 복제하여 연결 앵커를 추가한다.
    if (
        display_start <= today <= display_end
        and not inv_actual.empty
        and not inv_forecast.empty
    ):
        last_candidates = inv_actual[inv_actual["date"] <= today]
        if not last_candidates.empty:
            last_actual = (
                last_candidates.sort_values(["resource_code", "date"])
                .groupby("resource_code")
                .tail(1)
                .loc[:, ["resource_code", "date", "stock_qty"]]
                .rename(columns={"date": "actual_date"})
            )
        else:
            last_actual = pd.DataFrame(columns=["resource_code", "actual_date", "stock_qty"])

        if not last_actual.empty:
            fc_start = (
                inv_forecast.sort_values(["resource_code", "date"]).groupby("resource_code", as_index=False).first()[
                    ["resource_code", "date"]
                ]
                .rename(columns={"date": "forecast_start"})
            )
            anchor = (
                last_actual.merge(fc_start, on="resource_code", how="inner")
                .assign(date=lambda df: pd.to_datetime(df["forecast_start"]).dt.normalize())
                .loc[:, ["date", "resource_code", "stock_qty"]]
            )
        else:
            anchor = pd.DataFrame(columns=["date", "resource_code", "stock_qty"])

        if not anchor.empty:
            inv_forecast = pd.concat([anchor, inv_forecast], ignore_index=True)
            inv_forecast = (
                inv_forecast.sort_values(["resource_code", "date"]).drop_duplicates(
                    subset=["resource_code", "date"], keep="last"
                )
            )

    # lookback 기반 평균 + 프로모션 곱
    mean_by_sku: dict[str, float] = {}
    debug_rows: list[dict] = []
    for sku, grp in sales_hist.groupby("resource_code", dropna=True):
        hist = grp.set_index("date")["sales_qty"].asfreq("D").fillna(0.0)
        base = float(hist.tail(max(1, lookback_days)).mean()) if not hist.empty else 0.0
        mean_by_sku[str(sku)] = max(0.0, base)
        debug_rows.append({"resource_code": str(sku), "base_ma": round(base, 2), "promo_mult": round(promo_mult, 3)})

    # 입고/재고 캡핑 시뮬레이션
    idx = pd.date_range(max(today + pd.Timedelta(days=1), display_start), display_end, freq="D")
    fc_rows = []
    for sku in skus:
        base_rate = float(mean_by_sku.get(str(sku), 0.0))
        if base_rate <= 0:
            continue
        latest_stock = float(
            inv_actual[inv_actual["resource_code"] == sku]
            .sort_values("date")["stock_qty"].iloc[-1]
        ) if not inv_actual.empty and (inv_actual["resource_code"] == sku).any() else 0.0

        inbound = (
            inv_forecast[inv_forecast["resource_code"] == sku]
            .set_index("date")["stock_qty"].diff().clip(lower=0.0)
            if not inv_forecast.empty else pd.Series(0.0, index=idx)
        )
        inbound = inbound.reindex(idx, fill_value=0.0)
        # 날짜별 프로모션 계수 적용 (이벤트 기간만 uplift)
        uplift = pd.Series(1.0, index=idx)
        if promo_events:
            for ev in promo_events:
                s = pd.to_datetime(ev.get("start"), errors="coerce")
                e = pd.to_datetime(ev.get("end"), errors="coerce")
                u = float(ev.get("uplift", 0.0))
                if pd.notna(s) and pd.notna(e):
                    s = max(s.normalize(), idx[0])
                    e = min(e.normalize(), idx[-1])
                    if s <= e:
                        uplift.loc[s:e] = uplift.loc[s:e] * (1.0 + max(-1.0, min(3.0, u)))

        remain = max(0.0, latest_stock)
        out_vals: list[float] = []
        for day in idx:
            remain += float(inbound.loc[day])
            want = base_rate * float(uplift.loc[day])
            sell = min(want, max(remain, 0.0))
            out_vals.append(sell)
            remain -= sell
        if out_vals:
            fc_rows.append(pd.DataFrame({"date": idx, "resource_code": sku, "sales_qty": out_vals}))

    sales_fc = pd.concat(fc_rows, ignore_index=True) if fc_rows else pd.DataFrame(columns=["date","resource_code","sales_qty"])

    # 재고 예측을 기준으로 품절 이후 판매 0으로 강제 클램프 (v5 동작 정렬)
    try:
        if not inv_forecast.empty and not sales_fc.empty:
            zero_map: dict[str, pd.Timestamp] = {}
            for sku, grp in inv_forecast.groupby("resource_code"):
                g = grp.sort_values("date")
                # 첫 0 이하 시점
                z = g.loc[g["stock_qty"] <= 0, "date"]
                if not z.empty:
                    zero_map[str(sku)] = pd.to_datetime(z.iloc[0]).normalize()
            if zero_map:
                sales_fc["date"] = pd.to_datetime(sales_fc["date"]).dt.normalize()
                def _clip_row(r):
                    z = zero_map.get(str(r["resource_code"]))
                    if z is not None and r["date"] > z:
                        return 0.0
                    return r["sales_qty"]
                sales_fc["sales_qty"] = sales_fc.apply(_clip_row, axis=1)
    except Exception:
        pass

    # 시각화
    colors = sku_color_map(skus)
    fig = go.Figure()
    if not sales_hist.empty:
        for sku, grp in sales_hist.groupby("resource_code"):
            fig.add_bar(
                x=grp["date"],
                y=grp["sales_qty"],
                name=f"{sku} 판매(실측)",
                marker_color=colors.get(sku, "#4E79A7"),
                opacity=0.95,
                hovertemplate="날짜: %{x|%Y-%m-%d}<br>판매: %{y:,.0f} EA<br>%{fullData.name}<extra></extra>",
            )
    if not sales_fc.empty:
        for sku, grp in sales_fc.groupby("resource_code"):
            fig.add_bar(
                x=grp["date"],
                y=grp["sales_qty"],
                name=f"{sku} 판매(예측)",
                marker_color=colors.get(sku, "#4E79A7"),
                opacity=0.45,
                hovertemplate="날짜: %{x|%Y-%m-%d}<br>판매: %{y:,.0f} EA<br>%{fullData.name}<extra></extra>",
            )
    if not inv_actual.empty:
        for sku, grp in inv_actual.groupby("resource_code"):
            fig.add_trace(
                go.Scatter(
                    x=grp["date"],
                    y=grp["stock_qty"],
                    mode="lines",
                    name=f"{sku} 재고(실측)",
                    line=dict(color=colors.get(sku, "#4E79A7"), width=2),
                    line_shape="hv",
                    yaxis="y2",
                    hovertemplate="날짜: %{x|%Y-%m-%d}<br>재고: %{y:,.0f} EA<br>%{fullData.name}<extra></extra>",
                )
            )
    if not inv_forecast.empty:
        for sku, grp in inv_forecast.groupby("resource_code"):
            fig.add_trace(
                go.Scatter(
                    x=grp["date"],
                    y=grp["stock_qty"],
                    mode="lines",
                    name=f"{sku} 재고(예측)",
                    line=dict(color=colors.get(sku, "#4E79A7"), width=2, dash="dot"),
                    line_shape="hv",
                    yaxis="y2",
                    hovertemplate="날짜: %{x|%Y-%m-%d}<br>재고: %{y:,.0f} EA<br>%{fullData.name}<extra></extra>",
                )
            )

    fig.add_vline(x=today, line_color="crimson", line_dash="dash", line_width=2)
    fig.update_layout(
        title="AMZ 일간 판매량 x 재고 흐름 (추세 반영)",
        margin=dict(t=80),
        xaxis_title="Date", yaxis_title="판매량 (EA/Day)",
        yaxis2=dict(title="재고 (EA)", overlaying="y", side="right", showgrid=False),
        legend_orientation="h", legend_y=-0.2,
        barmode="stack",
        hovermode="x unified",
    )
    fig.update_xaxes(showspikes=True, spikemode="across", spikethickness=1, spikedash="dot", spikecolor="rgba(0,0,0,0.5)")
    return fig


def render_amazon_sales_vs_inventory(ctx: Any, **kwargs: Any) -> None:
    # v6 렌더러만 사용 (폴백 제거)
    fig = _build_amazon_sales_vs_inventory_fig(ctx)
    st.plotly_chart(fig, use_container_width=True)


