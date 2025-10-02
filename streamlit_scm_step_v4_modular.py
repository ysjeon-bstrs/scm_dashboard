"""Modularised entry point for the SCM dashboard v4."""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from scm_dashboard_v4.config import CENTER_COL, PALETTE, configure_page, initialize_session_state
from scm_dashboard_v4.consumption import apply_consumption_with_events
from scm_dashboard_v4.inventory import pivot_inventory_cost_from_raw
from scm_dashboard_v4.kpi import kpi_breakdown_per_sku
from scm_dashboard_v4.loaders import load_from_excel, load_from_gsheet_api, load_snapshot_raw
from scm_dashboard_v4.processing import (
    merge_wip_as_moves,
    normalize_center_name,
    normalize_moves,
    normalize_refined_snapshot,
    load_wip_from_incoming,
)
from scm_dashboard_v4.sales import prepare_amazon_sales_series
from scm_dashboard_v4.timeline import build_timeline
from scm_dashboard_v4.sales import prepare_amazon_daily_sales

configure_page()
initialize_session_state()


# ==================== Tabs for inputs ====================
tab1, tab2 = st.tabs(["엑셀 업로드", "Google Sheets"])

moves: Optional[pd.DataFrame] = None
snap_long: Optional[pd.DataFrame] = None

with tab1:
    xfile = st.file_uploader("엑셀 업로드 (.xlsx)", type=["xlsx"], key="excel_modular")
    if xfile is not None:
        df_move, df_ref, df_incoming, snap_raw_df = load_from_excel(xfile)
        st.session_state["_data_source"] = "excel"
        st.session_state["_snapshot_raw_cache"] = snap_raw_df

        moves_raw = normalize_moves(df_move)
        snap_long = normalize_refined_snapshot(df_ref)

        try:
            wip_df = load_wip_from_incoming(df_incoming)
            moves = merge_wip_as_moves(moves_raw, wip_df)
            st.success(
                f"WIP {len(wip_df)}건 반영 완료" if wip_df is not None and not wip_df.empty else "WIP 없음"
            )
        except Exception as exc:
            moves = moves_raw
            st.warning(f"WIP 불러오기 실패: {exc}")

with tab2:
    st.info("Google Sheets API를 사용하여 데이터를 로드합니다.")
    st.caption("서비스 계정 키 파일을 사용하여 인증합니다.")

    if st.button("Google Sheets에서 데이터 로드", type="primary"):
        try:
            df_move, df_ref, df_incoming = load_from_gsheet_api()

            if df_move.empty or df_ref.empty:
                st.error("❌ Google Sheets API로 데이터를 불러올 수 없습니다. 서비스 계정 권한을 확인해주세요.")
                st.stop()

            st.session_state["_data_source"] = "gsheet"

            moves_raw = normalize_moves(df_move)
            snap_long = normalize_refined_snapshot(df_ref)
            try:
                wip_df = load_wip_from_incoming(df_incoming)
                moves = merge_wip_as_moves(moves_raw, wip_df)
                st.success(
                    f"✅ Google Sheets 로드 완료! WIP {len(wip_df)}건 반영" if wip_df is not None and not wip_df.empty else "✅ Google Sheets 로드 완료! WIP 없음"
                )
            except Exception as exc:
                moves = moves_raw
                st.warning(f"⚠️ WIP 불러오기 실패: {exc}")
        except Exception as exc:
            st.error(f"❌ Google Sheets 데이터 로드 중 오류가 발생했습니다: {exc}")
            st.info(
                "💡 해결 방법:\n- 서비스 계정 키 파일이 올바른지 확인\n- 스프레드시트에 서비스 계정 이메일이 공유되어 있는지 확인\n- 시트명이 정확한지 확인 (SCM_통합, snap_정제)"
            )

if moves is None or snap_long is None:
    try:
        df_move, df_ref, df_incoming = load_from_gsheet_api()
        if not df_move.empty and not df_ref.empty:
            st.session_state["_data_source"] = "gsheet"
            moves = normalize_moves(df_move)
            snap_long = normalize_refined_snapshot(df_ref)
            try:
                wip_df = load_wip_from_incoming(df_incoming)
                moves = merge_wip_as_moves(moves, wip_df)
            except Exception:
                pass
            st.success("✅ Google Sheets에서 데이터 로드됨 (필요 시 엑셀 업로드 탭 사용 가능)")
        else:
            st.info("엑셀 업로드 또는 Google Sheets에서 데이터를 로드하면 필터/차트가 나타납니다.")
            st.stop()
    except Exception:
        st.info("엑셀 업로드 또는 Google Sheets에서 데이터를 로드하면 필터/차트가 나타납니다.")
        st.stop()

assert moves is not None and snap_long is not None

# -------------------- Filters --------------------
centers_snap = set(snap_long["center"].dropna().astype(str).unique().tolist())
centers_moves = set(
    moves["from_center"].dropna().astype(str).unique().tolist() + moves["to_center"].dropna().astype(str).unique().tolist()
)

all_centers = set()
for center in centers_snap | centers_moves:
    normalized = normalize_center_name(center)
    if normalized:
        all_centers.add(normalized)

centers = sorted(list(all_centers))
skus = sorted(snap_long["resource_code"].dropna().astype(str).unique().tolist())

today = pd.Timestamp.today().normalize()
PAST_DAYS = 42
FUTURE_DAYS = 60

snap_min = pd.to_datetime(snap_long["date"]).min().normalize()
snap_max = pd.to_datetime(snap_long["date"]).max().normalize()

bound_min = max(today - pd.Timedelta(days=PAST_DAYS), snap_min)
bound_max = min(today + pd.Timedelta(days=FUTURE_DAYS), snap_max + pd.Timedelta(days=60))


def _init_range() -> None:
    if "date_range" not in st.session_state:
        st.session_state.date_range = (
            max(today - pd.Timedelta(days=20), bound_min),
            min(today + pd.Timedelta(days=20), bound_max),
        )
    if "horizon_days" not in st.session_state:
        st.session_state.horizon_days = 20


def _apply_horizon_to_range() -> None:
    h = int(st.session_state.horizon_days)
    h = max(0, min(h, FUTURE_DAYS))
    st.session_state.horizon_days = h
    start = max(today - pd.Timedelta(days=h), bound_min)
    end = min(today + pd.Timedelta(days=h), bound_max)
    st.session_state.date_range = (start, end)


def _clamp_range(r: tuple[pd.Timestamp, pd.Timestamp]) -> tuple[pd.Timestamp, pd.Timestamp]:
    s, e = pd.Timestamp(r[0]).normalize(), pd.Timestamp(r[1]).normalize()
    s = max(min(s, bound_max), bound_min)
    e = max(min(e, bound_max), bound_min)
    if e < s:
        e = s
    return (s, e)


_init_range()

st.sidebar.header("필터")
centers_sel = st.sidebar.multiselect("센터 선택", centers, default=(["태광KR"] if "태광KR" in centers else centers[:1]))
skus_sel = st.sidebar.multiselect("SKU 선택", skus, default=([s for s in ["BA00022", "BA00021"] if s in skus] or skus[:2]))

st.sidebar.subheader("기간 설정")
st.sidebar.number_input(
    "미래 전망 일수",
    min_value=0,
    max_value=FUTURE_DAYS,
    step=1,
    key="horizon_days",
    on_change=_apply_horizon_to_range,
)

range_value = st.sidebar.slider(
    "기간",
    min_value=bound_min.to_pydatetime(),
    max_value=bound_max.to_pydatetime(),
    value=tuple(d.to_pydatetime() for d in _clamp_range(st.session_state.date_range)),
    format="YYYY-MM-DD",
)

start_dt = pd.Timestamp(range_value[0]).normalize()
end_dt = pd.Timestamp(range_value[1]).normalize()
_latest_snap = snap_long["date"].max()
proj_days_for_build = max(0, int((end_dt - _latest_snap).days))

st.sidebar.header("표시 옵션")
show_prod = st.sidebar.checkbox("생산중(미완료) 표시", value=True)
use_cons_forecast = st.sidebar.checkbox("추세 기반 재고 예측", value=True)
lookback_days = st.sidebar.number_input("추세 계산 기간(일)", min_value=7, max_value=56, value=28, step=7)

st.sidebar.subheader("입고 반영 가정")
lag_days = st.sidebar.number_input(
    "입고 반영 리드타임(일) – inbound 미기록 시 arrival+N",
    min_value=0,
    max_value=21,
    value=7,
    step=1,
)

with st.sidebar.expander("프로모션 가중치(+%)", expanded=False):
    enable_event = st.checkbox("가중치 적용", value=False)
    ev_start = st.date_input("시작일")
    ev_end = st.date_input("종료일")
    ev_pct = st.number_input("가중치(%)", min_value=-100.0, max_value=300.0, value=30.0, step=5.0)

if enable_event:
    events = [
        {
            "start": pd.Timestamp(ev_start).strftime("%Y-%m-%d"),
            "end": pd.Timestamp(ev_end).strftime("%Y-%m-%d"),
            "uplift": ev_pct / 100.0,
        }
    ]
else:
    events = []

# -------------------- KPIs (SKU별 분해) --------------------
st.subheader("요약 KPI")

_snap_date_col = "date" if "date" in snap_long.columns else "snapshot_date"
_latest_dt = pd.to_datetime(snap_long[_snap_date_col]).max().normalize()
_latest_dt_str = _latest_dt.strftime("%Y-%m-%d")

_name_col = None
for cand in ["resource_name", "상품명", "품명"]:
    if cand in snap_long.columns:
        _name_col = cand
        break
_name_map: Dict[str, str] = {}
if _name_col:
    name_rows = (
        snap_long[snap_long[_snap_date_col] == _latest_dt]
        .dropna(subset=["resource_code"])[["resource_code", _name_col]]
        .drop_duplicates()
    )
    _name_map = dict(zip(name_rows["resource_code"].astype(str), name_rows[_name_col].astype(str)))

_today = pd.Timestamp.today().normalize()
mv = moves.copy()
mv["carrier_mode"] = mv["carrier_mode"].astype(str).str.upper()
mv["resource_code"] = mv["resource_code"].astype(str)

kpi_df = kpi_breakdown_per_sku(
    snap_long,
    mv,
    centers_sel,
    skus_sel,
    _today,
    _snap_date_col,
    _latest_dt,
    int(lag_days),
)


def _chunks(lst: List[str], n: int):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


for group in _chunks(skus_sel, 2):
    cols = st.columns(len(group))
    for i, sku in enumerate(group):
        with cols[i].container(border=True):
            name = _name_map.get(sku, "")
            if name:
                st.markdown(f"**{name}**  \n`{sku}`")
            else:
                st.markdown(f"`{sku}`")
            c1, c2, c3 = st.columns(3)
            c1.metric("현재 재고", f"{kpi_df.loc[sku, 'current']:,}")
            c2.metric("이동중", f"{kpi_df.loc[sku, 'in_transit']:,}")
            c3.metric("생산중", f"{kpi_df.loc[sku, 'wip']:,}")

st.divider()

# -------------------- Step Chart --------------------
st.subheader("계단식 재고 흐름")

timeline = build_timeline(
    snap_long,
    moves,
    centers_sel,
    skus_sel,
    start_dt,
    end_dt,
    horizon_days=proj_days_for_build,
    today=today,
    lag_days=int(lag_days),
)

if use_cons_forecast and not timeline.empty:
    timeline = apply_consumption_with_events(
        timeline,
        snap_long,
        centers_sel,
        skus_sel,
        start_dt,
        end_dt,
        lookback_days=int(lookback_days),
        events=events,
    )

if timeline.empty:
    st.info("선택 조건에 해당하는 타임라인 데이터가 없습니다.")
else:
    vis_df = timeline[(timeline["date"] >= start_dt) & (timeline["date"] <= end_dt)].copy()
    vis_df.loc[vis_df["center"] == "WIP", "center"] = "생산중"
    if "태광KR" not in centers_sel:
        vis_df = vis_df[vis_df["center"] != "생산중"]
    if not show_prod:
        vis_df = vis_df[vis_df["center"] != "생산중"]

    vis_df = vis_df[vis_df["stock_qty"] > 0]

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
        legend_title_text="SKU @ Center / 생산중(점선)",
        margin=dict(l=20, r=20, t=60, b=20),
    )


    if vis_df.empty:
        st.info("선택한 조건에서 표시할 센터/생산중 데이터가 없습니다.")
    else:
        vis_df["label"] = vis_df["resource_code"] + " @ " + vis_df["center"]

        fig = px.line(
            vis_df,
            x="date",
            y="stock_qty",
            color="label",
            line_shape="hv",
            title="선택한 SKU × 센터(및 생산중) 계단식 재고 흐름",
            render_mode="svg",
        )
        fig.update_layout(
            hovermode="x unified",
            xaxis_title="날짜",
            yaxis_title="재고량(EA)",
            legend_title_text="SKU @ Center / 생산중(점선)",
            margin=dict(l=20, r=20, t=60, b=20),
        )

        if start_dt <= today <= end_dt:
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

        fig.update_yaxes(tickformat=",.0f")
        fig.update_traces(
            hovertemplate="날짜: %{x|%Y-%m-%d}<br>재고: %{y:,.0f} EA<br>%{fullData.name}<extra></extra>"
        )

        line_colors: Dict[str, str] = {}
        color_idx = 0
        for tr in fig.data:
            name = tr.name or ""
            if " @ " in name and name not in line_colors:
                line_colors[name] = PALETTE[color_idx % len(PALETTE)]
                color_idx += 1
        for i, tr in enumerate(fig.data):
            name = tr.name or ""
            if " @ " not in name:
                continue
            _, kind = name.split(" @ ", 1)
            line_color = line_colors.get(name, PALETTE[0])
            if kind == "생산중":
                fig.data[i].update(line=dict(color=line_color, dash="dash", width=1.0), opacity=0.8)
            else:
                fig.data[i].update(line=dict(color=line_color, dash="solid", width=1.5), opacity=1.0)

        chart_key = (
            f"stepchart|centers={','.join(centers_sel)}|skus={','.join(skus_sel)}|"
            f"{start_dt:%Y%m%d}-{end_dt:%Y%m%d}|h{int(st.session_state.horizon_days)}|"
            f"prod{int(show_prod)}"
        )
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False}, key=chart_key)

# ==================== Amazon US Sales vs Inventory ====================
amazon_centers = sorted({c for c in snap_long["center"].unique() if "amazon" in str(c).lower()})
selected_amazon_centers = [c for c in centers_sel if c in amazon_centers]

sales_series = prepare_amazon_daily_sales(
    snap_long,
    centers=selected_amazon_centers or amazon_centers,
    skus=skus_sel,
    rolling_window=7,
)

if not sales_series.empty:
    sales_df = sales_series.frame
    # This chart shows Amazon-related snapshot totals with derived sales deltas.
    # Users can toggle individual traces (sales, inventory, rolling avg) via the legend.
    st.divider()
    st.subheader("Amazon US 일별 판매 vs. 재고")

    chart = make_subplots(specs=[[{"secondary_y": True}]])
    chart.add_trace(
        go.Bar(
            x=sales_df["date"],
            y=sales_df["daily_sales"],
            name="Daily Sales (EA)",
            marker_color=PALETTE[0],
        ),
        secondary_y=False,
    )
    chart.add_trace(
        go.Scatter(
            x=sales_df["date"],
            y=sales_df["inventory_qty"],
            mode="lines+markers",
            name="Amazon Inventory (EA)",
            line=dict(color=PALETTE[1], width=2),
        ),
        secondary_y=True,
    )


    line_colors: Dict[str, str] = {}
    color_idx = 0
    for tr in fig.data:
        name = tr.name or ""
        if " @ " in name and name not in line_colors:
            line_colors[name] = PALETTE[color_idx % len(PALETTE)]
            color_idx += 1
    for i, tr in enumerate(fig.data):
        name = tr.name or ""
        if " @ " not in name:
            continue
        _, kind = name.split(" @ ", 1)
        line_color = line_colors.get(name, PALETTE[0])
        if kind == "생산중":
            fig.data[i].update(line=dict(color=line_color, dash="dash", width=1.0), opacity=0.8)
        else:
            fig.data[i].update(line=dict(color=line_color, dash="solid", width=1.5), opacity=1.0)

    chart_key = (
        f"stepchart|centers={','.join(centers_sel)}|skus={','.join(skus_sel)}|"
        f"{start_dt:%Y%m%d}-{end_dt:%Y%m%d}|h{int(st.session_state.horizon_days)}|"
        f"prod{int(show_prod)}"
    )
    sales_fig.update_yaxes(tickformat=",.0f")
    st.plotly_chart(sales_fig, use_container_width=True, config={"displaylogo": False})

# -------------------- Amazon US sales vs. inventory --------------------
st.subheader("Amazon US 일일 판매 & 재고")
sales_result = prepare_amazon_sales_series(snap_long, skus_sel, start_dt, end_dt)
sales_df = sales_result.data

if sales_df.empty:
    st.caption("선택된 SKU/기간에 대한 Amazon US 판매 데이터가 없습니다.")
else:
    sales_fig = go.Figure()
    sales_fig.add_bar(
        x=sales_df["date"],
        y=sales_df["sales_qty"],
        name="일일 판매량",
        marker_color="#ff7f0e",
        opacity=0.8,
    )
    sales_fig.add_scatter(
        x=sales_df["date"],
        y=sales_df["inventory_qty"],
        name="Amazon 재고",
        mode="lines",
        line=dict(color="#1f77b4", width=2),
        yaxis="y2",
    )
    sales_fig.add_scatter(
        x=sales_df["date"],
        y=sales_df["sales_roll_mean"],
        name="판매 7일 이동평균",
        mode="lines",
        line=dict(color="#d62728", dash="dash"),
        visible="legendonly",
    )

    sales_fig.update_layout(
        barmode="overlay",
        hovermode="x unified",
        legend_title_text="Amazon 판매/재고",
        xaxis=dict(title="날짜"),
        yaxis=dict(title="일일 판매량(EA)"),
        yaxis2=dict(title="Amazon 재고(EA)", overlaying="y", side="right"),
        margin=dict(l=20, r=40, t=40, b=20),
    )
    sales_fig.update_yaxes(tickformat=",.0f")
    st.plotly_chart(sales_fig, use_container_width=True, config={"displaylogo": False})

# -------------------- Upcoming Arrivals (fixed) --------------------
st.subheader("입고 예정 내역 (선택 센터/SKU)")
window_start = start_dt
window_end = end_dt

mv_view = mv.copy()
if not mv_view.empty:
    pred_inbound = pd.Series(pd.NaT, index=mv_view.index, dtype="datetime64[ns]")

    mask_inb = mv_view["inbound_date"].notna()
    pred_inbound.loc[mask_inb] = mv_view.loc[mask_inb, "inbound_date"]

    mask_arr = (~mask_inb) & mv_view["arrival_date"].notna()
    if mask_arr.any():
        past_arr = mask_arr & (mv_view["arrival_date"] <= today)
        pred_inbound.loc[past_arr] = mv_view.loc[past_arr, "arrival_date"] + pd.Timedelta(days=int(lag_days))

        fut_arr = mask_arr & (mv_view["arrival_date"] > today)
        pred_inbound.loc[fut_arr] = mv_view.loc[fut_arr, "arrival_date"]

    mv_view["pred_inbound_date"] = pred_inbound

arr_transport = mv_view[
    (mv_view["carrier_mode"] != "WIP")
    & (mv_view["to_center"].isin(centers_sel))
    & (mv_view["resource_code"].isin(skus_sel))
    & (mv_view["inbound_date"].isna())
].copy()

arr_transport["display_date"] = arr_transport["arrival_date"].fillna(arr_transport["onboard_date"])
arr_transport = arr_transport[arr_transport["display_date"].notna()]
arr_transport = arr_transport[
    (arr_transport["display_date"] >= window_start) & (arr_transport["display_date"] <= window_end)
]

arr_wip = pd.DataFrame()
if "태광KR" in centers_sel:
    arr_wip = mv_view[
        (mv_view["carrier_mode"] == "WIP")
        & (mv_view["to_center"] == "태광KR")
        & (mv_view["resource_code"].isin(skus_sel))
        & (mv_view["event_date"].notna())
        & (mv_view["event_date"] >= window_start)
        & (mv_view["event_date"] <= window_end)
    ].copy()
    arr_wip["display_date"] = arr_wip["event_date"]

confirmed_inbound = arr_transport.copy()
if _name_map and not confirmed_inbound.empty:
    confirmed_inbound["resource_name"] = confirmed_inbound["resource_code"].map(_name_map).fillna("")

st.markdown("#### ✅ 확정 입고 (Upcoming Inbound)")
if confirmed_inbound.empty:
    st.caption("선택한 조건에서 예정된 운송 입고가 없습니다. (오늘 이후 / 선택 기간)")
else:
    confirmed_inbound["days_to_arrival"] = (
        confirmed_inbound["display_date"].dt.normalize() - today
    ).dt.days
    confirmed_inbound["days_to_inbound"] = (
        confirmed_inbound["pred_inbound_date"].dt.normalize() - today
    ).dt.days
    confirmed_inbound = confirmed_inbound.sort_values(
        ["display_date", "to_center", "resource_code", "qty_ea"],
        ascending=[True, True, True, False],
    )
    inbound_cols = [
        "display_date",
        "days_to_arrival",
        "to_center",
        "resource_code",
        "resource_name",
        "qty_ea",
        "carrier_mode",
        "onboard_date",
        "pred_inbound_date",
        "days_to_inbound",
        "lot",
    ]
    inbound_cols = [c for c in inbound_cols if c in confirmed_inbound.columns]
    st.dataframe(
        confirmed_inbound[inbound_cols].head(1000),
        use_container_width=True,
        height=300,
    )
    st.caption("※ pred_inbound_date: 예상 입고일 (도착일 + 리드타임), days_to_inbound: 예상 입고까지 남은 일수")

if not arr_wip.empty:
    if _name_map:
        arr_wip["resource_name"] = arr_wip["resource_code"].map(_name_map).fillna("")
    st.markdown("#### 🛠 생산중 (WIP) 진행 현황")
    arr_wip = arr_wip.sort_values(
        ["display_date", "resource_code", "qty_ea"], ascending=[True, True, False]
    )
    arr_wip["days_to_completion"] = (
        arr_wip["display_date"].dt.normalize() - today
    ).dt.days
    wip_cols = [
        "display_date",
        "days_to_completion",
        "resource_code",
        "resource_name",
        "qty_ea",
        "pred_inbound_date",
        "lot",
    ]
    wip_cols = [c for c in wip_cols if c in arr_wip.columns]
    st.dataframe(arr_wip[wip_cols].head(1000), use_container_width=True, height=260)
else:
    st.caption("생산중(WIP) 데이터가 없습니다.")

# -------------------- 선택 센터 현재 재고 (전체 SKU) --------------------
st.subheader(f"선택 센터 현재 재고 (스냅샷 {_latest_dt_str} / 전체 SKU)")

cur = snap_long[(snap_long["date"] == _latest_dt) & (snap_long["center"].isin(centers_sel))].copy()
pivot = (
    cur.groupby(["resource_code", "center"], as_index=False)["stock_qty"].sum()
    .pivot(index="resource_code", columns="center", values="stock_qty")
    .fillna(0)
    .astype(int)
)
pivot["총합"] = pivot.sum(axis=1)

col1, col2 = st.columns([2, 1])
with col1:
    q = st.text_input(
        "SKU 필터 — 품목번호 검색 시 해당 SKU의 센터별 제조번호(LOT) 확인",
        "",
        key="sku_filter_text_modular",
    )
with col2:
    sort_by = st.selectbox("정렬 기준", ["총합"] + list(pivot.columns.drop("총합")), index=0)

col1, col2 = st.columns(2)
with col1:
    hide_zero = st.checkbox("총합=0 숨기기", value=True)
with col2:
    show_cost = st.checkbox("재고자산(제조원가) 표시", value=False)

view = pivot.copy()
if q.strip():
    view = view[view.index.str.contains(q.strip(), case=False, regex=False)]
if hide_zero:
    view = view[view["총합"] > 0]
view = view.sort_values(by=sort_by, ascending=False)

base_df = view.reset_index().rename(columns={"resource_code": "SKU"})
if _name_map:
    base_df.insert(1, "품명", base_df["SKU"].map(_name_map).fillna(""))

if show_cost:
    snap_raw_df = load_snapshot_raw()
    cost_pivot = pivot_inventory_cost_from_raw(snap_raw_df, _latest_dt, centers_sel)
    if cost_pivot.empty:
        st.warning(
            "재고자산 계산을 위한 'snapshot_raw' 데이터를 불러올 수 없어 수량만 표시합니다. (엑셀에 'snapshot_raw' 시트가 있으면 자동 사용됩니다)"
        )
        show_df = base_df
    else:
        merged = base_df.merge(cost_pivot.rename(columns={"resource_code": "SKU"}), on="SKU", how="left")
        cost_cols2 = [c for c in merged.columns if c.endswith("_재고자산")] + (
            ["총 재고자산"] if "총 재고자산" in merged.columns else []
        )
        if cost_cols2:
            merged[cost_cols2] = merged[cost_cols2].fillna(0).astype(int)
            for col in cost_cols2:
                merged[col] = merged[col].apply(lambda x: f"{x:,}원")
        qty_center_cols = [c for c in merged.columns if c not in ["SKU", "품명", "총합"] + cost_cols2]
        ordered = ["SKU"] + (["품명"] if "품명" in merged.columns else []) + qty_center_cols + (
            ["총합"] if "총합" in merged.columns else []
        ) + cost_cols2
        show_df = merged[ordered]
else:
    show_df = base_df

qty_cols = [c for c in show_df.columns if c not in ["SKU"] and not c.endswith("_재고자산") and c != "총 재고자산"]
for col in qty_cols:
    if col in show_df.columns:
        show_df[col] = show_df[col].apply(lambda x: f"{x:,}" if pd.notna(x) and isinstance(x, (int, float)) else x)

st.dataframe(show_df, use_container_width=True, height=380)

csv_bytes = show_df.to_csv(index=False).encode("utf-8-sig")
st.download_button(
    "현재 표 CSV 다운로드",
    data=csv_bytes,
    file_name=f"centers_{'-'.join(centers_sel)}_snapshot_{_latest_dt_str}.csv",
    mime="text/csv",
)

st.caption("※ 이 표는 **선택된 센터 전체 SKU**의 최신 스냅샷 재고입니다. 상단 'SKU 선택'과 무관하게 모든 SKU가 포함됩니다.")

filtered_df = show_df if "SKU" in show_df.columns else view.reset_index().rename(columns={"resource_code": "SKU"})
visible_skus = filtered_df["SKU"].dropna().astype(str).unique().tolist()

if len(visible_skus) == 1:
    lot_sku = visible_skus[0]
    snap_raw_df = load_snapshot_raw()
    if snap_raw_df is None or snap_raw_df.empty:
        latest_dt_str = pd.to_datetime(snap_long["date"]).max().strftime("%Y-%m-%d")
        st.markdown(f"### 로트 상세 (스냅샷 {latest_dt_str} / **{', '.join(centers_sel)}** / **{lot_sku}**)")
        st.caption("해당 조건의 로트 상세가 없습니다. (snapshot_raw 없음)")
    else:
        sr = snap_raw_df.copy()
        cols_map = {c.strip().lower(): c for c in sr.columns}
        col_date = cols_map.get("snapshot_date") or cols_map.get("date")
        col_sku = cols_map.get("resource_code") or cols_map.get("sku") or cols_map.get("상품코드")
        col_lot = cols_map.get("lot")
        used_centers = [ct for ct in centers_sel if CENTER_COL.get(ct) in sr.columns]
        if not all([col_date, col_sku, col_lot]) or not used_centers:
            st.caption("해당 조건의 로트 상세가 없습니다.")
        else:
            sr[col_date] = pd.to_datetime(sr[col_date], errors="coerce")
            sub = sr[(sr[col_date].dt.normalize() == _latest_dt.normalize()) & (sr[col_sku].astype(str) == str(lot_sku))].copy()
            if sub.empty:
                st.caption("해당 조건의 로트 상세가 없습니다.")
            else:
                for ct in used_centers:
                    c = CENTER_COL[ct]
                    sub[c] = pd.to_numeric(sub[c], errors="coerce").fillna(0).clip(lower=0)
                out = pd.DataFrame({"lot": sub[col_lot].astype(str).fillna("(no lot)")})
                for ct in used_centers:
                    out[ct] = sub.groupby(col_lot)[CENTER_COL[ct]].transform("sum")
                out = out.drop_duplicates()
                out["합계"] = out[used_centers].sum(axis=1)
                out = out[out["합계"] > 0]
                latest_dt_str = pd.to_datetime(snap_long["date"]).max().strftime("%Y-%m-%d")
                st.markdown(f"### 로트 상세 (스냅샷 {latest_dt_str} / **{', '.join(centers_sel)}** / **{lot_sku}**)")
                if out.empty:
                    st.caption("해당 조건의 로트 상세가 없습니다.")
                else:
                    st.dataframe(
                        out[["lot"] + used_centers + ["합계"]].sort_values("합계", ascending=False).reset_index(drop=True),
                        use_container_width=True,
                        height=320,
                    )
