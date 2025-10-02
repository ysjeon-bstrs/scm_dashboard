"""Streamlit entry point for the SCM dashboard v5 pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from scm_dashboard_v4.config import CENTER_COL, PALETTE
from scm_dashboard_v4.inventory import pivot_inventory_cost_from_raw
from scm_dashboard_v4.loaders import load_from_excel, load_from_gsheet_api, load_snapshot_raw
from scm_dashboard_v4.processing import (
    load_wip_from_incoming,
    merge_wip_as_moves,
    normalize_moves,
    normalize_refined_snapshot,
)
from scm_dashboard_v5.analytics import prepare_amazon_daily_sales
from scm_dashboard_v5.forecast import apply_consumption_with_events
from scm_dashboard_v5.pipeline import BuildInputs, build_timeline_bundle


@dataclass
class LoadedData:
    moves: pd.DataFrame
    snapshot: pd.DataFrame


def _load_from_excel_uploader() -> Optional[LoadedData]:
    """Return normalized data loaded from an uploaded Excel file."""

    file = st.file_uploader("엑셀 업로드 (.xlsx)", type=["xlsx"], key="v5_excel")
    if file is None:
        return None

    df_move, df_ref, df_incoming, _ = load_from_excel(file)
    moves = normalize_moves(df_move)
    snapshot = normalize_refined_snapshot(df_ref)

    try:
        wip_df = load_wip_from_incoming(df_incoming)
        moves = merge_wip_as_moves(moves, wip_df)
        if wip_df is not None and not wip_df.empty:
            st.success(f"WIP {len(wip_df)}건 반영 완료")
    except Exception as exc:  # pragma: no cover - streamlit feedback
        st.warning(f"WIP 불러오기 실패: {exc}")

    return LoadedData(moves=moves, snapshot=snapshot)


def _load_from_gsheet_button() -> Optional[LoadedData]:
    """Return normalized data retrieved from Google Sheets."""

    if not st.button("Google Sheets에서 데이터 로드", type="primary", key="v5_gsheet"):
        return None

    df_move, df_ref, df_incoming = load_from_gsheet_api()
    if df_move.empty or df_ref.empty:
        st.error("Google Sheets에서 데이터를 불러올 수 없습니다. 권한을 확인해주세요.")
        return None

    moves = normalize_moves(df_move)
    snapshot = normalize_refined_snapshot(df_ref)

    try:
        wip_df = load_wip_from_incoming(df_incoming)
        moves = merge_wip_as_moves(moves, wip_df)
        if wip_df is not None and not wip_df.empty:
            st.success(f"WIP {len(wip_df)}건 반영 완료")
    except Exception as exc:  # pragma: no cover - streamlit feedback
        st.warning(f"WIP 불러오기 실패: {exc}")

    return LoadedData(moves=moves, snapshot=snapshot)


def _ensure_data() -> Optional[LoadedData]:
    """Load data via the available tabs and persist it in the session state."""

    if "v5_data" in st.session_state:
        return st.session_state["v5_data"]

    tab_excel, tab_gsheet = st.tabs(["엑셀 업로드", "Google Sheets"])

    with tab_excel:
        data = _load_from_excel_uploader()
        if data is not None:
            st.session_state["_v5_source"] = "excel"
            st.session_state["v5_data"] = data
            return data

    with tab_gsheet:
        data = _load_from_gsheet_button()
        if data is not None:
            st.session_state["_v5_source"] = "gsheet"
            st.session_state["v5_data"] = data
            return data

    return None


def _center_and_sku_options(moves: pd.DataFrame, snapshot: pd.DataFrame) -> Tuple[list[str], list[str]]:
    """Derive selectable centers and SKUs from moves and snapshot frames."""

    move_centers = set(moves["from_center"].dropna().astype(str)) | set(
        moves["to_center"].dropna().astype(str)
    )
    snap_centers = set(snapshot["center"].dropna().astype(str))
    centers = sorted({c for c in move_centers | snap_centers if c and c.lower() != "nan"})

    skus = sorted(snapshot["resource_code"].dropna().astype(str).unique().tolist())
    if not skus:
        skus = sorted(moves["resource_code"].dropna().astype(str).unique().tolist())

    return centers, skus


def _date_bounds(moves: pd.DataFrame, snapshot: pd.DataFrame) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Compute a sensible default date window based on available data."""

    dates = [
        snapshot["date"].min() if not snapshot.empty else None,
        snapshot["date"].max() if not snapshot.empty else None,
        moves.get("onboard_date").min() if "onboard_date" in moves.columns else None,
        moves.get("pred_inbound_date").min() if "pred_inbound_date" in moves.columns else None,
        moves.get("pred_inbound_date").max() if "pred_inbound_date" in moves.columns else None,
        moves.get("event_date").max() if "event_date" in moves.columns else None,
    ]
    dates = [pd.to_datetime(d).normalize() for d in dates if pd.notna(d)]
    if not dates:
        today = pd.Timestamp.today().normalize()
        return today - pd.Timedelta(days=30), today + pd.Timedelta(days=30)

    return min(dates), max(dates)


def _build_timeline(
    *,
    data: LoadedData,
    centers: list[str],
    skus: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    lag_days: int,
) -> Optional[pd.DataFrame]:
    """Run the v5 pipeline and return the concatenated timeline."""

    today = pd.Timestamp.today().normalize()
    bundle = build_timeline_bundle(
        BuildInputs(snapshot=data.snapshot, moves=data.moves),
        centers=centers,
        skus=skus,
        start=start,
        end=end,
        today=today,
        lag_days=lag_days,
    )
    timeline = bundle.concat()
    if timeline.empty:
        return None
    return timeline


def _plot_timeline(
    timeline: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    show_production: bool,
    selected_centers: Iterable[str],
) -> None:
    """Render the timeline using Plotly with styling borrowed from v4."""

    if timeline.empty:
        st.info("선택한 조건에 해당하는 타임라인 데이터가 없습니다.")
        return

    vis_df = timeline.copy()
    vis_df["date"] = pd.to_datetime(vis_df["date"]).dt.normalize()
    vis_df = vis_df[(vis_df["date"] >= start) & (vis_df["date"] <= end)]

    center_translation = {"In-Transit": "이동중", "WIP": "생산중"}
    vis_df["center"] = vis_df["center"].replace(center_translation)

    centers_set = {str(c) for c in selected_centers}
    if ("태광KR" not in centers_set) or not show_production:
        vis_df = vis_df[vis_df["center"] != "생산중"]

    vis_df = vis_df[vis_df["stock_qty"] != 0]
    if vis_df.empty:
        st.info("선택한 조건에 해당하는 타임라인 데이터가 없습니다.")
        return

    vis_df["label"] = vis_df["resource_code"].astype(str) + " @ " + vis_df["center"].astype(str)

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
    fig.update_yaxes(tickformat=",.0f")
    fig.update_traces(
        hovertemplate="날짜: %{x|%Y-%m-%d}<br>재고: %{y:,.0f} EA<br>%{fullData.name}<extra></extra>"
    )

    line_colors: dict[str, str] = {}
    color_idx = 0
    for trace in fig.data:
        name = trace.name or ""
        if " @ " in name and name not in line_colors:
            line_colors[name] = PALETTE[color_idx % len(PALETTE)]
            color_idx += 1

    for idx, trace in enumerate(fig.data):
        name = trace.name or ""
        if " @ " not in name:
            continue
        _, center_name = name.split(" @ ", 1)
        line_color = line_colors.get(name, PALETTE[0])
        is_transit = center_name in {"이동중", "생산중"}
        fig.data[idx].update(
            line=dict(
                color=line_color,
                dash="dash" if is_transit else "solid",
                width=1.0 if is_transit else 1.5,
            ),
            opacity=0.8 if is_transit else 1.0,
        )

    today = pd.Timestamp.today().normalize()
    if start <= today <= end:
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

    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})


def main() -> None:
    """Entrypoint for running the v5 dashboard in Streamlit."""

    st.set_page_config(page_title="SCM Dashboard v5", layout="wide")
    st.title("SCM Dashboard v5")
    st.caption("모듈화된 v5 파이프라인을 이용한 Streamlit 엔트리 포인트")

    data = _ensure_data()
    if data is None:
        st.info("데이터를 로드하면 차트와 테이블이 표시됩니다.")
        return

    centers, skus = _center_and_sku_options(data.moves, data.snapshot)
    if not centers or not skus:
        st.warning("센터 또는 SKU 정보를 찾을 수 없습니다.")
        return

    min_dt, max_dt = _date_bounds(data.moves, data.snapshot)
    today = pd.Timestamp.today().normalize()
    default_start = max(min_dt, today - pd.Timedelta(days=30))
    default_end = min(max_dt, today + pd.Timedelta(days=60))

    with st.sidebar:
        st.header("필터")
        selected_centers = st.multiselect("센터", centers, default=centers)
        default_skus = skus if len(skus) <= 10 else skus[:10]
        selected_skus = st.multiselect("SKU", skus, default=default_skus)
        date_range = st.date_input(
            "타임라인 범위",
            value=(default_start.to_pydatetime(), default_end.to_pydatetime()),
            min_value=min_dt.to_pydatetime(),
            max_value=max_dt.to_pydatetime(),
        )
        st.header("표시 옵션")
        show_prod = st.checkbox("생산중(미완료) 표시", value=True)
        use_cons_forecast = st.checkbox("추세 기반 재고 예측", value=True)
        lookback_days = int(
            st.number_input(
                "추세 계산 기간(일)",
                min_value=7,
                max_value=56,
                value=28,
                step=7,
            )
        )

        st.subheader("입고 반영 가정")
        lag_days = int(
            st.number_input(
                "입고 반영 리드타임(일) – inbound 미기록 시 arrival+N",
                min_value=0,
                max_value=21,
                value=7,
                step=1,
            )
        )

        with st.expander("프로모션 가중치(+%)", expanded=False):
            enable_event = st.checkbox("가중치 적용", value=False)
            ev_start = st.date_input("시작일")
            ev_end = st.date_input("종료일")
            ev_pct = st.number_input(
                "가중치(%)",
                min_value=-100.0,
                max_value=300.0,
                value=30.0,
                step=5.0,
            )

        if enable_event:
            events = [
                {
                    "start": pd.Timestamp(ev_start).strftime("%Y-%m-%d"),
                    "end": pd.Timestamp(ev_end).strftime("%Y-%m-%d"),
                    "uplift": float(ev_pct) / 100.0,
                }
            ]
        else:
            events = []

    if not selected_centers:
        st.warning("최소 한 개의 센터를 선택하세요.")
        return
    if not selected_skus:
        st.warning("최소 한 개의 SKU를 선택하세요.")
        return

    if isinstance(date_range, tuple):
        start_date, end_date = date_range
    else:
        start_date = date_range
        end_date = date_range

    start_ts = pd.Timestamp(start_date).normalize()
    end_ts = pd.Timestamp(end_date).normalize()
    if end_ts < start_ts:
        st.warning("종료일이 시작일보다 빠릅니다.")
        return

    timeline = _build_timeline(
        data=data,
        centers=selected_centers,
        skus=selected_skus,
        start=start_ts,
        end=end_ts,
        lag_days=lag_days,
    )

    if timeline is None:
        st.info("선택한 조건에 해당하는 타임라인 데이터가 없습니다.")
        return

    if use_cons_forecast:
        timeline = apply_consumption_with_events(
            timeline,
            data.snapshot,
            centers=selected_centers,
            skus=selected_skus,
            start=start_ts,
            end=end_ts,
            lookback_days=lookback_days,
            events=events,
        )

    _plot_timeline(
        timeline,
        start=start_ts,
        end=end_ts,
        show_production=show_prod,
        selected_centers=selected_centers,
    )

    # -------------------- Amazon US sales vs. inventory --------------------
    amazon_candidates = [
        center
        for center in selected_centers
        if str(center).strip()
        and (
            "amazon" in str(center).lower()
            or str(center).upper().startswith("AMZ")
        )
    ]

    sales_result = prepare_amazon_daily_sales(
        data.snapshot,
        centers=amazon_candidates,
        skus=selected_skus,
        start_dt=start_ts,
        end_dt=end_ts,
        rolling_window=7,
    )

    st.divider()
    st.subheader("Amazon US 일별 판매 vs. 재고")
    sales_df = sales_result.data
    if sales_df.empty:
        st.caption("선택된 SKU/기간에 대한 Amazon US 판매 데이터가 없습니다.")
    else:
        sales_df = sales_df.copy()
        sales_df["date"] = pd.to_datetime(sales_df["date"], errors="coerce")
        sales_df = sales_df.sort_values("date")

        sales_fig = make_subplots(specs=[[{"secondary_y": True}]])
        sales_fig.add_trace(
            go.Bar(
                x=sales_df["date"],
                y=sales_df["sales_qty"],
                name="Daily Sales (EA)",
                marker_color=PALETTE[0],
                opacity=0.85,
            ),
            secondary_y=False,
        )
        sales_fig.add_trace(
            go.Scatter(
                x=sales_df["date"],
                y=sales_df["inventory_qty"],
                mode="lines+markers",
                name="Amazon Inventory (EA)",
                line=dict(color=PALETTE[1], width=2),
                marker=dict(size=4),
            ),
            secondary_y=True,
        )
        sales_fig.add_trace(
            go.Scatter(
                x=sales_df["date"],
                y=sales_df["sales_roll_mean"],
                name="Sales 7d Rolling Avg",
                mode="lines",
                line=dict(color=PALETTE[2], dash="dash"),
                visible="legendonly",
            ),
            secondary_y=False,
        )

        sales_fig.update_layout(
            hovermode="x unified",
            legend_title_text="Amazon 판매/재고",
            margin=dict(l=20, r=40, t=40, b=20),
        )
        sales_fig.update_xaxes(title_text="날짜")
        sales_fig.update_yaxes(
            title_text="일일 판매량(EA)",
            secondary_y=False,
            tickformat=",.0f",
        )
        sales_fig.update_yaxes(
            title_text="Amazon 재고(EA)",
            secondary_y=True,
            tickformat=",.0f",
        )
        st.plotly_chart(sales_fig, use_container_width=True, config={"displaylogo": False})

    window_start = start_ts
    window_end = end_ts
    today = pd.Timestamp.today().normalize()

    moves_view = data.moves.copy()
    for col in [
        "carrier_mode",
        "to_center",
        "resource_code",
        "inbound_date",
        "arrival_date",
        "onboard_date",
        "event_date",
        "lot",
    ]:
        if col not in moves_view.columns:
            if "date" in col:
                moves_view[col] = pd.Series(pd.NaT, index=moves_view.index, dtype="datetime64[ns]")
            else:
                moves_view[col] = pd.Series("", index=moves_view.index, dtype="object")

    if not moves_view.empty:
        pred_inbound = pd.Series(pd.NaT, index=moves_view.index, dtype="datetime64[ns]")

        if "inbound_date" in moves_view.columns:
            mask_inbound = moves_view["inbound_date"].notna()
            pred_inbound.loc[mask_inbound] = moves_view.loc[mask_inbound, "inbound_date"]
        else:
            mask_inbound = pd.Series(False, index=moves_view.index)

        arrival_series = moves_view.get("arrival_date")
        if arrival_series is not None:
            mask_arrival = (~mask_inbound) & arrival_series.notna()
        else:
            mask_arrival = pd.Series(False, index=moves_view.index)

        if mask_arrival.any():
            past_arr = mask_arrival & (arrival_series <= today)
            if past_arr.any():
                pred_inbound.loc[past_arr] = moves_view.loc[past_arr, "arrival_date"] + pd.Timedelta(
                    days=int(lag_days)
                )
            fut_arr = mask_arrival & (arrival_series > today)
            if fut_arr.any():
                pred_inbound.loc[fut_arr] = moves_view.loc[fut_arr, "arrival_date"]

        moves_view["pred_inbound_date"] = pred_inbound
    else:
        moves_view["pred_inbound_date"] = pd.Series(
            pd.NaT, index=moves_view.index, dtype="datetime64[ns]"
        )

    arr_transport = moves_view[
        (moves_view["carrier_mode"] != "WIP")
        & (moves_view["to_center"].isin(selected_centers))
        & (moves_view["resource_code"].isin(selected_skus))
        & (moves_view["inbound_date"].isna())
    ].copy()

    arr_transport["display_date"] = arr_transport["arrival_date"].fillna(
        arr_transport["onboard_date"]
    )
    arr_transport = arr_transport[arr_transport["display_date"].notna()]
    arr_transport = arr_transport[
        (arr_transport["display_date"] >= window_start)
        & (arr_transport["display_date"] <= window_end)
    ]

    arr_wip = pd.DataFrame()
    if "태광KR" in selected_centers:
        arr_wip = moves_view[
            (moves_view["carrier_mode"] == "WIP")
            & (moves_view["to_center"] == "태광KR")
            & (moves_view["resource_code"].isin(selected_skus))
            & (moves_view["event_date"].notna())
            & (moves_view["event_date"] >= window_start)
            & (moves_view["event_date"] <= window_end)
        ].copy()
        arr_wip["display_date"] = arr_wip["event_date"]

    resource_name_map: dict[str, str] = {}
    if "resource_name" in data.snapshot.columns:
        name_rows = data.snapshot.loc[data.snapshot["resource_name"].notna(), [
            "resource_code",
            "resource_name",
        ]].copy()
        name_rows["resource_code"] = name_rows["resource_code"].astype(str)
        name_rows["resource_name"] = name_rows["resource_name"].astype(str).str.strip()
        name_rows = name_rows[name_rows["resource_name"] != ""]
        if not name_rows.empty:
            resource_name_map = (
                name_rows.drop_duplicates("resource_code").set_index("resource_code")["resource_name"].to_dict()
            )

    confirmed_inbound = arr_transport.copy()
    if resource_name_map and not confirmed_inbound.empty:
        confirmed_inbound["resource_name"] = confirmed_inbound["resource_code"].map(resource_name_map).fillna("")

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
        if resource_name_map:
            arr_wip["resource_name"] = arr_wip["resource_code"].map(resource_name_map).fillna("")
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

    snapshot_df = data.snapshot.copy()
    if snapshot_df.empty or "date" not in snapshot_df.columns:
        st.info("스냅샷 데이터가 없습니다.")
        return

    snapshot_df["date"] = pd.to_datetime(snapshot_df["date"], errors="coerce").dt.normalize()
    latest_dt = snapshot_df["date"].max()
    if pd.isna(latest_dt):
        st.info("스냅샷 데이터의 날짜 정보를 확인할 수 없습니다.")
        return

    latest_dt_str = latest_dt.strftime("%Y-%m-%d")
    st.subheader(f"선택 센터 현재 재고 (스냅샷 {latest_dt_str} / 전체 SKU)")

    current_snapshot = snapshot_df[
        (snapshot_df["date"] == latest_dt) & (snapshot_df["center"].isin(selected_centers))
    ].copy()

    pivot = (
        current_snapshot.groupby(["resource_code", "center"], as_index=False)["stock_qty"].sum()
        .pivot(index="resource_code", columns="center", values="stock_qty")
        .fillna(0)
    )

    for center in selected_centers:
        if center not in pivot.columns:
            pivot[center] = 0
    if pivot.empty:
        pivot = pivot.reindex(columns=selected_centers)
    pivot = pivot.reindex(columns=selected_centers, fill_value=0)
    pivot = pivot.astype(int)
    pivot["총합"] = pivot.sum(axis=1)

    col_filter, col_sort = st.columns([2, 1])
    with col_filter:
        sku_query = st.text_input(
            "SKU 필터 — 품목번호 검색 시 해당 SKU의 센터별 제조번호(LOT) 확인",
            "",
            key="v5_sku_filter_text",
        )
    with col_sort:
        sort_candidates = ["총합"] + selected_centers
        sort_by = st.selectbox("정렬 기준", sort_candidates, index=0)

    col_zero, col_cost = st.columns(2)
    with col_zero:
        hide_zero = st.checkbox("총합=0 숨기기", value=True)
    with col_cost:
        show_cost = st.checkbox("재고자산(제조원가) 표시", value=False)

    view = pivot.copy()
    if sku_query.strip():
        view = view[view.index.astype(str).str.contains(sku_query.strip(), case=False, regex=False)]
    if hide_zero and "총합" in view.columns:
        view = view[view["총합"] > 0]
    if sort_by in view.columns:
        view = view.sort_values(by=sort_by, ascending=False)

    display_df = view.reset_index().rename(columns={"resource_code": "SKU"})
    if resource_name_map:
        display_df.insert(1, "품명", display_df["SKU"].map(resource_name_map).fillna(""))

    if show_cost:
        snap_raw_df = load_snapshot_raw()
        cost_pivot = pivot_inventory_cost_from_raw(snap_raw_df, latest_dt, selected_centers)
        if cost_pivot.empty:
            st.warning(
                "재고자산 계산을 위한 'snapshot_raw' 데이터를 불러올 수 없어 수량만 표시합니다. (엑셀에 'snapshot_raw' 시트가 있으면 자동 사용됩니다)"
            )
            merged_df = display_df
            cost_columns = []
        else:
            merged_df = display_df.merge(
                cost_pivot.rename(columns={"resource_code": "SKU"}),
                on="SKU",
                how="left",
            )
            cost_columns = [c for c in merged_df.columns if c.endswith("_재고자산")]
            if "총 재고자산" in merged_df.columns:
                cost_columns.append("총 재고자산")
            if cost_columns:
                merged_df[cost_columns] = merged_df[cost_columns].fillna(0).astype(int)
                for col in cost_columns:
                    merged_df[col] = merged_df[col].apply(
                        lambda x: f"{x:,}원" if isinstance(x, (int, float)) else x
                    )
        quantity_columns = [
            c
            for c in merged_df.columns
            if c not in {"SKU", "품명", "총합", *cost_columns}
        ]
        ordered_columns = ["SKU"]
        if "품명" in merged_df.columns:
            ordered_columns.append("품명")
        ordered_columns.extend([c for c in quantity_columns if not c.endswith("_재고자산")])
        if "총합" in merged_df.columns:
            ordered_columns.append("총합")
        ordered_columns.extend(cost_columns)
        show_df = merged_df[ordered_columns]
    else:
        show_df = display_df
        cost_columns = []

    qty_columns = [
        c
        for c in show_df.columns
        if c not in {"SKU", "품명"}
        and not c.endswith("_재고자산")
        and c != "총 재고자산"
    ]
    for column in qty_columns:
        show_df[column] = show_df[column].apply(
            lambda x: f"{int(x):,}" if pd.notna(x) and isinstance(x, (int, float)) else x
        )

    st.dataframe(show_df, use_container_width=True, height=380)

    csv_bytes = show_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "현재 표 CSV 다운로드",
        data=csv_bytes,
        file_name=f"centers_{'-'.join(selected_centers)}_snapshot_{latest_dt_str}.csv",
        mime="text/csv",
    )

    st.caption(
        "※ 이 표는 **선택된 센터 전체 SKU**의 최신 스냅샷 재고입니다. 상단 'SKU 선택'과 무관하게 모든 SKU가 포함됩니다."
    )

    filtered_df = (
        show_df if "SKU" in show_df.columns else view.reset_index().rename(columns={"resource_code": "SKU"})
    )
    visible_skus = filtered_df.get("SKU", pd.Series(dtype=str)).dropna().astype(str).unique().tolist()

    if len(visible_skus) == 1:
        lot_sku = visible_skus[0]
        snap_raw_df = load_snapshot_raw()
        if snap_raw_df is None or snap_raw_df.empty:
            st.markdown(
                f"### 로트 상세 (스냅샷 {latest_dt_str} / **{', '.join(selected_centers)}** / **{lot_sku}**)"
            )
            st.caption("해당 조건의 로트 상세가 없습니다. (snapshot_raw 없음)")
        else:
            raw_df = snap_raw_df.copy()
            cols_map = {str(col).strip().lower(): col for col in raw_df.columns}
            col_date = cols_map.get("snapshot_date") or cols_map.get("date")
            col_sku = (
                cols_map.get("resource_code")
                or cols_map.get("sku")
                or cols_map.get("상품코드")
            )
            col_lot = cols_map.get("lot")
            used_centers = [ct for ct in selected_centers if CENTER_COL.get(ct) in raw_df.columns]
            if not all([col_date, col_sku, col_lot]) or not used_centers:
                st.markdown(
                    f"### 로트 상세 (스냅샷 {latest_dt_str} / **{', '.join(selected_centers)}** / **{lot_sku}**)"
                )
                st.caption("해당 조건의 로트 상세가 없습니다.")
            else:
                raw_df[col_date] = pd.to_datetime(raw_df[col_date], errors="coerce").dt.normalize()
                lot_subset = raw_df[
                    (raw_df[col_date] == latest_dt) & (raw_df[col_sku].astype(str) == str(lot_sku))
                ].copy()
                if lot_subset.empty:
                    st.markdown(
                        f"### 로트 상세 (스냅샷 {latest_dt_str} / **{', '.join(selected_centers)}** / **{lot_sku}**)"
                    )
                    st.caption("해당 조건의 로트 상세가 없습니다.")
                else:
                    for center in used_centers:
                        src_col = CENTER_COL.get(center)
                        lot_subset[src_col] = (
                            pd.to_numeric(lot_subset[src_col], errors="coerce").fillna(0).clip(lower=0)
                        )
                    lot_table = pd.DataFrame({"lot": lot_subset[col_lot].astype(str).fillna("(no lot)")})
                    for center in used_centers:
                        src_col = CENTER_COL.get(center)
                        lot_table[center] = lot_subset.groupby(col_lot)[src_col].transform("sum")
                    lot_table = lot_table.drop_duplicates()
                    lot_table["합계"] = lot_table[used_centers].sum(axis=1)
                    lot_table = lot_table[lot_table["합계"] > 0]
                    st.markdown(
                        f"### 로트 상세 (스냅샷 {latest_dt_str} / **{', '.join(selected_centers)}** / **{lot_sku}**)"
                    )
                    if lot_table.empty:
                        st.caption("해당 조건의 로트 상세가 없습니다.")
                    else:
                        st.dataframe(
                            lot_table[["lot"] + used_centers + ["합계"]]
                            .sort_values("합계", ascending=False)
                            .reset_index(drop=True),
                            use_container_width=True,
                            height=320,
                        )


if __name__ == "__main__":
    main()
