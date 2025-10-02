"""Streamlit entry point for the SCM dashboard v5 pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import pandas as pd
import plotly.express as px
import streamlit as st

from scm_dashboard_v4.config import PALETTE
from scm_dashboard_v4.loaders import load_from_excel, load_from_gsheet_api
from scm_dashboard_v4.processing import (
    load_wip_from_incoming,
    merge_wip_as_moves,
    normalize_moves,
    normalize_refined_snapshot,
)
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
        lag_days = st.slider("WIP 지연 일수", min_value=0, max_value=30, value=7, step=1)

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

    _plot_timeline(timeline, start=start_ts, end=end_ts)

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


if __name__ == "__main__":
    main()
