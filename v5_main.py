"""Streamlit entry point for the SCM dashboard v5 pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import pandas as pd
import plotly.express as px
import streamlit as st

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


def _plot_timeline(timeline: pd.DataFrame) -> None:
    """Render the timeline using Plotly."""

    fig = px.line(
        timeline,
        x="date",
        y="stock_qty",
        color="center",
        line_dash="resource_code",
        markers=True,
        title="센터/라인별 재고 추이",
    )
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)


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

    _plot_timeline(timeline)

    st.subheader("센터별 타임라인")
    st.dataframe(timeline[timeline["center"].isin(selected_centers)])

    st.subheader("In-Transit / WIP")
    st.dataframe(timeline[~timeline["center"].isin(selected_centers)])


if __name__ == "__main__":
    main()
