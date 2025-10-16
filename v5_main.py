"""Streamlit entry point for the SCM dashboard v5 pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import pandas as pd
import streamlit as st

from center_alias import normalize_center_value
from scm_dashboard_v4.config import CENTER_COL
from scm_dashboard_v4.inventory import pivot_inventory_cost_from_raw
from scm_dashboard_v4.loaders import load_from_excel, load_from_gsheet_api, load_snapshot_raw
from scm_dashboard_v4.processing import (
    load_wip_from_incoming,
    merge_wip_as_moves,
    normalize_moves,
    normalize_refined_snapshot,
)

from scm_dashboard_v5.core import build_timeline as build_core_timeline
from scm_dashboard_v5.forecast import (
    apply_consumption_with_events,
    build_amazon_forecast_context,
    estimate_daily_consumption,
    forecast_sales_and_inventory,
    load_amazon_daily_sales_from_snapshot_raw,
)
from scm_dashboard_v5.ui import (
    render_amazon_sales_vs_inventory,
    render_step_chart,
    render_sku_summary_cards,
)
from scm_dashboard_v5.ui.charts import _sku_color_map, _timeline_inventory_matrix


def _validate_timeline_inputs(
    snapshot: object,
    moves: object,
    start: object,
    end: object,
) -> bool:
    """Return True if the timeline inputs look structurally correct."""

    if not isinstance(snapshot, pd.DataFrame):
        st.error("스냅샷 데이터가 손상되었습니다. 엑셀/시트를 다시 불러와 주세요.")
        return False
    if not isinstance(moves, pd.DataFrame):
        st.error("이동 원장 데이터가 손상되었습니다. 엑셀/시트를 다시 불러와 주세요.")
        return False

    required_snapshot_cols = {"center", "resource_code", "stock_qty"}
    missing_snapshot = [col for col in required_snapshot_cols if col not in snapshot.columns]
    if missing_snapshot:
        st.error(
            "스냅샷 데이터에 필요한 컬럼이 없습니다: "
            + ", ".join(sorted(missing_snapshot))
        )
        return False

    required_move_cols = {"from_center", "to_center", "resource_code"}
    missing_moves = [col for col in required_move_cols if col not in moves.columns]
    if missing_moves:
        st.error(
            "이동 원장 데이터에 필요한 컬럼이 없습니다: " + ", ".join(sorted(missing_moves))
        )
        return False

    if not isinstance(start, pd.Timestamp) or not isinstance(end, pd.Timestamp):
        st.error("기간 정보가 손상되었습니다. 기간 슬라이더를 다시 설정해 주세요.")
        return False

    if end < start:
        st.error("기간의 종료일이 시작일보다 빠릅니다. 기간을 다시 선택하세요.")
        return False

    return True


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


def _load_from_gsheet(*, show_spinner_message: str) -> Optional[LoadedData]:
    """Return normalized data retrieved from Google Sheets."""

    try:
        with st.spinner(show_spinner_message):
            df_move, df_ref, df_incoming = load_from_gsheet_api()
    except Exception as exc:  # pragma: no cover - streamlit feedback
        st.error(f"Google Sheets 데이터를 불러오는 중 오류가 발생했습니다: {exc}")
        return None

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

    st.success("Google Sheets 데이터가 업데이트되었습니다.")
    return LoadedData(moves=moves, snapshot=snapshot)


def _ensure_data() -> Optional[LoadedData]:
    """Load data via the available tabs and persist it in the session state."""

    data: Optional[LoadedData] = st.session_state.get("v5_data")

    st.markdown("### 데이터 소스")
    st.caption("대시보드 진입 시 Google Sheets 데이터를 자동으로 불러옵니다.")

    source_label = st.session_state.get("_v5_source")
    source_display = {
        "gsheet": "Google Sheets",
        "excel": "엑셀 업로드",
    }.get(source_label, "없음")

    source_caption = st.empty()

    refresh_clicked = st.button("Google Sheets 데이터 새로 고침", key="v5_gsheet_refresh")

    should_load_gsheet = data is None or refresh_clicked
    if should_load_gsheet:
        spinner_msg = (
            "Google Sheets 데이터 불러오는 중..."
            if data is None
            else "Google Sheets 데이터를 새로 불러오는 중..."
        )
        gsheet_data = _load_from_gsheet(show_spinner_message=spinner_msg)
        if gsheet_data is not None:
            st.session_state["_v5_source"] = "gsheet"
            st.session_state["v5_data"] = gsheet_data
            data = gsheet_data
            source_display = "Google Sheets"
        elif data is None:
            source_display = "없음"

    with st.expander("엑셀 파일 업로드 (선택)", expanded=False):
        st.caption("필요할 때만 수동으로 엑셀 파일을 업로드하여 데이터를 교체할 수 있습니다.")
        excel_data = _load_from_excel_uploader()
        if excel_data is not None:
            st.session_state["_v5_source"] = "excel"
            st.session_state["v5_data"] = excel_data
            st.success("엑셀 데이터가 로드되었습니다.")
            data = excel_data
            source_display = "엑셀 업로드"

    source_caption.caption(f"현재 데이터 소스: **{source_display}**")

    if data is None:
        return None

    return data


def _norm_center(x: str) -> str | None:
    return normalize_center_value(x)


def _center_and_sku_options(moves: pd.DataFrame, snapshot: pd.DataFrame) -> Tuple[list[str], list[str]]:
    """Derive selectable centers and SKUs from moves and snapshot frames."""

    snap_centers_src = snapshot.get("center")
    if snap_centers_src is None:
        snap_centers = pd.Series(dtype=str)
    else:
        snap_centers = snap_centers_src.dropna().astype(str).str.strip()
    move_centers = pd.concat(
        [
            moves.get("from_center", pd.Series(dtype=object)),
            moves.get("to_center", pd.Series(dtype=object)),
        ],
        ignore_index=True,
    ).dropna().astype(str).str.strip()

    all_candidates = pd.concat([snap_centers, move_centers], ignore_index=True).dropna()
    centers = sorted(
        {c for c in (_norm_center(value) for value in all_candidates) if c}
    )

    skus = sorted(snapshot["resource_code"].dropna().astype(str).unique().tolist())
    if not skus:
        skus = sorted(moves["resource_code"].dropna().astype(str).unique().tolist())

    return centers, skus


def _move_date_bounds(moves: pd.DataFrame) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """Return the earliest and latest movement dates available in *moves*."""

    date_columns = [
        col
        for col in ("onboard_date", "arrival_date", "inbound_date", "event_date")
        if col in moves.columns
    ]
    if not date_columns:
        return None, None

    normalized_dates = []
    for col in date_columns:
        series = pd.to_datetime(moves[col], errors="coerce").dropna()
        if not series.empty:
            normalized_dates.append(series.dt.normalize())

    if not normalized_dates:
        return None, None

    combined = pd.concat(normalized_dates, ignore_index=True)
    if combined.empty:
        return None, None

    min_dt = combined.min()
    max_dt = combined.max()
    if pd.isna(min_dt) or pd.isna(max_dt):
        return None, None

    return pd.Timestamp(min_dt).normalize(), pd.Timestamp(max_dt).normalize()


def _calculate_date_bounds(
    *,
    today: pd.Timestamp,
    snapshot_df: pd.DataFrame,
    moves_df: pd.DataFrame,
    base_past_days: int,
    base_future_days: int,
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Compute the selectable date range for the dashboard period slider."""

    normalized_today = pd.Timestamp(today).normalize()
    base_min = (normalized_today - pd.Timedelta(days=base_past_days)).normalize()
    base_max = (normalized_today + pd.Timedelta(days=base_future_days)).normalize()

    bound_min_candidates = [base_min]
    bound_max_candidates = [base_max]

    snap_dates = pd.to_datetime(snapshot_df.get("date"), errors="coerce").dropna()
    if not snap_dates.empty:
        bound_min_candidates.append(snap_dates.min().normalize())
        bound_max_candidates.append(snap_dates.max().normalize())

    move_min, move_max = _move_date_bounds(moves_df)
    if move_min is not None:
        bound_min_candidates.append(move_min)
    if move_max is not None:
        bound_max_candidates.append(move_max)

    dynamic_min = min(bound_min_candidates)
    dynamic_max = max(bound_max_candidates)

    bound_min = max(dynamic_min, base_min)
    bound_max = min(dynamic_max, base_max)

    if bound_min > bound_max:
        bound_min = bound_max

    return bound_min.normalize(), bound_max.normalize()


def get_consumption_params_from_ui() -> dict[str, object]:
    """Collect shared consumption parameters from the UI controls."""

    lookback_days = int(st.session_state.get("trend_lookback_days", 28))
    promo_on = bool(st.session_state.get("promo_enabled", False))
    promo_start = st.session_state.get("promo_start")
    promo_end = st.session_state.get("promo_end")
    promo_uplift = float(st.session_state.get("promo_uplift_pct", 0.0)) / 100.0

    events: list[dict[str, object]] = []
    if promo_on and promo_start and promo_end and promo_uplift != 0.0:
        events.append(
            {
                "start": pd.to_datetime(promo_start),
                "end": pd.to_datetime(promo_end),
                "uplift": promo_uplift,
            }
        )

    return {"lookback_days": lookback_days, "events": events}


def main() -> None:
    """Entrypoint for running the v5 dashboard in Streamlit."""

    st.set_page_config(page_title="SCM Dashboard v5", layout="wide")
    st.title("SCM Dashboard v5")
    st.caption("모듈화된 v5 파이프라인을 이용한 Streamlit 엔트리 포인트")

    data = _ensure_data()
    if data is None:
        st.info("데이터를 로드하면 차트와 테이블이 표시됩니다.")
        return

    snapshot_df = data.snapshot.copy()
    if "date" in snapshot_df.columns:
        snapshot_df["date"] = (
            pd.to_datetime(snapshot_df["date"], errors="coerce").dt.normalize()
        )
    elif "snapshot_date" in snapshot_df.columns:
        snapshot_df["date"] = (
            pd.to_datetime(snapshot_df["snapshot_date"], errors="coerce").dt.normalize()
        )
    else:
        snapshot_df["date"] = pd.NaT

    centers, skus = _center_and_sku_options(data.moves, snapshot_df)
    if not centers or not skus:
        st.warning("센터 또는 SKU 정보를 찾을 수 없습니다.")
        return

    today = pd.Timestamp.today().normalize()
    snap_dates = snapshot_df["date"].dropna()
    latest_dt = snap_dates.max() if not snap_dates.empty else pd.NaT
    latest_snapshot_dt = (
        None if pd.isna(latest_dt) else pd.to_datetime(latest_dt).normalize()
    )
    default_past_days = 10
    default_future_days = 30
    base_past_days = 42
    base_future_days = 42

    bound_min, bound_max = _calculate_date_bounds(
        today=today,
        snapshot_df=snapshot_df,
        moves_df=data.moves,
        base_past_days=base_past_days,
        base_future_days=base_future_days,
    )

    def _clamp_range(range_value: Tuple[pd.Timestamp, pd.Timestamp]) -> Tuple[pd.Timestamp, pd.Timestamp]:
        start_val, end_val = range_value
        start_val = pd.Timestamp(start_val).normalize()
        end_val = pd.Timestamp(end_val).normalize()
        start_val = max(min(start_val, bound_max), bound_min)
        end_val = max(min(end_val, bound_max), bound_min)
        if end_val < start_val:
            end_val = start_val
        return (start_val, end_val)

    def _init_range() -> None:
        if "date_range" not in st.session_state:
            default_start = max(today - pd.Timedelta(days=default_past_days), bound_min)
            default_end = min(today + pd.Timedelta(days=default_future_days), bound_max)
            if default_start > default_end:
                default_start = default_end
            st.session_state.date_range = (default_start, default_end)
        else:
            st.session_state.date_range = _clamp_range(tuple(st.session_state.date_range))

    _init_range()

    with st.sidebar:
        st.header("필터")
        st.caption(
            "기본값: 센터 태광KR·AMZUS / SKU BA00021·BA00022 / 기간 오늘−10일 ~ +30일."
            " 해당 항목이 없으면 전체 데이터를 기준으로 표시합니다."
        )
        preset_centers = ["태광KR", "AMZUS"]
        default_centers = [c for c in preset_centers if c in centers]
        if not default_centers:
            default_centers = centers
        selected_centers = st.multiselect("센터", centers, default=default_centers)
        preset_skus = ["BA00021", "BA00022"]
        default_skus = [s for s in preset_skus if s in skus]
        if not default_skus:
            default_skus = skus if len(skus) <= 10 else skus[:10]
        selected_skus = st.multiselect("SKU", skus, default=default_skus)
        st.subheader("기간 설정")
        date_range_value = st.slider(
            "기간",
            min_value=bound_min.to_pydatetime(),
            max_value=bound_max.to_pydatetime(),
            value=tuple(
                d.to_pydatetime() for d in _clamp_range(st.session_state.date_range)
            ),
            format="YYYY-MM-DD",
        )
        start_ts = pd.Timestamp(date_range_value[0]).normalize()
        end_ts = pd.Timestamp(date_range_value[1]).normalize()
        st.session_state.date_range = (start_ts, end_ts)
        st.divider()
        st.header("표시 옵션")
        show_prod = st.checkbox("생산중 표시", value=False)
        # 요청에 따라 이동중 노출 옵션은 기본 해제 상태로 숨긴다.
        show_transit = False
        st.caption("체크 시 계단식 차트에 생산중 라인이 표시됩니다.")
        use_cons_forecast = st.checkbox("추세 기반 재고 예측", value=True)
        st.subheader("추세 계산 설정")
        lookback_days = int(
            st.number_input(
                "추세 계산 기간(일)",
                min_value=7,
                max_value=56,
                value=28,
                step=7,
                key="trend_lookback_days",
            )
        )
        with st.expander("프로모션 가중치(+%)", expanded=False):
            st.checkbox("가중치 적용", value=False, key="promo_enabled")
            st.date_input("시작일", key="promo_start")
            st.date_input("종료일", key="promo_end")
            st.number_input(
                "가중치(%)",
                min_value=-100.0,
                max_value=300.0,
                value=30.0,
                step=5.0,
                key="promo_uplift_pct",
            )

        st.divider()
        st.header("입고 반영 가정")
        lag_days = int(
            st.number_input(
                "입고 반영 리드타임(일) – inbound 미기록 시 arrival+N",
                min_value=0,
                max_value=21,
                value=7,
                step=1,
            )
        )

    if not selected_centers:
        st.warning("최소 한 개의 센터를 선택하세요.")
        return
    if not selected_skus:
        st.warning("최소 한 개의 SKU를 선택하세요.")
        return

    selected_centers = [str(center) for center in selected_centers if str(center).strip()]
    selected_skus = [str(sku) for sku in selected_skus if str(sku).strip()]

    cons_params = get_consumption_params_from_ui()
    lookback_days = int(cons_params.get("lookback_days", 28))
    events = list(cons_params.get("events", []))

    st.subheader("요약 KPI")
    today_norm = pd.Timestamp.today().normalize()
    if latest_snapshot_dt is not None:
        proj_days_for_build = max(0, int((end_ts - latest_snapshot_dt).days))
    else:
        proj_days_for_build = max(0, int((end_ts - start_ts).days))
    render_sku_summary_cards(
        snapshot_df,
        data.moves,
        centers=selected_centers,
        skus=selected_skus,
        today=today_norm,
        latest_snapshot=latest_dt,
        lag_days=int(lag_days),
        start=start_ts,
        end=end_ts,
        lookback_days=lookback_days,
        horizon_pad_days=60,
        events=events,
    )

    st.divider()

    if not _validate_timeline_inputs(snapshot_df, data.moves, start_ts, end_ts):
        return

    timeline_actual = build_core_timeline(
        snapshot_df,
        data.moves,
        centers=selected_centers,
        skus=selected_skus,
        start=start_ts,
        end=end_ts,
        today=today_norm,
        lag_days=int(lag_days),
        horizon_days=int(proj_days_for_build),
    )

    if timeline_actual is None or timeline_actual.empty:
        st.info("선택한 조건에 해당하는 타임라인 데이터가 없습니다.")
        return

    cons_start = None
    if latest_snapshot_dt is not None:
        cons_start = (latest_snapshot_dt + pd.Timedelta(days=1)).normalize()

    timeline_forecast = apply_consumption_with_events(
        timeline_actual,
        snapshot_df,
        centers=selected_centers,
        skus=selected_skus,
        start=start_ts,
        end=end_ts,
        lookback_days=lookback_days,
        events=events,
        cons_start=cons_start,
    )

    if timeline_forecast is None or timeline_forecast.empty:
        timeline_forecast = timeline_actual.copy()

    timeline_for_chart = timeline_forecast.copy() if use_cons_forecast else timeline_actual.copy()

    render_step_chart(
        timeline_for_chart,
        start=start_ts,
        end=end_ts,
        centers=selected_centers,
        skus=selected_skus,
        show_production=show_prod,
        show_in_transit=show_transit,
        today=today_norm,
    )

    def _tidy_from_pivot(
        pivot: Optional[pd.DataFrame], mask: Optional[Sequence[bool]]
    ) -> pd.DataFrame:
        if pivot is None or pivot.empty:
            return pd.DataFrame(columns=["date", "resource_code", "stock_qty"])
        subset = pivot if mask is None else pivot.loc[mask]
        if subset.empty:
            return pd.DataFrame(columns=["date", "resource_code", "stock_qty"])
        tidy = (
            subset.stack()
            .reset_index()
            .rename(columns={"level_0": "date", "level_1": "resource_code", 0: "stock_qty"})
        )
        tidy["date"] = pd.to_datetime(tidy["date"]).dt.normalize()
        tidy["stock_qty"] = pd.to_numeric(tidy["stock_qty"], errors="coerce").fillna(0)
        return tidy
    # -------------------- Amazon US sales vs. inventory --------------------
    amazon_centers = [
        c
        for c in selected_centers
        if isinstance(c, str) and (c.upper().startswith("AMZ") or "AMAZON" in c.upper())
    ]
    if not amazon_centers and "AMZUS" in selected_centers:
        amazon_centers = ["AMZUS"]

    st.divider()
    st.subheader("Amazon US 일별 판매 vs. 재고")

    if not amazon_centers:
        st.info("Amazon 계열 센터가 선택되지 않았습니다.")
    else:
        amz_inv_pivot = _timeline_inventory_matrix(
            timeline_for_chart,
            centers=amazon_centers,
            skus=selected_skus,
            start=start_ts,
            end=end_ts,
        )
        if amz_inv_pivot is not None:
            mask_actual = amz_inv_pivot.index <= today_norm
            mask_forecast = amz_inv_pivot.index > today_norm
        else:
            mask_actual = None
            mask_forecast = None
        inv_actual_from_step = _tidy_from_pivot(amz_inv_pivot, mask_actual)
        inv_forecast_from_step = _tidy_from_pivot(amz_inv_pivot, mask_forecast)
        sku_colors_map = _sku_color_map(selected_skus)

        snapshot_raw_df = load_snapshot_raw()
        amz_ctx = build_amazon_forecast_context(
            snap_long=snapshot_df,
            moves=data.moves,
            snapshot_raw=snapshot_raw_df,
            centers=amazon_centers,
            skus=selected_skus,
            start=start_ts,
            end=end_ts,
            today=today_norm,
            lookback_days=int(lookback_days),
            promotion_events=events,
            use_consumption_forecast=use_cons_forecast,
        )
        render_amazon_sales_vs_inventory(
            amz_ctx,
            inv_actual=inv_actual_from_step,
            inv_forecast=inv_forecast_from_step,
            sku_colors=sku_colors_map,
            use_inventory_for_sales=True,
        )




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
    if "resource_name" in snapshot_df.columns:
        name_rows = snapshot_df.loc[snapshot_df["resource_name"].notna(), [
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

    st.markdown("#### 🛠 생산중 (WIP) 진행 현황")
    if not arr_wip.empty:
        if resource_name_map:
            arr_wip["resource_name"] = arr_wip["resource_code"].map(resource_name_map).fillna("")
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

    if snapshot_df.empty or "date" not in snapshot_df.columns:
        st.info("스냅샷 데이터가 없습니다.")
        return

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
