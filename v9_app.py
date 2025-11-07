"""
SCM Dashboard v9 메인 엔트리 포인트

이 파일은 v5_main.py의 모듈화된 버전으로,
로직을 도메인/데이터/UI 계층으로 분리하여 간결성을 개선했습니다.

주요 변경사항 (v5 대비):
- 1041줄 → ~300줄로 감소
- 데이터 로딩: data_sources 모듈로 분리
- 필터/검증: domain 모듈로 분리
- 테이블 렌더링: 주요 로직 유지 (향후 ui.tables로 분리 예정)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import streamlit as st

# 로깅 설정
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

from center_alias import normalize_center_value

# v9 모듈 임포트
from scm_dashboard_v9.analytics import pivot_inventory_cost_from_raw
from scm_dashboard_v9.core import build_timeline as build_core_timeline
from scm_dashboard_v9.core.config import CENTER_COL, CONFIG
from scm_dashboard_v9.data_sources import LoadedData, ensure_data
from scm_dashboard_v9.domain import (
    calculate_date_bounds,
    extract_center_and_sku_options,
    filter_by_centers,
    is_empty_or_none,
    safe_to_datetime,
    validate_timeline_inputs,
)
from scm_dashboard_v9.forecast import (
    apply_consumption_with_events,
    build_amazon_forecast_context,
)
from scm_dashboard_v9.ui import (
    build_amazon_snapshot_kpis,
    build_shopee_snapshot_kpis,
    render_amazon_sales_vs_inventory,
    render_amazon_snapshot_kpis,
    render_shopee_snapshot_kpis,
    render_sku_summary_cards,
    render_step_chart,
    render_taekwang_stock_dashboard,
)
from scm_dashboard_v9.ui.adapters import handle_domain_errors
from scm_dashboard_v9.ui.charts import _sku_color_map, _timeline_inventory_matrix
from scm_dashboard_v9.ui.tables import (
    build_resource_name_map,
    render_inbound_and_wip_tables,
    render_inventory_table,
    render_lot_details,
)


def _validate_data_quality(
    snapshot: pd.DataFrame,
    moves: pd.DataFrame,
) -> Tuple[bool, Optional[str]]:
    """
    데이터 품질을 검증합니다.

    Args:
        snapshot: 스냅샷 데이터프레임
        moves: 이동 원장 데이터프레임

    Returns:
        (is_valid, error_message) 튜플
    """
    # 필수 컬럼 검증
    required_snapshot_cols = ["resource_code", "center"]
    missing_snap_cols = [
        col for col in required_snapshot_cols if col not in snapshot.columns
    ]
    if missing_snap_cols:
        return (
            False,
            f"스냅샷 데이터에 필수 컬럼이 없습니다: {', '.join(missing_snap_cols)}",
        )

    required_moves_cols = ["resource_code", "to_center", "qty_ea"]
    missing_move_cols = [col for col in required_moves_cols if col not in moves.columns]
    if missing_move_cols:
        return (
            False,
            f"이동 원장에 필수 컬럼이 없습니다: {', '.join(missing_move_cols)}",
        )

    # 데이터 크기 검증
    if len(snapshot) == 0:
        return False, "스냅샷 데이터가 비어있습니다"

    if len(moves) == 0:
        logger.warning("이동 원장이 비어있음 (경고)")

    # 중복 데이터 검증
    if "date" in snapshot.columns:
        dup_count = snapshot.duplicated(
            subset=["date", "center", "resource_code"]
        ).sum()
        if dup_count > 0:
            logger.warning(f"스냅샷에 중복 데이터 {dup_count}건 발견")

    return True, None


def get_consumption_params_from_ui() -> dict[str, object]:
    """
    UI 컨트롤에서 소비 예측 관련 매개변수를 수집합니다.

    Returns:
        소비 예측 매개변수 딕셔너리:
        - lookback_days: 추세 계산 기간 (일)
        - events: 프로모션 이벤트 목록
    """
    lookback_days = int(
        st.session_state.get(
            "trend_lookback_days", CONFIG.consumption.default_lookback_days
        )
    )
    promo_on = bool(st.session_state.get("promo_enabled", False))
    promo_start = st.session_state.get("promo_start")
    promo_end = st.session_state.get("promo_end")
    promo_uplift = float(st.session_state.get("promo_uplift_pct", 0.0)) / 100.0

    events: list[dict[str, object]] = []
    if promo_on and promo_start and promo_end and promo_uplift != 0.0:
        # uplift 값을 설정된 범위로 클램핑
        promo_uplift = max(
            CONFIG.consumption.min_promo_uplift,
            min(promo_uplift, CONFIG.consumption.max_promo_uplift),
        )
        events.append(
            {
                "start": pd.to_datetime(promo_start),
                "end": pd.to_datetime(promo_end),
                "uplift": promo_uplift,
            }
        )

    return {"lookback_days": lookback_days, "events": events}


def _render_sidebar_filters(
    *,
    centers: List[str],
    skus: List[str],
    bound_min: pd.Timestamp,
    bound_max: pd.Timestamp,
    today: pd.Timestamp,
    default_past_days: int,
    default_future_days: int,
) -> Dict[str, Any]:
    """
    사이드바 필터를 렌더링하고 선택된 값들을 반환합니다.

    6-7단계: 세션 상태 초기화 & 사이드바 필터 렌더링
    """

    # 날짜 범위 클램핑 함수
    def _clamp_range(
        range_value: Tuple[pd.Timestamp, pd.Timestamp],
    ) -> Tuple[pd.Timestamp, pd.Timestamp]:
        start_val, end_val = range_value
        start_val = pd.Timestamp(start_val).normalize()
        end_val = pd.Timestamp(end_val).normalize()
        start_val = max(min(start_val, bound_max), bound_min)
        end_val = max(min(end_val, bound_max), bound_min)
        if end_val < start_val:
            end_val = start_val
        return (start_val, end_val)

    # 세션 상태 초기화
    def _init_range() -> None:
        if "date_range" not in st.session_state:
            default_start = max(today - pd.Timedelta(days=default_past_days), bound_min)
            default_end = min(today + pd.Timedelta(days=default_future_days), bound_max)
            if default_start > default_end:
                default_start = default_end
            st.session_state.date_range = (default_start, default_end)
        else:
            st.session_state.date_range = _clamp_range(
                tuple(st.session_state.date_range)
            )

    _init_range()

    # 사이드바 필터 렌더링
    with st.sidebar:
        # 데이터 새로고침 버튼
        if st.button("🔄 시트 새로고침", key="sidebar_gsheet_refresh", use_container_width=True):
            st.session_state["_trigger_refresh"] = True
            st.rerun()

        st.divider()
        st.header("필터")

        preset_centers = ["태광KR", "AMZUS"]
        default_centers = [c for c in preset_centers if c in centers]
        if not default_centers:
            default_centers = centers
        selected_centers = st.multiselect("센터", centers, default=default_centers)

        preset_skus = ["BA00021", "BA00022", "BA00047"]
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
        st.caption(
            "기본값: 센터 태광KR·AMZUS / SKU BA00021·BA00022·BA00047 / 기간 오늘−20일 ~ +30일"
        )

        st.divider()
        st.header("표시 옵션")
        show_prod = st.checkbox("생산중 표시", value=False)
        show_transit = False
        st.caption("체크 시 계단식 차트에 생산중 라인이 표시됩니다.")

        use_cons_forecast = st.checkbox("추세 기반 재고 예측", value=True)
        st.subheader("추세 계산 설정")
        lookback_days = int(
            st.number_input(
                "추세 계산 기간(일)",
                min_value=CONFIG.consumption.min_lookback_days,
                max_value=CONFIG.consumption.max_lookback_days,
                value=CONFIG.consumption.default_lookback_days,
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
                value=CONFIG.timeline.default_lag_days,
                step=1,
            )
        )

    return {
        "selected_centers": selected_centers,
        "selected_skus": selected_skus,
        "start_ts": start_ts,
        "end_ts": end_ts,
        "show_prod": show_prod,
        "show_transit": show_transit,
        "use_cons_forecast": use_cons_forecast,
        "lookback_days": lookback_days,
        "lag_days": lag_days,
    }


def _tidy_from_pivot(
    pivot: Optional[pd.DataFrame], mask: Optional[Sequence[bool]]
) -> pd.DataFrame:
    """
    피벗 테이블을 tidy 형식으로 변환합니다.

    Args:
        pivot: 피벗된 재고 데이터프레임
        mask: 필터링할 행 마스크 (선택적)

    Returns:
        tidy 형식의 데이터프레임 (date, resource_code, stock_qty 컬럼)
    """
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


def _filter_amazon_centers(selected_centers: List[str]) -> List[str]:
    """
    선택된 센터에서 Amazon 계열 센터만 필터링합니다.

    Args:
        selected_centers: 선택된 센터 목록

    Returns:
        Amazon 계열 센터 목록
    """
    amazon_centers = [
        c
        for c in selected_centers
        if isinstance(c, str) and (c.upper().startswith("AMZ") or "AMAZON" in c.upper())
    ]
    if not amazon_centers and "AMZUS" in selected_centers:
        amazon_centers = ["AMZUS"]
    return amazon_centers


def _infer_amazon_centers_from_snapshot(snapshot_df: pd.DataFrame) -> List[str]:
    """Return every Amazon-affiliated center detected in the snapshot."""

    if "center" not in snapshot_df.columns:
        return []

    centers = snapshot_df["center"].dropna().astype(str).str.strip()
    amazon_centers = [
        center
        for center in centers.unique()
        if center and (center.upper().startswith("AMZ") or "AMAZON" in center.upper())
    ]
    return sorted(amazon_centers)


def _build_amazon_kpi_data(
    *,
    snap_amz: pd.DataFrame,
    selected_skus: List[str],
    amazon_centers: List[str],
    show_delta: bool,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Amazon KPI 데이터를 빌드합니다.

    Args:
        snap_amz: Amazon 스냅샷 데이터
        selected_skus: 선택된 SKU 목록
        amazon_centers: Amazon 센터 목록
        show_delta: 전 스냅샷 대비 델타 표시 여부

    Returns:
        (kpi_df, previous_df) 튜플
    """
    kpi_df = build_amazon_snapshot_kpis(
        snap_amz,
        skus=selected_skus,
        center=amazon_centers,
        cover_base="available",
        use_ma7=True,
    )
    previous_df = None
    if show_delta and kpi_df is not None and not kpi_df.empty:
        latest_snap_ts = pd.to_datetime(kpi_df["snap_time"].max())
        if not pd.isna(latest_snap_ts):
            # snap_time이 모두 null이면 date 컬럼 사용
            time_col = "snap_time" if snap_amz["snap_time"].notna().any() else "date"

            # 성능 최적화: 필터링을 직접 수행하여 불필요한 copy 제거
            snap_prev_ts = pd.to_datetime(snap_amz[time_col], errors="coerce")
            snap_prev_mask = (snap_prev_ts.notna()) & (snap_prev_ts < latest_snap_ts)
            snap_prev = snap_amz[snap_prev_mask]
            if not snap_prev.empty:
                previous_df = build_amazon_snapshot_kpis(
                    snap_prev,
                    skus=selected_skus,
                    center=amazon_centers,
                    cover_base="available",
                    use_ma7=True,
                )
    return kpi_df, previous_df


def _build_shopee_kpi_data(
    *,
    snapshot_df: pd.DataFrame,
    selected_skus: List[str],
    shopee_centers: List[str],
    show_delta: bool,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    SHOPEE KPI 데이터를 빌드합니다.

    Args:
        snapshot_df: 스냅샷 데이터
        selected_skus: 선택된 SKU 목록
        shopee_centers: SHOPEE 센터 목록
        show_delta: 전 스냅샷 대비 델타 표시 여부

    Returns:
        (kpi_df, previous_df) 튜플
    """
    kpi_df = build_shopee_snapshot_kpis(
        snapshot_df,
        skus=selected_skus,
        centers=shopee_centers,
    )
    previous_df = None
    if show_delta and kpi_df is not None and not kpi_df.empty:
        # 센터별로 이전 스냅샷 찾기 (각 센터의 2번째 최신 시간)
        prev_snapshots = []
        debug_info = []  # 디버그용
        for center in shopee_centers:
            center_kpi = kpi_df[kpi_df["center"] == center]
            if center_kpi.empty:
                debug_info.append(f"{center}: KPI 데이터 없음")
                continue

            # 해당 센터의 현재 최신 시간
            center_latest_ts = pd.to_datetime(center_kpi["snap_time"].max())
            if pd.isna(center_latest_ts):
                debug_info.append(f"{center}: 최신 시간 없음")
                continue

            # 해당 센터에 대해 time_col 결정 (센터별로!)
            center_mask = snapshot_df["center"] == center
            center_data = snapshot_df[center_mask]

            # 이 센터에 snap_time이 있고 유효한 값이 있으면 사용, 아니면 date 사용
            center_time_col = "snap_time"
            if "snap_time" not in center_data.columns or not center_data[
                "snap_time"
            ].notna().any():
                center_time_col = "date"

            # 해당 센터의 모든 스냅샷 시간 가져오기
            snap_times = pd.to_datetime(center_data[center_time_col], errors="coerce")
            center_times = center_data[snap_times.notna()][center_time_col].unique()
            center_times_sorted = sorted(
                [pd.to_datetime(t) for t in center_times], reverse=True
            )

            # 2번째 최신 시간 찾기 (현재 최신 제외)
            prev_times = [t for t in center_times_sorted if t < center_latest_ts]

            # 디버그 정보 수집
            debug_info.append(
                f"{center}: time_col={center_time_col}, 최신={center_latest_ts:%Y-%m-%d %H:%M}, "
                f"전체시간={len(center_times_sorted)}개, "
                f"이전시간={len(prev_times)}개"
            )
            if prev_times:
                prev_latest_ts = prev_times[0]  # 바로 이전 스냅샷
                debug_info.append(f"  → 이전={prev_latest_ts:%Y-%m-%d %H:%M}")
                # 정확히 그 시간의 데이터만 선택
                prev_mask = center_mask & (
                    pd.to_datetime(snapshot_df[center_time_col], errors="coerce")
                    == prev_latest_ts
                )
                center_prev = snapshot_df[prev_mask]
                if not center_prev.empty:
                    prev_snapshots.append(center_prev)
                    debug_info.append(f"  → 데이터 {len(center_prev)}행 발견")
                else:
                    debug_info.append(f"  → 데이터 없음!")
            else:
                debug_info.append(f"  → 이전 스냅샷 시간 없음!")


        # 모든 센터의 이전 스냅샷 합치기
        if prev_snapshots:
            snap_prev = pd.concat(prev_snapshots, ignore_index=True)
            previous_df = build_shopee_snapshot_kpis(
                snap_prev,
                skus=selected_skus,
                centers=shopee_centers,
            )

        # 디버그 정보를 세션에 저장
        st.session_state["_shopee_delta_debug"] = debug_info
    return kpi_df, previous_df


def _render_amazon_section(
    *,
    selected_centers: List[str],
    snapshot_df: pd.DataFrame,
    selected_skus: List[str],
    timeline_for_chart: pd.DataFrame,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    today_norm: pd.Timestamp,
    moves_df: pd.DataFrame,
    lookback_days: int,
    events: List[Dict[str, Any]],
    use_cons_forecast: bool,
    lag_days: int,
    horizon_days: int,
    latest_snapshot_dt: Optional[pd.Timestamp],
) -> None:
    """
    Amazon US 판매 vs. 재고 차트 섹션을 렌더링합니다.

    13단계: Amazon US 판매 vs 재고 차트
    """
    amazon_centers = _filter_amazon_centers(selected_centers)
    fallback_centers: List[str] = []
    if not amazon_centers:
        fallback_centers = _infer_amazon_centers_from_snapshot(snapshot_df)
        amazon_centers = fallback_centers

    st.divider()
    st.subheader("Amazon US 대시보드")

    if not amazon_centers:
        st.info("Amazon 계열 센터 데이터를 찾을 수 없습니다.")
        return

    if fallback_centers:
        st.caption("선택된 센터와 무관하게 Amazon 데이터를 표시합니다.")

    sku_colors_map = _sku_color_map(selected_skus)
    snap_amz = filter_by_centers(snapshot_df, amazon_centers)

    # Amazon KPI 설정 토글
    # 설정: 전 스냅샷 대비 Δ만 유지 (커버일 기준 토글 제거)
    show_delta = st.toggle("전 스냅샷 대비 Δ", value=True)

    kpi_df, previous_df = _build_amazon_kpi_data(
        snap_amz=snap_amz,
        selected_skus=selected_skus,
        amazon_centers=amazon_centers,
        show_delta=show_delta,
    )

    # SKU → 품명 매핑
    amz_resource_name_map = build_resource_name_map(snap_amz)

    render_amazon_snapshot_kpis(
        kpi_df,
        sku_colors=sku_colors_map,
        show_delta=show_delta,
        previous_df=previous_df,
        max_cols=4,
        resource_name_map=amz_resource_name_map,
    )

    # 재고 피벗 생성 및 실제/예측 분리
    amz_inv_pivot = _timeline_inventory_matrix(
        timeline_for_chart,
        centers=amazon_centers,
        skus=selected_skus,
        start=start_ts,
        end=end_ts,
    )

    amazon_timeline_for_chart: Optional[pd.DataFrame] = timeline_for_chart
    if amz_inv_pivot is None:
        amazon_timeline_actual = build_core_timeline(
            snapshot_df,
            moves_df,
            centers=amazon_centers,
            skus=selected_skus,
            start=start_ts,
            end=end_ts,
            today=today_norm,
            lag_days=int(lag_days),
            horizon_days=int(max(0, horizon_days)),
        )

        amazon_timeline_for_chart = amazon_timeline_actual
        if (
            use_cons_forecast
            and amazon_timeline_actual is not None
            and not amazon_timeline_actual.empty
        ):
            cons_start = None
            if latest_snapshot_dt is not None:
                cons_start = (latest_snapshot_dt + pd.Timedelta(days=1)).normalize()

            amazon_timeline_forecast = apply_consumption_with_events(
                amazon_timeline_actual,
                snapshot_df,
                centers=amazon_centers,
                skus=selected_skus,
                start=start_ts,
                end=end_ts,
                lookback_days=lookback_days,
                events=events,
                cons_start=cons_start,
            )
            if (
                amazon_timeline_forecast is not None
                and not amazon_timeline_forecast.empty
            ):
                amazon_timeline_for_chart = amazon_timeline_forecast

        amz_inv_pivot = _timeline_inventory_matrix(
            amazon_timeline_for_chart,
            centers=amazon_centers,
            skus=selected_skus,
            start=start_ts,
            end=end_ts,
        )

    mask_actual = None
    mask_forecast = None
    if amz_inv_pivot is not None:
        mask_actual = amz_inv_pivot.index <= today_norm
        mask_forecast = amz_inv_pivot.index > today_norm
    inv_actual_from_step = _tidy_from_pivot(amz_inv_pivot, mask_actual)
    inv_forecast_from_step = _tidy_from_pivot(amz_inv_pivot, mask_forecast)

    # snap_정제 시트의 sales_qty 컬럼을 사용하여 판매 데이터 로드
    # (snapshot_raw의 fba_output_stock 대신 snap_정제의 sales_qty 사용)
    amz_ctx = build_amazon_forecast_context(
        snap_long=snapshot_df,
        moves=moves_df,
        snapshot_raw=snapshot_df,  # snap_정제 데이터 전달 (sales_qty 컬럼 포함)
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


def main() -> None:
    """
    v9 대시보드 메인 함수.

    전체 대시보드 UI를 렌더링하고 데이터 파이프라인을 실행합니다.
    """
    logger.info("SCM Dashboard v9 시작")

    # ========================================
    # 1단계: 페이지 설정
    # ========================================
    st.set_page_config(page_title="SCM Dashboard v9", layout="wide")
    st.title("SCM Dashboard v9")

    # ========================================
    # 2단계: 데이터 로드 (세션 관리)
    # ========================================
    logger.info("데이터 로드 시작")
    data = ensure_data()
    if data is None:
        logger.warning("데이터가 로드되지 않음")
        st.info("데이터를 로드하면 차트와 테이블이 표시됩니다.")
        return
    logger.info(
        f"데이터 로드 완료: 스냅샷 {len(data.snapshot)}행, 이동 {len(data.moves)}행"
    )

    # 데이터 품질 검증
    is_valid, error_msg = _validate_data_quality(data.snapshot, data.moves)
    if not is_valid:
        logger.error(f"데이터 검증 실패: {error_msg}")
        st.error(f"데이터 품질 오류: {error_msg}")
        return

    # ========================================
    # 3단계: 스냅샷 데이터 정규화
    # ========================================
    snapshot_df = data.snapshot.copy()
    if "date" in snapshot_df.columns:
        snapshot_df["date"] = safe_to_datetime(snapshot_df["date"])
    elif "snapshot_date" in snapshot_df.columns:
        snapshot_df["date"] = safe_to_datetime(snapshot_df["snapshot_date"])
    else:
        snapshot_df["date"] = pd.NaT

    # SKU → 품명 매핑을 미리 생성하여 각 대시보드에서 재사용
    resource_name_map = build_resource_name_map(snapshot_df)

    # ========================================
    # 4단계: 센터 및 SKU 옵션 추출
    # ========================================
    centers, skus = extract_center_and_sku_options(data.moves, snapshot_df)
    logger.info(f"센터 {len(centers)}개, SKU {len(skus)}개 추출")
    if not centers or not skus:
        logger.error("센터 또는 SKU 정보 없음")
        st.warning("센터 또는 SKU 정보를 찾을 수 없습니다.")
        return

    # ========================================
    # 5단계: 날짜 범위 계산
    # ========================================
    today = pd.Timestamp.today().normalize()
    snap_dates = snapshot_df["date"].dropna()
    latest_dt = snap_dates.max() if not snap_dates.empty else pd.NaT
    latest_snapshot_dt = (
        None if pd.isna(latest_dt) else pd.to_datetime(latest_dt).normalize()
    )

    # CONFIG에서 설정값 가져오기
    default_past_days = CONFIG.ui.default_past_days
    default_future_days = CONFIG.ui.default_future_days
    base_past_days = CONFIG.ui.base_past_days
    base_future_days = CONFIG.ui.base_future_days

    bound_min, bound_max = calculate_date_bounds(
        today=today,
        snapshot_df=snapshot_df,
        moves_df=data.moves,
        base_past_days=base_past_days,
        base_future_days=base_future_days,
    )

    # ========================================
    # 6-7단계: 세션 상태 초기화 & 사이드바 필터 렌더링
    # ========================================
    filters = _render_sidebar_filters(
        centers=centers,
        skus=skus,
        bound_min=bound_min,
        bound_max=bound_max,
        today=today,
        default_past_days=default_past_days,
        default_future_days=default_future_days,
    )

    selected_centers = filters["selected_centers"]
    selected_skus = filters["selected_skus"]
    start_ts = filters["start_ts"]
    end_ts = filters["end_ts"]
    show_prod = filters["show_prod"]
    show_transit = filters["show_transit"]
    use_cons_forecast = filters["use_cons_forecast"]
    lag_days = filters["lag_days"]

    # ========================================
    # 8단계: 필터 유효성 검증
    # ========================================
    if not selected_centers:
        logger.warning("센터가 선택되지 않음")
        st.warning("최소 한 개의 센터를 선택하세요.")
        return
    if not selected_skus:
        logger.warning("SKU가 선택되지 않음")
        st.warning("최소 한 개의 SKU를 선택하세요.")
        return

    selected_centers = [
        str(center) for center in selected_centers if str(center).strip()
    ]
    selected_skus = [str(sku) for sku in selected_skus if str(sku).strip()]
    logger.info(
        f"필터 적용: 센터 {selected_centers}, SKU {len(selected_skus)}개, 기간 {start_ts} ~ {end_ts}"
    )

    cons_params = get_consumption_params_from_ui()
    lookback_days = int(cons_params.get("lookback_days", 28))
    events = list(cons_params.get("events", []))

    # ========================================
    # 9단계: 타임라인 빌드 (입력 검증)
    # ========================================
    # 도메인 예외를 UI 에러 메시지로 변환
    today_norm = pd.Timestamp.today().normalize()
    if latest_snapshot_dt is not None:
        proj_days_for_build = max(0, int((end_ts - latest_snapshot_dt).days))
    else:
        proj_days_for_build = max(0, int((end_ts - start_ts).days))

    logger.info("타임라인 빌드 시작")
    with handle_domain_errors():
        validate_timeline_inputs(snapshot_df, data.moves, start_ts, end_ts)

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

    # 타임라인이 비어있어도 스냅샷 기반 데이터는 표시되어야 하므로 return하지 않음
    has_timeline_data = timeline_actual is not None and not timeline_actual.empty
    if not has_timeline_data:
        logger.warning("타임라인 데이터가 비어있음 - 스냅샷 기반 데이터만 표시")
    else:
        logger.info(f"타임라인 빌드 완료: {len(timeline_actual)}행")

    # ========================================
    # 10단계: 소비 예측 적용 (타임라인이 있는 경우에만)
    # ========================================
    timeline_for_chart = None
    if has_timeline_data:
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

        # 성능 최적화: 불필요한 copy 제거
        if timeline_forecast is None or timeline_forecast.empty:
            timeline_forecast = timeline_actual

        timeline_for_chart = timeline_forecast if use_cons_forecast else timeline_actual

    # ========================================
    # 11단계: 탭 구조로 대시보드 렌더링
    # ========================================
    tab1, tab2 = st.tabs(["📊 재고 대시보드", "🏢 센터별 대시보드"])

    with tab1:
        # ========================================
        # 재고 대시보드: 요약 KPI
        # ========================================
        with st.expander("📊 요약 KPI", expanded=True):
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
                horizon_pad_days=CONFIG.timeline.horizon_pad_days,
                events=events,
            )

        st.divider()

        # ========================================
        # 재고 대시보드: 계단식 차트
        # ========================================
        if has_timeline_data and timeline_for_chart is not None:
            render_step_chart(
                timeline_for_chart,
                start=start_ts,
                end=end_ts,
                centers=selected_centers,
                skus=selected_skus,
                show_production=show_prod,
                show_in_transit=show_transit,
                today=today_norm,
                snapshot=snapshot_df,
            )
        else:
            st.info("선택한 조건에 해당하는 타임라인 데이터가 없습니다. 다른 센터/SKU/기간을 선택해주세요.")

        st.divider()

        # ========================================
        # 재고 대시보드: 입고 예정 및 WIP 테이블
        # ========================================
        render_inbound_and_wip_tables(
            moves=data.moves,
            snapshot=snapshot_df,
            selected_centers=selected_centers,
            selected_skus=selected_skus,
            start=start_ts,
            end=end_ts,
            lag_days=lag_days,
            today=today_norm,
        )

        # ========================================
        # 재고 대시보드: 재고 현황 테이블
        # ========================================
        display_df = render_inventory_table(
            snapshot=snapshot_df,
            selected_centers=selected_centers,
            latest_dt=latest_dt,
            resource_name_map=resource_name_map,
        )

        # ========================================
        # 재고 대시보드: 로트 상세
        # ========================================
        # center_latest_dates 계산
        center_latest_series = (
            filter_by_centers(snapshot_df, selected_centers).groupby("center")["date"].max()
        )
        center_latest_dates = {
            center: ts.normalize()
            for center, ts in center_latest_series.items()
            if pd.notna(ts)
        }

        visible_skus = (
            display_df.get("SKU", pd.Series(dtype=str))
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        render_lot_details(
            visible_skus=visible_skus,
            selected_centers=selected_centers,
            center_latest_dates=center_latest_dates,
            latest_dt=latest_dt,
        )

    with tab2:
        # ========================================
        # 센터별 대시보드: 태광KR 가상창고
        # ========================================
        # 태광KR 가상창고(운영/키핑) 배분 데이터를 구버전 세션에서도 안전하게 조회
        taekwang_stock_df = getattr(data, "tk_stock_distrib", None)

        if taekwang_stock_df is not None:
            render_taekwang_stock_dashboard(
                taekwang_stock_df,
                selected_skus=selected_skus,
                resource_name_map=resource_name_map,
                sku_colors=_sku_color_map(selected_skus),
                inbound_moves=data.moves,
            )
        else:
            logger.warning(
                "tk_stock_distrib 속성이 없습니다. "
                "Google Sheets에 'tk_stock_distrib' 시트가 있는지 확인하거나, "
                "Streamlit 세션을 새로고침(Ctrl+R)해주세요."
            )

        # ========================================
        # 센터별 대시보드: Amazon US
        # ========================================
        _render_amazon_section(
            selected_centers=selected_centers,
            snapshot_df=snapshot_df,
            selected_skus=selected_skus,
            timeline_for_chart=timeline_for_chart,
            start_ts=start_ts,
            end_ts=end_ts,
            today_norm=today_norm,
            moves_df=data.moves,
            lookback_days=lookback_days,
            events=events,
            use_cons_forecast=use_cons_forecast,
            lag_days=int(lag_days),
            horizon_days=int(proj_days_for_build),
            latest_snapshot_dt=latest_snapshot_dt,
        )

        # ========================================
        # 센터별 대시보드: SHOPEE
        # ========================================
        # SHOPEE 센터 목록 (필터와 무관)
        shopee_centers = ["SBSMY", "SBSSG", "SBSTH", "SBSPH"]

        # 스냅샷에 SHOPEE 센터 데이터가 있는지 확인
        has_shopee_data = False
        if not snapshot_df.empty and "center" in snapshot_df.columns:
            snapshot_centers = snapshot_df["center"].dropna().astype(str).str.strip().unique()
            has_shopee_data = any(c in shopee_centers for c in snapshot_centers)

        if has_shopee_data:
            st.divider()
            # st.expander로 토글 가능하게 만들기 (기본값: 열림)
            with st.expander("🛍️ SHOPEE", expanded=True):
                st.subheader("SHOPEE 대시보드")

                # SHOPEE KPI 설정 토글
                shopee_show_delta = st.toggle("전 스냅샷 대비 Δ", value=True, key="shopee_delta")

                # KPI 데이터 빌드 (현재 + 이전 스냅샷)
                shopee_kpi_df, shopee_previous_df = _build_shopee_kpi_data(
                    snapshot_df=snapshot_df,
                    selected_skus=selected_skus,
                    shopee_centers=shopee_centers,
                    show_delta=shopee_show_delta,
                )

                # KPI 카드 렌더링
                render_shopee_snapshot_kpis(
                    shopee_kpi_df,
                    selected_skus=selected_skus,
                    sku_colors=_sku_color_map(selected_skus),
                    resource_name_map=resource_name_map,
                    show_delta=shopee_show_delta,
                    previous_df=shopee_previous_df,
                    max_cols=4,
                )

    # ========================================
    # 18단계: AI 어시스턴트 (1.5단계 하이브리드)
    # ========================================
    st.divider()
    st.header("🤖 AI 어시스턴트")

    try:
        from ai_chatbot_simple import render_simple_chatbot_tab
        render_simple_chatbot_tab(
            snapshot_df=snapshot_df,
            moves_df=data.moves,
            timeline_df=timeline_for_chart,  # 🆕 30일치 시계열 + 예측!
            selected_centers=selected_centers,
            selected_skus=selected_skus
        )
    except ImportError as e:
        st.warning(f"AI 어시스턴트 모듈을 찾을 수 없습니다: {e}")
        st.info("ai_chatbot_simple.py 파일이 있는지 확인하세요.")
    except Exception as e:
        st.error(f"AI 어시스턴트 로드 실패: {e}")
        st.caption("secrets.toml에 Gemini API 키가 설정되어 있는지 확인하세요.")


if __name__ == "__main__":
    main()
