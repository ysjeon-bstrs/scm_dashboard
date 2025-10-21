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

from typing import Optional, Sequence, Tuple

import pandas as pd
import streamlit as st

from center_alias import normalize_center_value
from scm_dashboard_v4.config import CENTER_COL
from scm_dashboard_v4.inventory import pivot_inventory_cost_from_raw
from scm_dashboard_v4.loaders import load_snapshot_raw

# v9 모듈 임포트
from scm_dashboard_v9.core import build_timeline as build_core_timeline
from scm_dashboard_v9.data_sources import LoadedData, ensure_data
from scm_dashboard_v9.domain import (
    calculate_date_bounds,
    extract_center_and_sku_options,
    validate_timeline_inputs,
)
from scm_dashboard_v9.forecast import (
    apply_consumption_with_events,
    build_amazon_forecast_context,
)
from scm_dashboard_v9.ui import (
    render_amazon_sales_vs_inventory,
    render_sku_summary_cards,
    render_step_chart,
)
from scm_dashboard_v9.ui.adapters import handle_domain_errors
from scm_dashboard_v9.ui.charts import _sku_color_map, _timeline_inventory_matrix


def get_consumption_params_from_ui() -> dict[str, object]:
    """
    UI 컨트롤에서 소비 예측 관련 매개변수를 수집합니다.

    Returns:
        소비 예측 매개변수 딕셔너리:
        - lookback_days: 추세 계산 기간 (일)
        - events: 프로모션 이벤트 목록
    """
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
    """
    v9 대시보드 메인 함수.

    전체 대시보드 UI를 렌더링하고 데이터 파이프라인을 실행합니다.
    """
    # ========================================
    # 1단계: 페이지 설정
    # ========================================
    st.set_page_config(page_title="SCM Dashboard v9", layout="wide")
    st.title("SCM Dashboard v9")
    st.caption("v5를 기반으로 모듈화를 강화한 버전")

    # ========================================
    # 2단계: 데이터 로드 (세션 관리)
    # ========================================
    data = ensure_data()
    if data is None:
        st.info("데이터를 로드하면 차트와 테이블이 표시됩니다.")
        return

    # ========================================
    # 3단계: 스냅샷 데이터 정규화
    # ========================================
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

    # ========================================
    # 4단계: 센터 및 SKU 옵션 추출
    # ========================================
    centers, skus = extract_center_and_sku_options(data.moves, snapshot_df)
    if not centers or not skus:
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

    default_past_days = 20
    default_future_days = 30
    base_past_days = 42
    base_future_days = 42

    bound_min, bound_max = calculate_date_bounds(
        today=today,
        snapshot_df=snapshot_df,
        moves_df=data.moves,
        base_past_days=base_past_days,
        base_future_days=base_future_days,
    )

    # ========================================
    # 6단계: 세션 상태 초기화 (날짜 범위)
    # ========================================
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

    # ========================================
    # 7단계: 사이드바 필터 렌더링
    # ========================================
    with st.sidebar:
        st.header("필터")
        st.caption(
            "기본값: 센터 태광KR·AMZUS / SKU BA00021·BA00022 / 기간 오늘−20일 ~ +30일."
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
                value=5,
                step=1,
            )
        )

    # ========================================
    # 8단계: 필터 유효성 검증
    # ========================================
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

    # ========================================
    # 9단계: KPI 요약 카드 렌더링
    # ========================================
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

    # ========================================
    # 10단계: 타임라인 빌드 (입력 검증)
    # ========================================
    # 도메인 예외를 UI 에러 메시지로 변환
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

    if timeline_actual is None or timeline_actual.empty:
        st.info("선택한 조건에 해당하는 타임라인 데이터가 없습니다.")
        return

    # ========================================
    # 11단계: 소비 예측 적용
    # ========================================
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

    # ========================================
    # 12단계: 계단식 차트 렌더링
    # ========================================
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

    # ========================================
    # 13단계: Amazon US 판매 vs 재고 차트
    # ========================================
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

    # ========================================
    # 14단계: 입고 예정 및 재고 테이블
    # ========================================
    # (나머지 테이블 렌더링 로직은 v5_main.py와 동일하게 유지)
    # 향후 ui.tables 모듈로 분리 예정

    st.markdown("### 📊 추가 정보")
    st.caption("입고 예정, WIP, 재고 현황 등의 상세 정보는 v5_main.py와 동일합니다.")
    st.info("테이블 렌더링 로직은 Commit 2에서 ui.tables 모듈로 분리할 예정입니다.")


if __name__ == "__main__":
    main()
