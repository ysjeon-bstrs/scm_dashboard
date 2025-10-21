"""
사이드바/입력 컨트롤 수집 (v7)

설계(한글):
- v7 초기 단계에서는 v5의 UX를 유지합니다.
- 본 모듈은 세션 상태를 최소화하고, 선택값을 데이터 클래스로 반환합니다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd
import streamlit as st


@dataclass
class UiSelections:
    """사이드바에서 수집한 사용자 선택값 컨테이너"""

    centers: List[str]
    skus: List[str]
    start: pd.Timestamp
    end: pd.Timestamp
    lookback_days: int
    promotion_enabled: bool
    promotion_percent: float
    promotion_start: pd.Timestamp
    promotion_end: pd.Timestamp
    inbound_lead_days: int
    show_production: bool
    show_in_transit: bool


def collect_sidebar_controls(
    *,
    centers: list[str],
    skus: list[str],
    bound_min: pd.Timestamp,
    bound_max: pd.Timestamp,
    default_past_days: int = 20,
    default_future_days: int = 30,
) -> UiSelections:
    """
    사이드바에서 선택값을 수집한다. (v7)

    - 반환값은 테스트 용이성을 위해 DataFrame이 아닌 단순 타입으로 구성한다.
    """

    today = pd.Timestamp.today().normalize()

    with st.sidebar:
        st.header("필터")
        st.caption("기본값: 센터 태광KR·AMZUS / SKU BA00021·BA00022 / 기간 오늘−20일 ~ +30일.")

        preset_centers = ["태광KR", "AMZUS"]
        default_centers = [c for c in preset_centers if c in centers] or centers
        sel_centers = st.multiselect("센터", centers, default=default_centers)

        preset_skus = ["BA00021", "BA00022"]
        default_skus = [s for s in preset_skus if s in skus] or (skus if len(skus) <= 10 else skus[:10])
        sel_skus = st.multiselect("SKU", skus, default=default_skus)

        st.subheader("기간 설정")
        start_dt, end_dt = st.slider(
            "기간",
            min_value=bound_min.to_pydatetime(),
            max_value=bound_max.to_pydatetime(),
            value=(
                max(today - pd.Timedelta(days=default_past_days), bound_min).to_pydatetime(),
                min(today + pd.Timedelta(days=default_future_days), bound_max).to_pydatetime(),
            ),
            format="YYYY-MM-DD",
        )
        start_ts = pd.Timestamp(start_dt).normalize()
        end_ts = pd.Timestamp(end_dt).normalize()

        st.divider()
        st.header("표시 옵션")
        show_prod = st.checkbox("생산중 표시", value=False)
        show_transit = False  # 요청에 따라 이동중은 기본 해제
        st.caption("체크 시 계단식 차트에 생산중 라인이 표시됩니다.")

        st.subheader("추세 계산 설정")
        lookback_days = int(
            st.number_input(
                "추세 계산 기간(스냅샷 기준 최대 42일)",
                min_value=7,
                max_value=42,
                value=28,
                step=7,
                key="v7_trend_lookback_days",
            )
        )

        with st.expander("프로모션 가중치", expanded=False):
            promo_enabled = bool(st.checkbox("가중치 적용", value=False, key="v7_promo_enabled"))
            promo_start = pd.Timestamp(st.date_input("시작일", value=today.to_pydatetime(), key="v7_promo_start")).normalize()
            promo_end = pd.Timestamp(st.date_input("종료일", value=today.to_pydatetime(), key="v7_promo_end")).normalize()
            promo_percent = float(
                st.number_input("가중치(%)", min_value=0.0, max_value=500.0, value=30.0, step=1.0, format="%.2f", key="v7_promo_percent")
            )

        st.subheader("입고 반영 가정")
        inbound_lead_days = int(
            st.number_input(
                "입고 반영 리드타임(일) – inbound 미기록 시 arrival+N",
                min_value=0,
                max_value=30,
                value=5,
                step=1,
                key="v7_inbound_lead_days",
            )
        )

    return UiSelections(
        centers=[str(c) for c in sel_centers if str(c).strip()],
        skus=[str(s) for s in sel_skus if str(s).strip()],
        start=start_ts,
        end=end_ts,
        lookback_days=lookback_days,
        promotion_enabled=promo_enabled,
        promotion_percent=promo_percent,
        promotion_start=promo_start,
        promotion_end=promo_end,
        inbound_lead_days=inbound_lead_days,
        show_production=bool(show_prod),
        show_in_transit=bool(show_transit),
    )


