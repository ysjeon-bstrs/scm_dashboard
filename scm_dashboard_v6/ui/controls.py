"""
사이드바/입력 컨트롤 수집 (v6) — 초기에는 v5 로직을 래핑하여 동작 유지
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd
import streamlit as st


@dataclass
class UiSelections:
    centers: List[str]
    skus: List[str]
    start: pd.Timestamp
    end: pd.Timestamp
    lookback_days: int
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
    """사이드바에서 선택값을 수집한다. (v6)

    - v6 초기 단계: v5의 기본 UX를 유지한다.
    - 향후에는 프리셋/권장값/검증을 강화한다.
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
        show_transit = False  # 요청에 따라 이동중 노출은 기본 해제
        st.caption("체크 시 계단식 차트에 생산중 라인이 표시됩니다.")

        st.subheader("추세 계산 설정")
        lookback_days = int(
            st.number_input(
                "추세 계산 기간(일)", min_value=7, max_value=56, value=28, step=7, key="trend_lookback_days"
            )
        )

    return UiSelections(
        centers=[str(c) for c in sel_centers if str(c).strip()],
        skus=[str(s) for s in sel_skus if str(s).strip()],
        start=start_ts,
        end=end_ts,
        lookback_days=lookback_days,
        show_production=bool(show_prod),
        show_in_transit=bool(show_transit),
    )

