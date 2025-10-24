"""
타임라인 섹션 유스케이스 (v7)

설명(한글):
- v5 코어/소비 보정 로직을 안전하게 래핑하여 동일 동작을 보장합니다.
- 본 모듈은 DataFrame 입출력만 수행하며, UI 의존이 없습니다.
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd

from scm_dashboard_v5.core import build_timeline as v5_build_timeline
from scm_dashboard_v5.forecast import apply_consumption_with_events as v5_apply_consumption
from scm_dashboard_v7.ui.charts.step import render_step_chart


def render_timeline_section(
    *,
    snapshot: pd.DataFrame,
    moves: pd.DataFrame,
    centers: List[str],
    skus: List[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    today: pd.Timestamp,
    lookback_days: int,
    lag_days: int,
    promotion_events: Optional[List[dict]] = None,
    show_production: bool,
    show_in_transit: bool,
) -> pd.DataFrame:
    """
    타임라인 생성 + (옵션)소비 반영 + 스텝 차트 렌더링을 한 번에 수행.

    반환값: 최종 차트에 사용된 타임라인 DataFrame (후속 비교/검증에 활용 가능)
    """

    # v5 코어로 타임라인 생성
    timeline = v5_build_timeline(
        snapshot,
        moves,
        centers=centers,
        skus=skus,
        start=start,
        end=end,
        today=today,
        lag_days=int(lag_days),
        horizon_days=max(0, int((end - today).days)),
    )

    if timeline is None or timeline.empty:
        return timeline

    # 소비 기반 보정 적용 (옵션)
    timeline_adj = v5_apply_consumption(
        timeline,
        snapshot,
        centers=centers,
        skus=skus,
        start=start,
        end=end,
        lookback_days=int(lookback_days),
        events=(promotion_events or []),
    )
    if timeline_adj is None or timeline_adj.empty:
        timeline_adj = timeline

    # 차트 렌더링 (v7 래퍼)
    render_step_chart(
        timeline_adj,
        centers=centers,
        skus=skus,
        start=start,
        end=end,
        today=today,
        show_wip=show_production,
        show_in_transit=show_in_transit,
        title="선택한 SKU × 센터 재고",
    )

    return timeline_adj


