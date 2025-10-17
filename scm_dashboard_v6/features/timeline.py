"""
타임라인 섹션 유스케이스 (v6)

- 초기에는 v5 코어/차트를 호출하여 동작을 유지한다.
- 점진적으로 데이터 준비/검증/표현을 분리하고 테스트 용이성을 높인다.
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd

from scm_dashboard_v5.core import build_timeline as v5_build_timeline
from scm_dashboard_v5.forecast import apply_consumption_with_events as v5_apply_consumption
from scm_dashboard_v6.ui.charts.step import render_step_chart


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
    """타임라인 생성 + 소비 반영 + 스텝 차트 렌더링을 한 번에 수행한다.

    반환값: 최종 차트에 사용된 타임라인 데이터프레임(검증/후속처리에 활용 가능)
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

    # 차트 렌더링 (v6 래퍼)
    render_step_chart(
        timeline_adj,
        centers=centers,
        skus=skus,
        start=start,
        end=end,
        today=today,
        # v5 렌더러가 show_wip/ show_production 둘 다 받지만, 명시적으로 show_wip을 전달
        show_wip=show_production,
        show_in_transit=show_in_transit,
        title="선택한 SKU × 센터 재고",
    )

    return timeline_adj

