"""타임라인 페이지 전용 컴포넌트."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd

from scm_dashboard_v5.ui import render_step_chart


@dataclass(frozen=True)
class TimelineControls:
    """계단형 차트에 필요한 기간·센터·SKU 선택값을 묶어 전달하는 데이터 클래스."""

    start: pd.Timestamp
    end: pd.Timestamp
    centers: Sequence[str]
    skus: Sequence[str]
    show_production: bool = False
    show_in_transit: bool = False
    today: pd.Timestamp | None = None


def render_summary_and_timeline(
    timeline_for_chart: pd.DataFrame,
    controls: TimelineControls,
) -> None:
    """선택된 조건과 타임라인 데이터를 이용해 기존 v5 차트를 그대로 그린다."""

    # ✅ v5에서 검증된 렌더러를 그대로 호출해 시각화 결과가 바뀌지 않도록 보장한다.
    today_value = controls.today if controls.today is not None else pd.Timestamp.today().normalize()
    render_step_chart(
        timeline_for_chart,
        start=controls.start,
        end=controls.end,
        centers=list(controls.centers),
        skus=list(controls.skus),
        show_production=controls.show_production,
        show_in_transit=controls.show_in_transit,
        today=today_value,
    )
