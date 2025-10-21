"""v8 애플리케이션 계층에서 타임라인 빌드를 담당하는 래퍼 모듈."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from scm_dashboard_v5.domain.models import TimelineBundle
from scm_dashboard_v5.pipeline import BuildInputs as _V5BuildInputs
from scm_dashboard_v5.pipeline import build_timeline_bundle as _v5_build_timeline_bundle


@dataclass(frozen=True)
class BuildInputs:
    """타임라인 계산에 필요한 스냅샷·무빙 데이터프레임을 한 번에 전달하기 위한 컨테이너.

    Attributes:
        snapshot: 재고 스냅샷 원본 DataFrame. v5에서 사용하던 스키마(date/snapshot_date, center 등)를 그대로 따른다.
        moves: 입출고 이력 DataFrame. v5 파이프라인에서 사용하는 move 스키마를 유지한다.
    """

    snapshot: pd.DataFrame
    moves: pd.DataFrame

    def to_v5(self) -> _V5BuildInputs:
        """v5 파이프라인이 기대하는 BuildInputs 객체로 변환한다."""

        return _V5BuildInputs(snapshot=self.snapshot, moves=self.moves)


def build_timeline_bundle(
    inputs: BuildInputs,
    *,
    centers: Iterable[str],
    skus: Iterable[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    today: pd.Timestamp,
    lag_days: int = 7,
    horizon_days: int = 0,
    move_fallback_days: int = 1,
) -> TimelineBundle:
    """동일한 입력을 v5 파이프라인에 전달하여 완전히 동일한 타임라인 결과를 생성한다.

    Args:
        inputs: 스냅샷/무빙 데이터 프레임을 포함한 BuildInputs 인스턴스.
        centers: 분석 대상 센터 목록.
        skus: 분석 대상 SKU 목록.
        start: 타임라인 시작일.
        end: 타임라인 종료일.
        today: 기준일(today)로 사용될 날짜.
        lag_days: 리드타임 보정 일수. 기본값 7일.
        horizon_days: 확장 예측 기간. 기본값 0일.
        move_fallback_days: move 데이터가 비었을 때 보완할 일수. 기본값 1일.

    Returns:
        v5 파이프라인과 동일한 구조의 TimelineBundle.
    """

    # ✅ v5 파이프라인과 완전 동일한 동작을 보장하기 위해 변환만 수행하고 나머지는 위임한다.
    return _v5_build_timeline_bundle(
        inputs.to_v5(),
        centers=centers,
        skus=skus,
        start=start,
        end=end,
        today=today,
        lag_days=lag_days,
        horizon_days=horizon_days,
        move_fallback_days=move_fallback_days,
    )
