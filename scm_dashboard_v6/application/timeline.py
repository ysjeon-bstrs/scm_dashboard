"""v6 애플리케이션 계층에서 v5 타임라인 파이프라인을 위임 호출한다."""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from scm_dashboard_v5.pipeline import BuildInputs as _BuildInputs
from scm_dashboard_v5.pipeline import build_timeline_bundle as _build_timeline_bundle

BuildInputs = _BuildInputs

__all__ = ["BuildInputs", "build_timeline_bundle"]


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
):
    """v5 파이프라인과 동일한 인터페이스로 타임라인 번들을 생성한다.

    Parameters
    ----------
    inputs : BuildInputs
        v5에서 정의한 `snapshot`·`moves` 데이터프레임 쌍을 보관하는 불변 입력값.
    centers : Iterable[str]
        필터링할 센터 코드 목록. `TimelineContext`의 축이 되므로 순서가 유지된다.
    skus : Iterable[str]
        관심 있는 SKU 코드 목록. 센터와 동일하게 필터링 및 정렬에 사용된다.
    start : pandas.Timestamp
        타임라인 시작일. v5 컨텍스트에 그대로 전달되어 날짜 인덱스를 초기화한다.
    end : pandas.Timestamp
        타임라인 종료일. 필요 시 horizon_days와 함께 확장된다.
    today : pandas.Timestamp
        현재 기준일. 이동 데이터의 지연(lag) 계산 및 WIP 산정에 활용된다.
    lag_days : int, optional
        입고 지연을 고려할 기본 일수. 음수 입력을 방지하기 위해 v5 단계에서 보정된다.
    horizon_days : int, optional
        미래 스케줄을 확장할 추가 일수. 0보다 작으면 v5 단계에서 0으로 처리된다.
    move_fallback_days : int, optional
        이동 스케줄이 누락된 경우 적용할 기본 납기 일수. v5 단계에서 음수를 보정한다.

    Returns
    -------
    TimelineBundle
        센터별, 입고/출고, WIP 시계열을 포함한 v5 `TimelineBuilder` 결과 객체.
    """

    # v6에서는 입력 사전 검증이나 변환을 수행하지 않고, v5 파이프라인의 계약을 그대로 따른다.
    # BuildInputs(snapshot, moves) 구조를 변경하면 v5 `BuildInputs` 데이터클래스와 호환되지 않으므로
    # 모든 데이터 전처리는 상위 단계(예: 컨트롤러, API)에서 완료된 것을 전제한다.
    return _build_timeline_bundle(
        inputs,
        centers=centers,
        skus=skus,
        start=start,
        end=end,
        today=today,
        lag_days=lag_days,
        horizon_days=horizon_days,
        move_fallback_days=move_fallback_days,
    )
