"""
v9 타임라인 빌더 테스트

TimelineBuilder 및 관련 유틸리티 함수들의 동작을 검증합니다.
"""

from __future__ import annotations

import pandas as pd
import pytest

from scm_dashboard_v9.planning.timeline import (
    TimelineBuilder,
    TimelineContext,
    prepare_moves,
    prepare_snapshot,
)
from scm_dashboard_v9.planning.series import SeriesIndex


@pytest.fixture
def sample_snapshot() -> pd.DataFrame:
    """테스트용 스냅샷 데이터"""
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-03",
                ]
            ),
            "center": ["태광KR", "AMZUS", "태광KR", "AMZUS", "태광KR", "AMZUS"],
            "resource_code": [
                "BA00021",
                "BA00021",
                "BA00021",
                "BA00021",
                "BA00021",
                "BA00021",
            ],
            "stock_qty": [100, 200, 90, 210, 85, 205],
        }
    )


@pytest.fixture
def sample_moves() -> pd.DataFrame:
    """테스트용 이동 원장"""
    return pd.DataFrame(
        {
            "resource_code": ["BA00021", "BA00021", "BA00022"],
            "qty_ea": [50, 30, 100],
            "carrier_mode": ["SEA", "AIR", "WIP"],
            "from_center": ["상해CN", "태광KR", "WIP"],
            "to_center": ["태광KR", "AMZUS", "태광KR"],
            "onboard_date": pd.to_datetime(["2024-01-05", "2024-01-06", "2024-01-01"]),
            "arrival_date": pd.to_datetime(["2024-01-10", "2024-01-08", pd.NaT]),
            "inbound_date": pd.to_datetime([pd.NaT, pd.NaT, pd.NaT]),
            "event_date": pd.to_datetime([pd.NaT, pd.NaT, "2024-01-15"]),
        }
    )


@pytest.fixture
def timeline_context() -> TimelineContext:
    """테스트용 타임라인 컨텍스트"""
    return TimelineContext(
        centers=["태광KR", "AMZUS"],
        skus=["BA00021"],
        start=pd.Timestamp("2024-01-01"),
        end=pd.Timestamp("2024-01-10"),
        today=pd.Timestamp("2024-01-05"),
        lag_days=5,
        horizon_days=10,
    )


def test_series_index_range():
    """SeriesIndex 날짜 범위 생성"""
    index = SeriesIndex(
        start=pd.Timestamp("2024-01-01"), end=pd.Timestamp("2024-01-05")
    )

    date_range = index.range
    assert len(date_range) == 5
    assert date_range[0] == pd.Timestamp("2024-01-01")
    assert date_range[-1] == pd.Timestamp("2024-01-05")


def test_prepare_snapshot(sample_snapshot):
    """스냅샷 준비 (정규화)"""
    from scm_dashboard_v9.domain.models import SnapshotTable

    result = prepare_snapshot(sample_snapshot)

    assert isinstance(result, SnapshotTable)
    assert not result.data.empty
    assert all(
        col in result.data.columns
        for col in ["date", "center", "resource_code", "stock_qty"]
    )


def test_prepare_moves(sample_moves, timeline_context):
    """이동 원장 준비 (스케줄링)"""
    from scm_dashboard_v9.domain.models import MoveTable

    result = prepare_moves(sample_moves, context=timeline_context, fallback_days=1)

    assert isinstance(result, MoveTable)
    assert not result.data.empty
    assert "pred_inbound_date" in result.data.columns


def test_timeline_builder_empty_snapshot():
    """빈 스냅샷 처리"""
    context = TimelineContext(
        centers=["태광KR"],
        skus=["BA00021"],
        start=pd.Timestamp("2024-01-01"),
        end=pd.Timestamp("2024-01-10"),
        today=pd.Timestamp("2024-01-05"),
    )
    builder = TimelineBuilder(context)

    empty_snapshot = prepare_snapshot(
        pd.DataFrame({"date": [], "center": [], "resource_code": [], "stock_qty": []})
    )

    empty_moves = pd.DataFrame(
        {
            "resource_code": [],
            "qty_ea": [],
            "carrier_mode": [],
            "from_center": [],
            "to_center": [],
            "onboard_date": [],
            "arrival_date": [],
            "inbound_date": [],
            "event_date": [],
        }
    )

    from scm_dashboard_v9.domain.models import MoveTable

    bundle = builder.build(empty_snapshot, MoveTable(empty_moves))

    assert bundle.center_lines.empty
    assert bundle.in_transit_lines.empty
    assert bundle.wip_lines.empty


def test_timeline_builder_basic(sample_snapshot, sample_moves, timeline_context):
    """기본 타임라인 빌드"""
    builder = TimelineBuilder(timeline_context)

    snapshot = prepare_snapshot(sample_snapshot)
    moves = prepare_moves(sample_moves, context=timeline_context)

    bundle = builder.build(snapshot, moves)

    # 센터 라인이 생성되어야 함
    assert not bundle.center_lines.empty
    assert all(
        col in bundle.center_lines.columns
        for col in ["date", "center", "resource_code", "stock_qty"]
    )

    # 타임라인 결합
    combined = bundle.concat()
    assert not combined.empty
    assert "stock_qty" in combined.columns


def test_timeline_builder_wip_handling(sample_moves, timeline_context):
    """WIP 처리 검증"""
    # WIP만 있는 이동
    wip_moves = sample_moves[sample_moves["carrier_mode"] == "WIP"].copy()

    context = TimelineContext(
        centers=["태광KR"],
        skus=["BA00022"],
        start=pd.Timestamp("2024-01-01"),
        end=pd.Timestamp("2024-01-20"),
        today=pd.Timestamp("2024-01-05"),
    )

    builder = TimelineBuilder(context)
    snapshot = prepare_snapshot(
        pd.DataFrame(
            {
                "date": [pd.Timestamp("2024-01-01")],
                "center": ["태광KR"],
                "resource_code": ["BA00022"],
                "stock_qty": [50],
            }
        )
    )
    moves = prepare_moves(wip_moves, context=context)

    bundle = builder.build(snapshot, moves)

    # WIP 라인이 있어야 함
    assert not bundle.wip_lines.empty
    assert "WIP" in bundle.wip_lines["center"].values


def test_timeline_bundle_concat():
    """타임라인 번들 결합"""
    from scm_dashboard_v9.domain.models import TimelineBundle

    center_lines = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-01")],
            "center": ["태광KR"],
            "resource_code": ["BA00021"],
            "stock_qty": [100],
        }
    )

    in_transit = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-02")],
            "center": ["In-Transit"],
            "resource_code": ["BA00021"],
            "stock_qty": [50],
        }
    )

    wip = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-03")],
            "center": ["WIP"],
            "resource_code": ["BA00021"],
            "stock_qty": [30],
        }
    )

    bundle = TimelineBundle(
        center_lines=center_lines, in_transit_lines=in_transit, wip_lines=wip
    )

    combined = bundle.concat()

    assert len(combined) == 3
    assert set(combined["center"].unique()) == {"태광KR", "In-Transit", "WIP"}
    assert all(combined["stock_qty"] >= 0)  # 음수 재고 없음


def test_timeline_no_negative_stock():
    """재고 음수 방지"""
    from scm_dashboard_v9.domain.models import TimelineBundle

    # 음수가 포함된 데이터
    center_lines = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")],
            "center": ["태광KR", "태광KR"],
            "resource_code": ["BA00021", "BA00021"],
            "stock_qty": [100, -50],  # 음수
        }
    )

    bundle = TimelineBundle(
        center_lines=center_lines,
        in_transit_lines=pd.DataFrame(),
        wip_lines=pd.DataFrame(),
    )

    combined = bundle.concat()

    # 음수가 0으로 클리핑되어야 함
    assert all(combined["stock_qty"] >= 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
