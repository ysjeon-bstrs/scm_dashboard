"""
V5 ↔ V9 동등성 자동 테스트 (Golden Master Testing)

V5 베이스라인과 V9 산출물을 자동으로 비교하여 동일성을 검증합니다.

Usage:
    pytest tests/golden/test_equivalence.py -v
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

# V9 모듈 import (리팩터링 후)
from scm_dashboard_v9.core import build_timeline as build_core_timeline_v9
from scm_dashboard_v9.forecast import apply_consumption_with_events as apply_consumption_v9


@pytest.fixture
def test_fixtures() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    테스트 픽스처 데이터를 로드하는 fixture.

    Returns:
        (snapshot, moves, snapshot_raw) 튜플
    """
    fixtures_dir = Path(__file__).parent.parent / "fixtures"

    snapshot = pd.read_csv(fixtures_dir / "sample_snapshot.csv", parse_dates=["date"])
    moves = pd.read_csv(
        fixtures_dir / "sample_moves.csv",
        parse_dates=["onboard_date", "arrival_date", "inbound_date", "event_date"]
    )
    snapshot_raw = pd.read_csv(
        fixtures_dir / "sample_snapshot_raw.csv",
        parse_dates=["snapshot_date"]
    )

    return snapshot, moves, snapshot_raw


@pytest.fixture
def test_scenarios() -> list[dict[str, Any]]:
    """
    테스트 시나리오를 로드하는 fixture.

    Returns:
        시나리오 딕셔너리 리스트
    """
    fixtures_dir = Path(__file__).parent.parent / "fixtures"

    with open(fixtures_dir / "test_scenarios.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    return data["scenarios"]


def load_v5_baseline(scenario_id: str, output_key: str) -> pd.DataFrame:
    """
    V5 베이스라인을 로드합니다.

    Args:
        scenario_id: 시나리오 ID
        output_key: 산출물 키 (예: "timeline_actual", "timeline_forecast")

    Returns:
        V5 베이스라인 데이터프레임
    """
    baseline_dir = Path(__file__).parent / "baselines"
    filename = f"{scenario_id}_{output_key}.csv"
    filepath = baseline_dir / filename

    if not filepath.exists():
        pytest.skip(f"Baseline not found: {filepath}. Run baseline_generator.py first.")

    df = pd.read_csv(filepath, parse_dates=["date"])
    return df


def generate_v9_output(
    snapshot: pd.DataFrame,
    moves: pd.DataFrame,
    params: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    """
    V9 파이프라인을 실행하여 산출물을 생성합니다.

    Args:
        snapshot: 스냅샷 데이터프레임
        moves: 이동 원장 데이터프레임
        params: 시나리오 매개변수

    Returns:
        {
            "timeline_actual": 타임라인 (실제),
            "timeline_forecast": 타임라인 (예측 적용),
        }
    """
    # ========================================
    # 매개변수 파싱 (V5와 동일)
    # ========================================
    centers = params["centers"]
    skus = params["skus"]
    start = pd.to_datetime(params["start"]).normalize()
    end = pd.to_datetime(params["end"]).normalize()
    lag_days = int(params["lag_days"])
    lookback_days = int(params["lookback_days"])

    # 이벤트 파싱
    events = []
    for event_dict in params.get("events", []):
        events.append({
            "start": pd.to_datetime(event_dict["start"]),
            "end": pd.to_datetime(event_dict["end"]),
            "uplift": float(event_dict["uplift"]),
        })

    today = pd.Timestamp.today().normalize()

    # ========================================
    # V9 파이프라인 실행 (리팩터링 후)
    # ========================================

    # 1. 타임라인 빌드 (실제)
    timeline_actual = build_core_timeline_v9(
        snapshot,
        moves,
        centers=centers,
        skus=skus,
        start=start,
        end=end,
        today=today,
        lag_days=lag_days,
        horizon_days=60,  # 고정값
    )

    # 2. 소비 예측 적용
    snap_dates = snapshot["date"].dropna()
    latest_dt = snap_dates.max() if not snap_dates.empty else pd.NaT

    cons_start = None
    if pd.notna(latest_dt):
        cons_start = (pd.Timestamp(latest_dt).normalize() + pd.Timedelta(days=1)).normalize()

    timeline_forecast = apply_consumption_v9(
        timeline_actual,
        snapshot,
        centers=centers,
        skus=skus,
        start=start,
        end=end,
        lookback_days=lookback_days,
        events=events,
        cons_start=cons_start,
    )

    if timeline_forecast is None or timeline_forecast.empty:
        timeline_forecast = timeline_actual.copy()

    return {
        "timeline_actual": timeline_actual,
        "timeline_forecast": timeline_forecast,
    }


def assert_dataframes_equal(
    v5_df: pd.DataFrame,
    v9_df: pd.DataFrame,
    scenario_id: str,
    output_key: str,
) -> None:
    """
    V5와 V9 데이터프레임의 완전 동등성을 검증합니다.

    Args:
        v5_df: V5 베이스라인 데이터프레임
        v9_df: V9 산출 데이터프레임
        scenario_id: 시나리오 ID (에러 메시지용)
        output_key: 산출물 키 (에러 메시지용)

    Raises:
        AssertionError: 데이터프레임이 동일하지 않을 경우
    """
    # ========================================
    # 컬럼명 정렬 (순서 무관하게 비교)
    # ========================================
    v5_df = v5_df.sort_index(axis=1)
    v9_df = v9_df.sort_index(axis=1)

    # ========================================
    # 인덱스 초기화 (인덱스 순서 무관하게 비교)
    # ========================================
    v5_df = v5_df.reset_index(drop=True)
    v9_df = v9_df.reset_index(drop=True)

    # ========================================
    # 데이터프레임 동등성 검증
    # ========================================
    try:
        pd.testing.assert_frame_equal(
            v5_df,
            v9_df,
            check_dtype=False,  # 타입은 유연하게 (int64 vs int32 등)
            check_exact=False,  # 부동소수점 비교는 근사값 허용
            rtol=1e-5,          # 상대 오차 허용 범위
            atol=1e-8,          # 절대 오차 허용 범위
        )
    except AssertionError as e:
        # 상세한 에러 메시지 출력
        print(f"\n{'=' * 60}")
        print(f"❌ MISMATCH: {scenario_id} / {output_key}")
        print(f"{'=' * 60}")
        print(f"\nV5 shape: {v5_df.shape}")
        print(f"V9 shape: {v9_df.shape}")
        print(f"\nV5 columns: {v5_df.columns.tolist()}")
        print(f"V9 columns: {v9_df.columns.tolist()}")
        print(f"\nV5 sample:\n{v5_df.head()}")
        print(f"\nV9 sample:\n{v9_df.head()}")
        print(f"\nDifference:\n{e}")
        raise


def _load_scenarios_for_parametrize():
    """pytest.mark.parametrize를 위한 시나리오 로드 헬퍼."""
    fixtures_dir = Path(__file__).parent.parent / "fixtures"
    scenarios_path = fixtures_dir / "test_scenarios.json"

    if not scenarios_path.exists():
        return []

    with open(scenarios_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data["scenarios"]


@pytest.mark.parametrize("scenario", [
    pytest.param(
        scenario,
        id=scenario["id"]
    )
    for scenario in _load_scenarios_for_parametrize()
])
def test_v9_equals_v5_timeline_actual(
    scenario: dict[str, Any],
    test_fixtures: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    """
    V9 타임라인(실제)이 V5 베이스라인과 완전히 동일한지 테스트합니다.

    Args:
        scenario: 테스트 시나리오
        test_fixtures: (snapshot, moves, snapshot_raw) 픽스처
    """
    scenario_id = scenario["id"]
    params = scenario["params"]
    snapshot, moves, _ = test_fixtures

    # V5 베이스라인 로드
    v5_baseline = load_v5_baseline(scenario_id, "timeline_actual")

    # V9 산출물 생성
    v9_outputs = generate_v9_output(snapshot, moves, params)
    v9_timeline = v9_outputs["timeline_actual"]

    # 동등성 검증
    assert_dataframes_equal(v5_baseline, v9_timeline, scenario_id, "timeline_actual")


@pytest.mark.parametrize("scenario", [
    pytest.param(
        scenario,
        id=scenario["id"]
    )
    for scenario in _load_scenarios_for_parametrize()
])
def test_v9_equals_v5_timeline_forecast(
    scenario: dict[str, Any],
    test_fixtures: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    """
    V9 타임라인(예측)이 V5 베이스라인과 완전히 동일한지 테스트합니다.

    Args:
        scenario: 테스트 시나리오
        test_fixtures: (snapshot, moves, snapshot_raw) 픽스처
    """
    scenario_id = scenario["id"]
    params = scenario["params"]
    snapshot, moves, _ = test_fixtures

    # V5 베이스라인 로드
    v5_baseline = load_v5_baseline(scenario_id, "timeline_forecast")

    # V9 산출물 생성
    v9_outputs = generate_v9_output(snapshot, moves, params)
    v9_forecast = v9_outputs["timeline_forecast"]

    # 동등성 검증
    assert_dataframes_equal(v5_baseline, v9_forecast, scenario_id, "timeline_forecast")
