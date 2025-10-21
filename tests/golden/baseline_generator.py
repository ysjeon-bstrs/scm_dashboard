"""
V5 파이프라인 베이스라인 생성기

V5 코드를 사용하여 골든 마스터 베이스라인을 생성합니다.
한 번만 실행하여 baseline/ 디렉토리에 저장합니다.

Usage:
    python tests/golden/baseline_generator.py
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

# V5 모듈 import (기존 v5에서 사용하는 것들)
from scm_dashboard_v5.core import build_timeline as build_core_timeline_v5
from scm_dashboard_v5.forecast import apply_consumption_with_events as apply_consumption_v5


def load_test_fixtures() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    테스트 픽스처 데이터를 로드합니다.

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


def load_test_scenarios() -> list[dict[str, Any]]:
    """
    테스트 시나리오를 로드합니다.

    Returns:
        시나리오 딕셔너리 리스트
    """
    fixtures_dir = Path(__file__).parent.parent / "fixtures"

    with open(fixtures_dir / "test_scenarios.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    return data["scenarios"]


def generate_v5_baseline(
    scenario_id: str,
    snapshot: pd.DataFrame,
    moves: pd.DataFrame,
    params: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    """
    V5 파이프라인을 실행하여 베이스라인 산출물을 생성합니다.

    Args:
        scenario_id: 시나리오 ID
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
    # 매개변수 파싱
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
    # V5 파이프라인 실행 (기존 로직 그대로)
    # ========================================

    # 1. 타임라인 빌드 (실제)
    timeline_actual = build_core_timeline_v5(
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

    timeline_forecast = apply_consumption_v5(
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


def save_baseline(
    scenario_id: str,
    outputs: dict[str, pd.DataFrame],
) -> None:
    """
    베이스라인 산출물을 CSV로 저장합니다.

    Args:
        scenario_id: 시나리오 ID
        outputs: 산출물 딕셔너리
    """
    baseline_dir = Path(__file__).parent / "baselines"
    baseline_dir.mkdir(parents=True, exist_ok=True)

    for key, df in outputs.items():
        filename = f"{scenario_id}_{key}.csv"
        filepath = baseline_dir / filename

        # 날짜 컬럼을 문자열로 변환 (재현성 보장)
        df_copy = df.copy()
        for col in df_copy.columns:
            if pd.api.types.is_datetime64_any_dtype(df_copy[col]):
                df_copy[col] = df_copy[col].dt.strftime("%Y-%m-%d %H:%M:%S")

        df_copy.to_csv(filepath, index=False, encoding="utf-8-sig")
        print(f"✅ Saved baseline: {filepath}")


def main() -> None:
    """
    모든 시나리오에 대해 V5 베이스라인을 생성합니다.

    Usage:
        python tests/golden/baseline_generator.py
    """
    print("=" * 60)
    print("V5 골든 마스터 베이스라인 생성")
    print("=" * 60)

    # 테스트 데이터 로드
    snapshot, moves, snapshot_raw = load_test_fixtures()
    print(f"✅ Loaded fixtures: {len(snapshot)} snapshot rows, {len(moves)} move rows")

    # 시나리오 로드
    scenarios = load_test_scenarios()
    print(f"✅ Loaded {len(scenarios)} test scenarios")

    # 각 시나리오 실행
    for scenario in scenarios:
        scenario_id = scenario["id"]
        description = scenario["description"]
        params = scenario["params"]

        print(f"\n📋 Scenario: {scenario_id}")
        print(f"   {description}")

        try:
            # V5 베이스라인 생성
            outputs = generate_v5_baseline(scenario_id, snapshot, moves, params)

            # 저장
            save_baseline(scenario_id, outputs)
        except Exception as e:
            print(f"❌ Error in scenario {scenario_id}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("✅ All baselines generated successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
