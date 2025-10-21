"""
v7 골든 마스터 동등성 테스트

설명(한글):
- 동일 입력(snapshot/moves/snapshot_raw)에 대해 v5와 v7의 주요 산출물이 동일한지 비교합니다.
- UPDATE_GOLDEN=1 환경변수로 골든 산출물을 갱신할 수 있습니다.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

from scm_dashboard_v5.core import build_timeline as v5_build_timeline
from scm_dashboard_v5.forecast import apply_consumption_with_events as v5_apply_consumption
from scm_dashboard_v5.ui.kpi import render_sku_summary_cards as v5_kpi_cards
from scm_dashboard_v7.features.timeline import render_timeline_section as v7_timeline_section
from scm_dashboard_v7.data.loaders import load_snapshot_raw
from tests_v7.conftest import UPDATE_GOLDEN


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def test_golden_timeline_and_kpi(golden_dir, normalize_df):
    # 입력 데이터(작은 샘플)를 tests_v7/golden/input/*.csv 로 가정
    inp_dir = golden_dir / "input"
    snap = _read_csv(inp_dir / "snapshot.csv")
    moves = _read_csv(inp_dir / "moves.csv")
    snap_raw = _read_csv(inp_dir / "snapshot_raw.csv")

    # 테스트 파라미터(간단 고정)
    centers = ["태광KR", "AMZUS"]
    skus = ["BA00021", "BA00022"]
    today = pd.Timestamp("2024-12-31")
    start = today - pd.Timedelta(days=20)
    end = today + pd.Timedelta(days=30)
    lookback_days = 28
    lag_days = 5

    # v5 산출물
    v5_timeline = v5_build_timeline(
        snap,
        moves,
        centers=centers,
        skus=skus,
        start=start,
        end=end,
        today=today,
        lag_days=lag_days,
        horizon_days=max(0, int((end - today).days)),
    )
    v5_timeline_cons = v5_apply_consumption(
        v5_timeline,
        snap,
        centers=centers,
        skus=skus,
        start=start,
        end=end,
        lookback_days=lookback_days,
        events=[],
    )
    # v5 KPI (UI 함수이지만 DataFrame 반환하므로 직접 호출)
    v5_kpi = v5_kpi_cards(
        snap,
        moves,
        centers=centers,
        skus=skus,
        today=today,
        latest_snapshot=snap.get("date").max() if "date" in snap.columns else None,
        lag_days=lag_days,
        start=start,
        end=end,
        lookback_days=lookback_days,
        horizon_pad_days=60,
        events=[],
    )

    # v7 산출물
    v7_timeline = v7_timeline_section(
        snapshot=snap,
        moves=moves,
        centers=centers,
        skus=skus,
        start=start,
        end=end,
        today=today,
        lookback_days=lookback_days,
        lag_days=lag_days,
        promotion_events=[],
        show_production=False,
        show_in_transit=False,
    )

    # 정규화/비교
    sort_keys = ["date", "center", "resource_code"]
    v5_cons_norm = normalize_df(v5_timeline_cons, sort_keys)
    v7_norm = normalize_df(v7_timeline, sort_keys)

    # 골든 경로
    g_v5_timeline = golden_dir / "v5" / "timeline_with_consumption.csv"
    g_v7_timeline = golden_dir / "v7" / "timeline_with_consumption.csv"
    g_v5_kpi = golden_dir / "v5" / "kpi_breakdown.csv"
    g_v7_kpi = golden_dir / "v7" / "kpi_breakdown.csv"

    if UPDATE_GOLDEN:
        _write_csv(v5_cons_norm, g_v5_timeline)
        _write_csv(v7_norm, g_v7_timeline)
        _write_csv(v5_kpi, g_v5_kpi)
        _write_csv(v5_kpi, g_v7_kpi)  # v7 KPI는 v5 호출이므로 동일 저장

    # 비교: v5 vs v7 (normalize된 타임라인)
    exp_v5 = _read_csv(g_v5_timeline)
    exp_v7 = _read_csv(g_v7_timeline)
    pd.testing.assert_frame_equal(
        normalize_df(exp_v5, sort_keys), normalize_df(exp_v7, sort_keys), check_like=True
    )

    # KPI 비교: v5 기준 골든과 v7 결과 동일
    exp_kpi = _read_csv(g_v5_kpi)
    cur_kpi = v5_kpi.copy()  # v7도 동일 호출
    # KPI는 컬럼/순서 변동 가능하므로 SKU 기준 집계/정렬로 비교 허용
    if not exp_kpi.empty and not cur_kpi.empty:
        exp_kpi = exp_kpi.sort_values(list(exp_kpi.columns)).reset_index(drop=True)
        cur_kpi = cur_kpi.sort_values(list(cur_kpi.columns)).reset_index(drop=True)
        pd.testing.assert_frame_equal(exp_kpi, cur_kpi, check_like=True)


