"""v5 대비 v8 결과 동등성을 검증하는 골든 마스터 테스트."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
from pandas.testing import assert_frame_equal

from scm_dashboard_v5.pipeline import BuildInputs as V5BuildInputs, build_timeline_bundle as v5_build
from scm_dashboard_v8.application.timeline import BuildInputs as V8BuildInputs, build_timeline_bundle as v8_build


def _normalize_frame(frame: pd.DataFrame, *, sort_keys: Iterable[str]) -> pd.DataFrame:
    """날짜·센터 컬럼을 정규화하고 지정된 키로 정렬한 사본을 반환한다."""

    out = frame.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    for col in ("center", "resource_code"):
        if col in out.columns:
            out[col] = out[col].astype(str)
    if "stock_qty" in out.columns:
        qty = pd.to_numeric(out["stock_qty"], errors="coerce").fillna(0).round()
        out["stock_qty"] = qty.astype(int)
    return out.sort_values(list(sort_keys)).reset_index(drop=True)


def _load_input(name: str) -> pd.DataFrame:
    """golden 입력 CSV를 로드한다."""

    base = Path("tests/golden/inputs")
    return pd.read_csv(base / f"{name}.csv")


def _load_output(name: str) -> pd.DataFrame:
    """golden 출력 CSV를 로드한다."""

    base = Path("tests/golden/outputs")
    return pd.read_csv(base / f"{name}.csv")


def test_timeline_bundle_matches_v5_baseline() -> None:
    """v8 타임라인 빌더가 v5 골든 출력과 완전히 동일한 결과를 생성하는지 검증한다."""

    snapshot = _load_input("snapshot")
    moves = _load_input("moves")

    centers = ["SEOUL", "BUSAN"]
    skus = ["SKU-1"]
    start = pd.Timestamp("2024-01-01")
    end = pd.Timestamp("2024-01-07")
    today = pd.Timestamp("2024-01-02")

    # ✅ baseline이 최신 v5 결과와 일치하는지 먼저 확인한다.
    v5_bundle = v5_build(
        V5BuildInputs(snapshot=snapshot, moves=moves),
        centers=centers,
        skus=skus,
        start=start,
        end=end,
        today=today,
    )

    expected_center = _normalize_frame(_load_output("timeline_center"), sort_keys=("center", "resource_code", "date"))
    expected_in_transit = _normalize_frame(_load_output("timeline_in_transit"), sort_keys=("center", "resource_code", "date"))
    expected_wip = _normalize_frame(_load_output("timeline_wip"), sort_keys=("center", "resource_code", "date"))

    assert_frame_equal(
        expected_center,
        _normalize_frame(v5_bundle.center_lines, sort_keys=("center", "resource_code", "date")),
        check_like=True,
    )
    assert_frame_equal(
        expected_in_transit,
        _normalize_frame(v5_bundle.in_transit_lines, sort_keys=("center", "resource_code", "date")),
        check_like=True,
    )
    assert_frame_equal(
        expected_wip,
        _normalize_frame(v5_bundle.wip_lines, sort_keys=("center", "resource_code", "date")),
        check_like=True,
    )

    # ✅ v8 빌더 결과도 동일한지 검증한다.
    v8_bundle = v8_build(
        V8BuildInputs(snapshot=snapshot, moves=moves),
        centers=centers,
        skus=skus,
        start=start,
        end=end,
        today=today,
    )

    assert_frame_equal(
        expected_center,
        _normalize_frame(v8_bundle.center_lines, sort_keys=("center", "resource_code", "date")),
        check_like=True,
    )
    assert_frame_equal(
        expected_in_transit,
        _normalize_frame(v8_bundle.in_transit_lines, sort_keys=("center", "resource_code", "date")),
        check_like=True,
    )
    assert_frame_equal(
        expected_wip,
        _normalize_frame(v8_bundle.wip_lines, sort_keys=("center", "resource_code", "date")),
        check_like=True,
    )

    # ✅ 전체 병합 결과도 비교해 추가적인 회귀를 방지한다.
    v8_concat = v8_bundle.concat()
    filtered = v8_concat[~v8_concat["center"].isin(["In-Transit", "WIP"])]
    assert_frame_equal(
        expected_center,
        _normalize_frame(filtered, sort_keys=("center", "resource_code", "date")),
        check_like=True,
    )
