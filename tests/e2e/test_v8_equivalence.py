"""v5 대비 v8 동등성을 자동으로 검증하는 엔드투엔드 테스트."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
from pandas.testing import assert_frame_equal

from scm_dashboard_v5.analytics import kpi_breakdown_per_sku as v5_kpi_breakdown
from scm_dashboard_v5.core import build_timeline as v5_build_timeline
from scm_dashboard_v5.forecast import (
    apply_consumption_with_events as v5_apply_consumption,
    build_amazon_forecast_context as v5_build_amazon_context,
)
from scm_dashboard_v5.pipeline import (
    BuildInputs as V5BuildInputs,
    build_timeline_bundle as v5_build_bundle,
)
from scm_dashboard_v8.analytics import kpi_breakdown_per_sku as v8_kpi_breakdown
from scm_dashboard_v8.application.timeline import (
    BuildInputs as V8BuildInputs,
    build_timeline_bundle as v8_build_bundle,
)
from scm_dashboard_v8.core import build_timeline as v8_build_timeline
from scm_dashboard_v8.forecast import (
    apply_consumption_with_events as v8_apply_consumption,
    build_amazon_forecast_context as v8_build_amazon_context,
)


def _normalize_frame(
    frame: pd.DataFrame,
    *,
    sort_keys: Iterable[str],
    int_columns: Iterable[str] = (),
    numeric_columns: Iterable[str] = (),
) -> pd.DataFrame:
    """데이터프레임을 비교 가능한 형태로 정규화한다."""

    out = frame.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    for col in ("center", "resource_code"):
        if col in out.columns:
            out[col] = out[col].astype(str)
    for col in numeric_columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    for col in int_columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).round().astype(int)
    if sort_keys:
        out = out.sort_values(list(sort_keys))
    elif not out.empty:
        out = out.sort_values(list(out.columns))
    return out.reset_index(drop=True)


def _load_input(name: str) -> pd.DataFrame:
    """골든 입력 CSV를 로드한다."""

    base = Path("tests/golden/inputs")
    return pd.read_csv(base / f"{name}.csv")


def _load_output(name: str) -> pd.DataFrame:
    """골든 출력 CSV를 로드한다."""

    base = Path("tests/golden/outputs")
    return pd.read_csv(base / f"{name}.csv")


def _prepare_moves_for_kpi(moves: pd.DataFrame) -> pd.DataFrame:
    """KPI/아마존 컨텍스트 계산에서 필요한 날짜 컬럼을 미리 datetime으로 변환한다."""

    out = moves.copy()
    for col in ("event_date", "onboard_date", "arrival_date", "inbound_date"):
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")
    return out


def _latest_snapshot_date(snapshot: pd.DataFrame) -> pd.Timestamp:
    """스냅샷 데이터에서 가장 최근 날짜를 추출한다."""

    return (
        pd.to_datetime(snapshot.get("snapshot_date"), errors="coerce")
        .dropna()
        .max()
        .normalize()
    )


def _assert_timeline(
    expected_name: str,
    actual_frame: pd.DataFrame,
    *,
    sort_keys: Sequence[str],
) -> None:
    """타임라인 계열 결과를 골든 데이터와 비교한다."""

    expected = _normalize_frame(
        _load_output(expected_name),
        sort_keys=sort_keys,
        int_columns=("stock_qty",),
    )
    assert_frame_equal(
        expected,
        _normalize_frame(actual_frame, sort_keys=sort_keys, int_columns=("stock_qty",)),
        check_like=True,
    )


def test_v8_pipeline_matches_v5_baseline() -> None:
    """v8 모듈이 v5와 완전히 동일한 타임라인/예측/KPI 출력을 만드는지 검증한다."""

    snapshot = _load_input("snapshot")
    moves = _prepare_moves_for_kpi(_load_input("moves"))
    snapshot_raw = _load_input("snapshot_raw")

    centers = ["SEOUL", "BUSAN"]
    skus = ["SKU-1"]
    start = pd.Timestamp("2024-01-01")
    end = pd.Timestamp("2024-01-07")
    today = pd.Timestamp("2024-01-02")
    lag_days = 5
    lookback_days = 28

    latest_snapshot = _latest_snapshot_date(snapshot)

    # ✅ baseline: v5 결과가 골든 파일과 일치하는지 확인한다.
    v5_bundle = v5_build_bundle(
        V5BuildInputs(snapshot=snapshot, moves=moves),
        centers=centers,
        skus=skus,
        start=start,
        end=end,
        today=today,
        lag_days=lag_days,
    )
    _assert_timeline(
        "timeline_center",
        v5_bundle.center_lines,
        sort_keys=("center", "resource_code", "date"),
    )
    _assert_timeline(
        "timeline_in_transit",
        v5_bundle.in_transit_lines,
        sort_keys=("center", "resource_code", "date"),
    )
    _assert_timeline(
        "timeline_wip",
        v5_bundle.wip_lines,
        sort_keys=("center", "resource_code", "date"),
    )

    v5_actual = v5_build_timeline(
        snapshot,
        moves,
        centers=centers,
        skus=skus,
        start=start,
        end=end,
        today=today,
        lag_days=lag_days,
    )
    _assert_timeline(
        "timeline_actual",
        v5_actual,
        sort_keys=("center", "resource_code", "date"),
    )

    v5_consumption = v5_apply_consumption(
        v5_actual,
        snapshot,
        centers=centers,
        skus=skus,
        start=start,
        end=end,
        lookback_days=lookback_days,
        events=[],
    )
    _assert_timeline(
        "timeline_with_consumption",
        v5_consumption,
        sort_keys=("center", "resource_code", "date"),
    )

    v5_kpi = v5_kpi_breakdown(
        snapshot,
        moves,
        centers=centers,
        skus=skus,
        today=today,
        snapshot_date_col="snapshot_date",
        latest_snapshot=latest_snapshot,
        lag_days=lag_days,
    )
    expected_kpi = _normalize_frame(
        _load_output("kpi_breakdown"),
        sort_keys=(),
        int_columns=("current", "in_transit", "wip"),
    )
    assert_frame_equal(
        expected_kpi,
        _normalize_frame(
            v5_kpi,
            sort_keys=(),
            int_columns=("current", "in_transit", "wip"),
        ),
        check_like=True,
    )

    v5_amazon = v5_build_amazon_context(
        snap_long=snapshot,
        moves=moves,
        snapshot_raw=snapshot_raw,
        centers=centers,
        skus=skus,
        start=start,
        end=end,
        today=today,
        lookback_days=lookback_days,
        promotion_events=[],
        use_consumption_forecast=True,
    )
    _assert_timeline(
        "amazon_inventory_actual",
        v5_amazon.inv_actual,
        sort_keys=("center", "resource_code", "date"),
    )
    _assert_timeline(
        "amazon_inventory_forecast",
        v5_amazon.inv_forecast,
        sort_keys=("center", "resource_code", "date"),
    )
    expected_sales_hist = _normalize_frame(
        _load_output("amazon_sales_history"),
        sort_keys=("center", "resource_code", "date"),
        int_columns=("sales_ea",),
    )
    assert_frame_equal(
        expected_sales_hist,
        _normalize_frame(
            v5_amazon.sales_hist,
            sort_keys=("center", "resource_code", "date"),
            int_columns=("sales_ea",),
        ),
        check_like=True,
    )
    expected_sales_ma7 = _normalize_frame(
        _load_output("amazon_sales_ma7"),
        sort_keys=("center", "resource_code", "date"),
        int_columns=("sales_ea",),
    )
    assert_frame_equal(
        expected_sales_ma7,
        _normalize_frame(
            v5_amazon.sales_ma7,
            sort_keys=("center", "resource_code", "date"),
            int_columns=("sales_ea",),
        ),
        check_like=True,
    )
    expected_sales_forecast = _normalize_frame(
        _load_output("amazon_sales_forecast"),
        sort_keys=("center", "resource_code", "date"),
        int_columns=("sales_ea",),
    )
    assert_frame_equal(
        expected_sales_forecast,
        _normalize_frame(
            v5_amazon.sales_forecast,
            sort_keys=("center", "resource_code", "date"),
            int_columns=("sales_ea",),
        ),
        check_like=True,
    )

    # ✅ v8 결과도 동일한 골든 파일과 일치해야 한다.
    v8_bundle = v8_build_bundle(
        V8BuildInputs(snapshot=snapshot, moves=moves),
        centers=centers,
        skus=skus,
        start=start,
        end=end,
        today=today,
        lag_days=lag_days,
    )
    _assert_timeline(
        "timeline_center",
        v8_bundle.center_lines,
        sort_keys=("center", "resource_code", "date"),
    )
    _assert_timeline(
        "timeline_in_transit",
        v8_bundle.in_transit_lines,
        sort_keys=("center", "resource_code", "date"),
    )
    _assert_timeline(
        "timeline_wip",
        v8_bundle.wip_lines,
        sort_keys=("center", "resource_code", "date"),
    )

    v8_actual = v8_build_timeline(
        snapshot,
        moves,
        centers=centers,
        skus=skus,
        start=start,
        end=end,
        today=today,
        lag_days=lag_days,
    )
    _assert_timeline(
        "timeline_actual",
        v8_actual,
        sort_keys=("center", "resource_code", "date"),
    )

    v8_consumption = v8_apply_consumption(
        v8_actual,
        snapshot,
        centers=centers,
        skus=skus,
        start=start,
        end=end,
        lookback_days=lookback_days,
        events=[],
    )
    _assert_timeline(
        "timeline_with_consumption",
        v8_consumption,
        sort_keys=("center", "resource_code", "date"),
    )

    v8_kpi = v8_kpi_breakdown(
        snapshot,
        moves,
        centers=centers,
        skus=skus,
        today=today,
        snapshot_date_col="snapshot_date",
        latest_snapshot=latest_snapshot,
        lag_days=lag_days,
    )
    assert_frame_equal(
        expected_kpi,
        _normalize_frame(
            v8_kpi,
            sort_keys=(),
            int_columns=("current", "in_transit", "wip"),
        ),
        check_like=True,
    )

    v8_amazon = v8_build_amazon_context(
        snap_long=snapshot,
        moves=moves,
        snapshot_raw=snapshot_raw,
        centers=centers,
        skus=skus,
        start=start,
        end=end,
        today=today,
        lookback_days=lookback_days,
        promotion_events=[],
        use_consumption_forecast=True,
    )
    _assert_timeline(
        "amazon_inventory_actual",
        v8_amazon.inv_actual,
        sort_keys=("center", "resource_code", "date"),
    )
    _assert_timeline(
        "amazon_inventory_forecast",
        v8_amazon.inv_forecast,
        sort_keys=("center", "resource_code", "date"),
    )
    assert_frame_equal(
        expected_sales_hist,
        _normalize_frame(
            v8_amazon.sales_hist,
            sort_keys=("center", "resource_code", "date"),
            int_columns=("sales_ea",),
        ),
        check_like=True,
    )
    assert_frame_equal(
        expected_sales_ma7,
        _normalize_frame(
            v8_amazon.sales_ma7,
            sort_keys=("center", "resource_code", "date"),
            int_columns=("sales_ea",),
        ),
        check_like=True,
    )
    assert_frame_equal(
        expected_sales_forecast,
        _normalize_frame(
            v8_amazon.sales_forecast,
            sort_keys=("center", "resource_code", "date"),
            int_columns=("sales_ea",),
        ),
        check_like=True,
    )

    # ✅ 번들 병합 결과 역시 골든 데이터와 동일해야 한다.
    merged = v8_bundle.concat()
    _assert_timeline(
        "timeline_actual",
        merged,
        sort_keys=("center", "resource_code", "date"),
    )
