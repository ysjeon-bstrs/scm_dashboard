"""
v9 도메인 모델 및 정규화 테스트

데이터 모델, 정규화, 검증 로직을 테스트합니다.
"""
from __future__ import annotations

import pandas as pd
import pytest

from scm_dashboard_v9.domain.exceptions import ValidationError
from scm_dashboard_v9.domain.filters import (
    calculate_date_bounds,
    extract_center_and_sku_options,
    filter_by_centers,
    filter_by_skus,
    safe_to_datetime,
)
from scm_dashboard_v9.domain.models import MoveTable, SnapshotTable, TimelineBundle
from scm_dashboard_v9.domain.normalization import normalize_moves, normalize_snapshot
from scm_dashboard_v9.domain.validation import validate_timeline_inputs


# ============================================================
# 정규화 테스트
# ============================================================

def test_normalize_snapshot_basic():
    """스냅샷 정규화 - 기본 케이스"""
    raw = pd.DataFrame({
        "date": ["2024-01-01", "2024-01-02"],
        "center": ["태광KR", "AMZUS"],
        "resource_code": ["BA00021", "BA00022"],
        "stock_qty": [100, 200],
    })
    
    result = normalize_snapshot(raw)
    
    assert not result.empty
    assert result["date"].dtype == "datetime64[ns]"
    assert result["stock_qty"].dtype == "float64"
    assert len(result) == 2


def test_normalize_snapshot_alternative_columns():
    """스냅샷 정규화 - 대체 컬럼명"""
    raw = pd.DataFrame({
        "snapshot_date": ["2024-01-01"],
        "센터": ["태광KR"],
        "SKU": ["BA00021"],
        "수량": [100],
    })
    
    result = normalize_snapshot(raw)
    
    assert not result.empty
    assert "date" in result.columns
    assert "center" in result.columns
    assert "resource_code" in result.columns
    assert "stock_qty" in result.columns


def test_normalize_snapshot_with_sales():
    """스냅샷 정규화 - 판매량 포함"""
    raw = pd.DataFrame({
        "date": ["2024-01-01"],
        "center": ["AMZUS"],
        "resource_code": ["BA00021"],
        "stock_qty": [100],
        "sales_qty": [10],
    })
    
    result = normalize_snapshot(raw)
    
    assert "sales_qty" in result.columns
    assert result.iloc[0]["sales_qty"] == 10


def test_normalize_moves_basic():
    """이동 원장 정규화 - 기본 케이스"""
    raw = pd.DataFrame({
        "resource_code": ["BA00021"],
        "qty_ea": [100],
        "carrier_mode": ["sea"],
        "from_center": ["상해CN"],
        "to_center": ["태광KR"],
        "onboard_date": ["2024-01-01"],
        "arrival_date": ["2024-01-10"],
    })
    
    result = normalize_moves(raw)
    
    assert not result.empty
    assert result.iloc[0]["carrier_mode"] == "SEA"  # 대문자 변환
    assert result.iloc[0]["onboard_date"] == pd.Timestamp("2024-01-01")


def test_normalize_moves_alternative_columns():
    """이동 원장 정규화 - 대체 컬럼명"""
    raw = pd.DataFrame({
        "상품코드": ["BA00021"],
        "수량": ["100"],
        "운송방법": ["항공"],
        "출발창고": ["상해CN"],
        "도착창고": ["태광KR"],
        "출발일": ["2024-01-01"],
        "도착일": ["2024-01-10"],
    })
    
    result = normalize_moves(raw)
    
    assert not result.empty
    assert "resource_code" in result.columns
    assert "qty_ea" in result.columns
    assert result.iloc[0]["qty_ea"] == 100  # 문자열 → 숫자 변환


def test_normalize_moves_qty_with_comma():
    """이동 원장 정규화 - 천단위 콤마 처리"""
    raw = pd.DataFrame({
        "resource_code": ["BA00021"],
        "qty_ea": ["1,234"],
        "carrier_mode": ["SEA"],
        "from_center": ["상해CN"],
        "to_center": ["태광KR"],
        "onboard_date": ["2024-01-01"],
    })
    
    result = normalize_moves(raw)
    
    assert result.iloc[0]["qty_ea"] == 1234


# ============================================================
# 도메인 모델 테스트
# ============================================================

def test_snapshot_table_from_dataframe():
    """SnapshotTable 생성"""
    raw = pd.DataFrame({
        "date": ["2024-01-01"],
        "center": ["태광KR"],
        "resource_code": ["BA00021"],
        "stock_qty": [100],
    })
    
    snapshot = SnapshotTable.from_dataframe(raw)
    
    assert not snapshot.data.empty
    assert snapshot.data.iloc[0]["stock_qty"] == 100


def test_snapshot_table_filter():
    """SnapshotTable 필터링"""
    raw = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        "center": ["태광KR", "AMZUS", "태광KR"],
        "resource_code": ["BA00021", "BA00021", "BA00022"],
        "stock_qty": [100, 200, 150],
    })
    
    snapshot = SnapshotTable.from_dataframe(raw)
    filtered = snapshot.filter(
        centers=["태광KR"],
        skus=["BA00021"],
        start=pd.Timestamp("2024-01-01"),
        end=pd.Timestamp("2024-01-02")
    )
    
    assert len(filtered) == 1
    assert filtered.iloc[0]["center"] == "태광KR"
    assert filtered.iloc[0]["resource_code"] == "BA00021"


def test_move_table_slice_by_center():
    """MoveTable 센터별 슬라이싱"""
    moves = pd.DataFrame({
        "resource_code": ["BA00021", "BA00022", "BA00023"],
        "to_center": ["태광KR", "AMZUS", "태광KR"],
        "carrier_mode": ["SEA", "AIR", "WIP"],
        "qty_ea": [100, 200, 50],
    })
    
    move_table = MoveTable(moves)
    
    # 태광KR 향하는 이동 (WIP 포함)
    result = move_table.slice_by_center(center="태광KR", include_wip=True)
    assert len(result) == 2
    
    # 태광KR 향하는 이동 (WIP 제외)
    result_no_wip = move_table.slice_by_center(center="태광KR", include_wip=False)
    assert len(result_no_wip) == 1
    assert result_no_wip.iloc[0]["carrier_mode"] == "SEA"


def test_timeline_bundle_empty():
    """빈 TimelineBundle"""
    bundle = TimelineBundle(
        center_lines=pd.DataFrame(),
        in_transit_lines=pd.DataFrame(),
        wip_lines=pd.DataFrame()
    )
    
    combined = bundle.concat()
    
    assert combined.empty
    assert all(col in combined.columns for col in ["date", "center", "resource_code", "stock_qty"])


# ============================================================
# 검증 테스트
# ============================================================

def test_validate_timeline_inputs_valid():
    """유효한 입력 검증"""
    snapshot = pd.DataFrame({
        "date": ["2024-01-01"],
        "center": ["태광KR"],
        "resource_code": ["BA00021"],
        "stock_qty": [100],
    })
    
    moves = pd.DataFrame({
        "from_center": ["상해CN"],
        "to_center": ["태광KR"],
        "resource_code": ["BA00021"],
        "qty_ea": [50],
    })
    
    start = pd.Timestamp("2024-01-01")
    end = pd.Timestamp("2024-01-10")
    
    # 예외 발생하지 않아야 함
    validate_timeline_inputs(snapshot, moves, start, end)


def test_validate_timeline_inputs_missing_snapshot_columns():
    """스냅샷 필수 컬럼 누락"""
    snapshot = pd.DataFrame({
        "date": ["2024-01-01"],
        # center 컬럼 누락
        "resource_code": ["BA00021"],
        "stock_qty": [100],
    })
    
    moves = pd.DataFrame({
        "from_center": ["상해CN"],
        "to_center": ["태광KR"],
        "resource_code": ["BA00021"],
    })
    
    with pytest.raises(ValidationError, match="필요한 컬럼이 없습니다"):
        validate_timeline_inputs(
            snapshot, moves,
            pd.Timestamp("2024-01-01"),
            pd.Timestamp("2024-01-10")
        )


def test_validate_timeline_inputs_invalid_date_range():
    """잘못된 날짜 범위"""
    snapshot = pd.DataFrame({
        "date": ["2024-01-01"],
        "center": ["태광KR"],
        "resource_code": ["BA00021"],
        "stock_qty": [100],
    })
    
    moves = pd.DataFrame({
        "from_center": ["상해CN"],
        "to_center": ["태광KR"],
        "resource_code": ["BA00021"],
    })
    
    # end < start
    with pytest.raises(ValidationError, match="종료일이 시작일보다 빠릅니다"):
        validate_timeline_inputs(
            snapshot, moves,
            pd.Timestamp("2024-01-10"),
            pd.Timestamp("2024-01-01")  # 시작보다 이른 종료일
        )


# ============================================================
# 필터 테스트
# ============================================================

def test_filter_by_centers():
    """센터 필터링"""
    df = pd.DataFrame({
        "center": ["태광KR", "AMZUS", "상해CN"],
        "qty": [100, 200, 150]
    })
    
    result = filter_by_centers(df, ["태광KR", "AMZUS"])
    
    assert len(result) == 2
    assert set(result["center"]) == {"태광KR", "AMZUS"}


def test_filter_by_skus():
    """SKU 필터링"""
    df = pd.DataFrame({
        "resource_code": ["BA00021", "BA00022", "BA00023"],
        "qty": [100, 200, 150]
    })
    
    result = filter_by_skus(df, ["BA00021", "BA00023"])
    
    assert len(result) == 2
    assert set(result["resource_code"]) == {"BA00021", "BA00023"}


def test_safe_to_datetime():
    """안전한 datetime 변환"""
    # 단일 값
    result = safe_to_datetime("2024-01-01 14:30:00")
    assert result == pd.Timestamp("2024-01-01")
    
    # Series
    series = pd.Series(["2024-01-01", "2024-01-02", "invalid"])
    result_series = safe_to_datetime(series)
    assert result_series.iloc[0] == pd.Timestamp("2024-01-01")
    assert pd.isna(result_series.iloc[2])  # invalid → NaT


def test_extract_center_and_sku_options():
    """센터 및 SKU 옵션 추출"""
    moves = pd.DataFrame({
        "from_center": ["상해CN", "태광KR"],
        "to_center": ["태광KR", "AMZUS"],
        "resource_code": ["BA00021", "BA00022"],
    })
    
    snapshot = pd.DataFrame({
        "center": ["태광KR", "품고KR"],
        "resource_code": ["BA00021", "BA00023"],
    })
    
    centers, skus = extract_center_and_sku_options(moves, snapshot)
    
    assert "태광KR" in centers
    assert "AMZUS" in centers
    assert "BA00021" in skus
    assert "BA00023" in skus


def test_calculate_date_bounds():
    """날짜 범위 계산"""
    today = pd.Timestamp("2024-01-15")
    
    snapshot = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2024-01-10"]),
        "center": ["태광KR", "태광KR"],
        "resource_code": ["BA00021", "BA00021"],
        "stock_qty": [100, 90],
    })
    
    moves = pd.DataFrame({
        "onboard_date": pd.to_datetime(["2024-01-05"]),
        "arrival_date": pd.to_datetime(["2024-01-20"]),
    })
    
    bound_min, bound_max = calculate_date_bounds(
        today=today,
        snapshot_df=snapshot,
        moves_df=moves,
        base_past_days=30,
        base_future_days=30
    )
    
    assert bound_min <= today
    assert bound_max >= today
    assert bound_min <= bound_max


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
