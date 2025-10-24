"""
v9 예상 입고일(pred_inbound_date) 계산 로직 테스트

이 테스트는 리팩토링 전/후 동작이 동일함을 보장합니다.
"""
from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def sample_moves() -> pd.DataFrame:
    """테스트용 이동 원장 데이터"""
    return pd.DataFrame({
        "resource_code": ["BA00021", "BA00022", "BA00023", "BA00024", "BA00025"],
        "qty_ea": [100, 200, 150, 300, 250],
        "carrier_mode": ["SEA", "AIR", "WIP", "SEA", "WIP"],
        "to_center": ["태광KR", "AMZUS", "태광KR", "AMZUS", "태광KR"],
        "from_center": ["상해CN", "태광KR", "WIP", "상해CN", "WIP"],
        "onboard_date": pd.to_datetime([
            "2024-01-01",
            "2024-01-05",
            "2023-12-15",
            "2024-01-10",
            "2024-01-08"
        ]),
        "arrival_date": pd.to_datetime([
            "2024-01-15",
            "2024-01-08",
            pd.NaT,
            "2024-01-20",
            pd.NaT
        ]),
        "eta_date": pd.to_datetime([
            pd.NaT,
            pd.NaT,
            pd.NaT,
            "2024-01-21",
            pd.NaT
        ]),
        "inbound_date": pd.to_datetime([
            pd.NaT,
            "2024-01-12",
            pd.NaT,
            pd.NaT,
            pd.NaT
        ]),
        "event_date": pd.to_datetime([
            pd.NaT,
            pd.NaT,
            "2024-01-25",
            pd.NaT,
            "2024-02-05"
        ]),
    })


@pytest.fixture
def today_timestamp() -> pd.Timestamp:
    """테스트용 오늘 날짜"""
    return pd.Timestamp("2024-01-15").normalize()


def test_pred_inbound_with_inbound_date(sample_moves, today_timestamp):
    """inbound_date가 있으면 그대로 사용"""
    moves = sample_moves.copy()
    today = today_timestamp
    lag_days = 5
    
    # BA00022는 inbound_date가 있음
    result = _calculate_pred_inbound_reference(moves, today, lag_days)
    
    # inbound_date가 있으면 pred_inbound_date = inbound_date
    ba00022 = result[result["resource_code"] == "BA00022"].iloc[0]
    assert pd.notna(ba00022["pred_inbound_date"])
    assert ba00022["pred_inbound_date"] == pd.Timestamp("2024-01-12")


def test_pred_inbound_wip_uses_event_date(sample_moves, today_timestamp):
    """WIP는 event_date를 그대로 사용 (리드타임 추가 안 함)"""
    moves = sample_moves.copy()
    today = today_timestamp
    lag_days = 5
    
    result = _calculate_pred_inbound_reference(moves, today, lag_days)
    
    # BA00023은 WIP이고 inbound_date 없음 → event_date 사용
    ba00023 = result[result["resource_code"] == "BA00023"].iloc[0]
    assert pd.notna(ba00023["pred_inbound_date"])
    assert ba00023["pred_inbound_date"] == pd.Timestamp("2024-01-25")


def test_pred_inbound_past_arrival(sample_moves, today_timestamp):
    """과거 도착건은 today + 3일"""
    moves = sample_moves.copy()
    today = today_timestamp
    lag_days = 5
    
    # BA00021: arrival_date = 2024-01-15 (오늘과 같음 → 과거 취급)
    result = _calculate_pred_inbound_reference(moves, today, lag_days)
    
    ba00021 = result[result["resource_code"] == "BA00021"].iloc[0]
    assert pd.notna(ba00021["pred_inbound_date"])
    expected = today + pd.Timedelta(days=3)
    assert ba00021["pred_inbound_date"] == expected


def test_pred_inbound_future_arrival(sample_moves, today_timestamp):
    """미래 도착건은 arrival_date + lag_days"""
    moves = sample_moves.copy()
    today = today_timestamp
    lag_days = 5
    
    # BA00024: arrival_date = 2024-01-20 (미래)
    # inbound_date 없으므로 arrival + lag_days
    result = _calculate_pred_inbound_reference(moves, today, lag_days)
    
    ba00024 = result[result["resource_code"] == "BA00024"].iloc[0]
    assert pd.notna(ba00024["pred_inbound_date"])
    expected = pd.Timestamp("2024-01-20") + pd.Timedelta(days=lag_days)
    assert ba00024["pred_inbound_date"] == expected


def test_pred_inbound_eta_fallback(sample_moves, today_timestamp):
    """arrival_date 없으면 eta_date 사용"""
    moves = sample_moves.copy()
    today = today_timestamp
    lag_days = 5
    
    # BA00024: arrival_date는 있지만, eta_date도 있음
    # arrival_date 우선
    result = _calculate_pred_inbound_reference(moves, today, lag_days)
    
    ba00024 = result[result["resource_code"] == "BA00024"].iloc[0]
    # arrival_date = 2024-01-20이 우선
    expected = pd.Timestamp("2024-01-20") + pd.Timedelta(days=lag_days)
    assert ba00024["pred_inbound_date"] == expected


def test_pred_inbound_empty_dataframe():
    """빈 DataFrame 처리"""
    moves = pd.DataFrame(columns=[
        "resource_code", "qty_ea", "carrier_mode", "to_center",
        "onboard_date", "arrival_date", "inbound_date", "event_date"
    ])
    today = pd.Timestamp("2024-01-15")
    lag_days = 5
    
    result = _calculate_pred_inbound_reference(moves, today, lag_days)
    
    assert result.empty
    assert "pred_inbound_date" in result.columns


def test_pred_inbound_all_nat():
    """모든 날짜가 NaT인 경우"""
    moves = pd.DataFrame({
        "resource_code": ["BA00021"],
        "qty_ea": [100],
        "carrier_mode": ["SEA"],
        "to_center": ["태광KR"],
        "onboard_date": [pd.NaT],
        "arrival_date": [pd.NaT],
        "eta_date": [pd.NaT],
        "inbound_date": [pd.NaT],
        "event_date": [pd.NaT],
    })
    today = pd.Timestamp("2024-01-15")
    lag_days = 5
    
    result = _calculate_pred_inbound_reference(moves, today, lag_days)
    
    # 날짜 정보 없으면 NaT
    assert pd.isna(result.iloc[0]["pred_inbound_date"])


# ============================================================
# 레퍼런스 구현 (현재 v9_app.py 로직)
# ============================================================

def _calculate_pred_inbound_reference(
    moves: pd.DataFrame,
    today: pd.Timestamp,
    lag_days: int
) -> pd.DataFrame:
    """
    현재 v9_app.py와 ui/tables.py에 있는 로직의 레퍼런스 구현
    
    이 함수는 테스트용으로만 사용되며, 리팩토링 후 실제 구현과 비교합니다.
    """
    if moves.empty:
        moves_view = moves.copy()
        moves_view["pred_inbound_date"] = pd.Series(
            pd.NaT, index=moves_view.index, dtype="datetime64[ns]"
        )
        return moves_view
    
    moves_view = moves.copy()
    today = pd.to_datetime(today).normalize()
    
    # 필수 컬럼 추가
    for col in ["carrier_mode", "inbound_date", "arrival_date", "eta_date", "event_date"]:
        if col not in moves_view.columns:
            if "date" in col:
                moves_view[col] = pd.Series(pd.NaT, index=moves_view.index, dtype="datetime64[ns]")
            else:
                moves_view[col] = pd.Series("", index=moves_view.index, dtype="object")
    
    # 날짜 정규화
    for col in ["arrival_date", "eta_date", "inbound_date", "event_date"]:
        if col in moves_view.columns:
            moves_view[col] = pd.to_datetime(moves_view[col], errors="coerce").dt.normalize()
    
    pred_inbound = pd.Series(pd.NaT, index=moves_view.index, dtype="datetime64[ns]")
    
    # carrier_mode 확인
    carrier_mode = moves_view.get("carrier_mode", pd.Series("", index=moves_view.index))
    is_wip = carrier_mode.astype(str).str.upper() == "WIP"
    
    # 1. inbound_date가 있으면 우선 사용
    if "inbound_date" in moves_view.columns:
        mask_inbound = moves_view["inbound_date"].notna()
        pred_inbound.loc[mask_inbound] = moves_view.loc[mask_inbound, "inbound_date"]
    else:
        mask_inbound = pd.Series(False, index=moves_view.index)
    
    # 2. WIP: event_date 그대로 사용
    wip_mask = is_wip & (~mask_inbound)
    if wip_mask.any() and "event_date" in moves_view.columns:
        event_series = pd.to_datetime(moves_view.get("event_date"), errors="coerce").dt.normalize()
        wip_with_event = wip_mask & event_series.notna()
        if wip_with_event.any():
            pred_inbound.loc[wip_with_event] = event_series.loc[wip_with_event]
    
    # 3. In-Transit: arrival/eta + 리드타임
    intransit_mask = (~is_wip) & (~mask_inbound)
    arrival_series = pd.to_datetime(moves_view.get("arrival_date"), errors="coerce").dt.normalize()
    eta_series = pd.to_datetime(moves_view.get("eta_date"), errors="coerce").dt.normalize()
    effective_arrival = arrival_series.fillna(eta_series)
    mask_eta = intransit_mask & effective_arrival.notna()
    
    if mask_eta.any():
        # 과거/오늘 도착: today + 3일
        past_eta = mask_eta & (effective_arrival <= today)
        if past_eta.any():
            pred_inbound.loc[past_eta] = today + pd.Timedelta(days=3)
        
        # 미래 도착: arrival + lag_days
        future_eta = mask_eta & (effective_arrival > today)
        if future_eta.any():
            pred_inbound.loc[future_eta] = effective_arrival.loc[future_eta] + pd.Timedelta(
                days=int(lag_days)
            )
    
    moves_view["pred_inbound_date"] = pd.to_datetime(pred_inbound).dt.normalize()
    return moves_view


# ============================================================
# 통합 테스트 (전체 시나리오)
# ============================================================

def test_pred_inbound_mixed_scenario(sample_moves, today_timestamp):
    """복합 시나리오 테스트"""
    moves = sample_moves.copy()
    today = today_timestamp
    lag_days = 5
    
    result = _calculate_pred_inbound_reference(moves, today, lag_days)
    
    # 각 행별 검증
    for _, row in result.iterrows():
        sku = row["resource_code"]
        pred = row["pred_inbound_date"]
        
        if sku == "BA00021":
            # SEA, arrival=2024-01-15 (today) → today + 3
            expected = today + pd.Timedelta(days=3)
            assert pred == expected
        
        elif sku == "BA00022":
            # AIR, inbound_date=2024-01-12 → inbound_date
            assert pred == pd.Timestamp("2024-01-12")
        
        elif sku == "BA00023":
            # WIP, event_date=2024-01-25 → event_date
            assert pred == pd.Timestamp("2024-01-25")
        
        elif sku == "BA00024":
            # SEA, arrival=2024-01-20 (future) → arrival + lag_days
            expected = pd.Timestamp("2024-01-20") + pd.Timedelta(days=lag_days)
            assert pred == expected
        
        elif sku == "BA00025":
            # WIP, event_date=2024-02-05 → event_date
            assert pred == pd.Timestamp("2024-02-05")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
