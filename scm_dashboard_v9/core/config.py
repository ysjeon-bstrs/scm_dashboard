"""Configuration and constants for the SCM dashboard v9.

센터별 재고 컬럼 매핑, Google Sheets ID 등 전역 설정을 제공합니다.
"""

from __future__ import annotations
from dataclasses import dataclass, field

# ============================================================
# Google Sheets 설정
# ============================================================

# Google Sheets 문서 ID
GSHEET_ID = "1RYjKW2UDJ2kWJLAqQH26eqx2-r9Xb0_qE_hfwu9WIj8"


# ============================================================
# 센터별 재고 컬럼 매핑
# ============================================================

# 센터별 snapshot_raw에서 사용하는 재고 컬럼명 매핑
CENTER_COL = {
    "태광KR": "stock2",
    "AMZUS": "fba_available_stock",
    "품고KR": "poomgo_v2_available_stock",
    "SBSPH": "shopee_ph_available_stock",
    "SBSSG": "shopee_sg_available_stock",
    "SBSMY": "shopee_my_available_stock",
    "AcrossBUS": "acrossb_available_stock",
    "어크로스비US": "acrossb_available_stock",
}


# ============================================================
# 타임라인 설정
# ============================================================

@dataclass(frozen=True)
class TimelineConfig:
    """타임라인 빌드 및 예측 관련 설정"""
    
    # 과거 도착건 처리 시 적용할 버퍼 일수
    # (과거에 도착했지만 아직 입고되지 않은 물품의 예상 입고일)
    past_arrival_buffer_days: int = 3
    
    # 기본 입고 반영 리드타임 (일)
    # (도착일로부터 실제 입고까지 걸리는 시간)
    default_lag_days: int = 5
    
    # 날짜 정보가 없을 때 사용할 폴백 일수
    fallback_days: int = 1
    
    # 타임라인 예측 범위 확장 (일)
    horizon_pad_days: int = 60


@dataclass(frozen=True)
class ConsumptionConfig:
    """소비 예측 관련 설정"""
    
    # 추세 계산 기본 기간 (일)
    default_lookback_days: int = 28
    
    # 추세 계산 최소 기간 (일)
    min_lookback_days: int = 7
    
    # 추세 계산 최대 기간 (일)
    max_lookback_days: int = 56
    
    # 프로모션 uplift 최소값 (비율)
    min_promo_uplift: float = -1.0
    
    # 프로모션 uplift 최대값 (비율)
    max_promo_uplift: float = 3.0


@dataclass(frozen=True)
class UIConfig:
    """UI 표시 관련 설정"""
    
    # 기본 과거 표시 기간 (일)
    default_past_days: int = 20
    
    # 기본 미래 표시 기간 (일)
    default_future_days: int = 30
    
    # 날짜 슬라이더 기본 범위 (과거, 일)
    base_past_days: int = 42
    
    # 날짜 슬라이더 기본 범위 (미래, 일)
    base_future_days: int = 42
    
    # 테이블 최대 표시 행 수
    max_table_rows: int = 1000
    
    # KPI 카드 청크 크기
    kpi_card_chunk_size: int = 2
    
    # 테이블 기본 높이 (픽셀)
    table_height_inbound: int = 300
    table_height_wip: int = 260
    table_height_inventory: int = 380
    table_height_lot: int = 320


@dataclass(frozen=True)
class DashboardConfig:
    """대시보드 전역 설정"""
    
    timeline: TimelineConfig = field(default_factory=TimelineConfig)
    consumption: ConsumptionConfig = field(default_factory=ConsumptionConfig)
    ui: UIConfig = field(default_factory=UIConfig)


# ============================================================
# 전역 설정 인스턴스
# ============================================================

# 전역 설정 객체 (불변)
CONFIG = DashboardConfig()
