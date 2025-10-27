"""Forecast 데이터 모델 정의.

Amazon 예측 컨텍스트 데이터클래스를 제공합니다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class AmazonForecastContext:
    """Amazon 패널에 필요한 실측/예측 시계열 데이터를 묶는 컨테이너.

    Attributes:
        start: 시작 날짜
        end: 종료 날짜
        today: 오늘 날짜
        centers: 센터 리스트
        skus: SKU 리스트
        inv_actual: 실측 재고 DataFrame
        inv_forecast: 예측 재고 DataFrame
        sales_hist: 과거 판매 DataFrame
        sales_ma7: 판매 7일 이동평균 DataFrame
        sales_forecast: 예측 판매 DataFrame
        snapshot_raw: 원본 스냅샷 DataFrame
        snapshot_long: 정제된 스냅샷 DataFrame
        moves: 입고 예정 DataFrame
        lookback_days: 과거 조회 일수
        promotion_multiplier: 프로모션 가중치
    """

    start: pd.Timestamp
    end: pd.Timestamp
    today: pd.Timestamp
    centers: list[str]
    skus: list[str]
    inv_actual: pd.DataFrame
    inv_forecast: pd.DataFrame
    sales_hist: pd.DataFrame
    sales_ma7: pd.DataFrame
    sales_forecast: pd.DataFrame
    snapshot_raw: pd.DataFrame = field(default_factory=pd.DataFrame)
    snapshot_long: pd.DataFrame = field(default_factory=pd.DataFrame)
    moves: pd.DataFrame = field(default_factory=pd.DataFrame)
    lookback_days: int = 28
    promotion_multiplier: float = 1.0
