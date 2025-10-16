"""
Forecast 컨텍스트 (v6)

- v5의 AmazonForecastContext를 가져와 타입/경계를 명확히 하며,
  초기에는 동일 구조를 재사용한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import pandas as pd


@dataclass
class AmazonForecastContext:
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


