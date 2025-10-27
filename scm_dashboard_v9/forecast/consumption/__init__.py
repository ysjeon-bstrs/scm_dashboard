"""Forecast 소비량 계산 모듈.

기존 scm_dashboard_v9/forecast/consumption.py를 여러 서브모듈로 분리했습니다.
기존 import 경로를 유지하기 위해 모든 클래스와 함수를 re-export합니다.
"""

# Context builder
from .context import build_amazon_forecast_context

# Consumption estimation
from .estimation import apply_consumption_with_events, estimate_daily_consumption

# Inventory forecasting
from .inventory import forecast_sales_and_inventory

# Models
from .models import AmazonForecastContext

# Sales helpers
from .sales import load_amazon_daily_sales_from_snapshot_raw, make_forecast_sales_capped

__all__ = [
    "AmazonForecastContext",
    "make_forecast_sales_capped",
    "load_amazon_daily_sales_from_snapshot_raw",
    "forecast_sales_and_inventory",
    "build_amazon_forecast_context",
    "estimate_daily_consumption",
    "apply_consumption_with_events",
]
