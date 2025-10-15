"""Forecasting facade for SCM dashboard v5."""

from .consumption import (
    AmazonForecastContext,
    apply_consumption_with_events,
    build_amazon_forecast_context,
    estimate_daily_consumption,
    forecast_sales_and_inventory,
    load_amazon_daily_sales_from_snapshot_raw,
)

__all__ = [
    "AmazonForecastContext",
    "apply_consumption_with_events",
    "build_amazon_forecast_context",
    "estimate_daily_consumption",
    "forecast_sales_and_inventory",
    "load_amazon_daily_sales_from_snapshot_raw",
]
