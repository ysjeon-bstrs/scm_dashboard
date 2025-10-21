"""Forecasting facade for the v8 namespace."""

from . import consumption as _consumption

AmazonForecastContext = _consumption.AmazonForecastContext
apply_consumption_with_events = _consumption.apply_consumption_with_events
build_amazon_forecast_context = _consumption.build_amazon_forecast_context
estimate_daily_consumption = _consumption.estimate_daily_consumption
forecast_sales_and_inventory = _consumption.forecast_sales_and_inventory
load_amazon_daily_sales_from_snapshot_raw = (
    _consumption.load_amazon_daily_sales_from_snapshot_raw
)
make_forecast_sales_capped = _consumption.make_forecast_sales_capped

__all__ = [
    "AmazonForecastContext",
    "apply_consumption_with_events",
    "build_amazon_forecast_context",
    "estimate_daily_consumption",
    "forecast_sales_and_inventory",
    "load_amazon_daily_sales_from_snapshot_raw",
    "make_forecast_sales_capped",
]
