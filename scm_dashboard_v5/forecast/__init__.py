"""Forecasting facade for SCM dashboard v5."""

from .consumption import (
    apply_consumption_with_events,
    estimate_daily_consumption,
    forecast_sales_and_inventory,
    load_amazon_daily_sales_from_snapshot_raw,
)

__all__ = [
    "apply_consumption_with_events",
    "estimate_daily_consumption",
    "forecast_sales_and_inventory",
    "load_amazon_daily_sales_from_snapshot_raw",
]
