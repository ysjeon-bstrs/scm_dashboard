"""Forecasting facade for SCM dashboard v5.

Historically callers imported helpers straight from the ``scm_dashboard_v5.forecast``
package.  A recent refactor introduced ``build_amazon_forecast_context`` but the
Streamlit entrypoint still expects it to be re-exported at the package level.
Some environments ended up loading an outdated ``__init__`` module (e.g. cached
bytecode) and therefore raised ``ImportError`` when the new helper was missing.
To make the facade more robust we now proxy attributes explicitly from the
``consumption`` submodule and declare ``__all__`` accordingly.  This guarantees
that the package always exposes the same surface area regardless of import
ordering or stale caches.
"""

from . import consumption as _consumption

AmazonForecastContext = _consumption.AmazonForecastContext
apply_consumption_with_events = _consumption.apply_consumption_with_events
build_amazon_forecast_context = _consumption.build_amazon_forecast_context
estimate_daily_consumption = _consumption.estimate_daily_consumption
forecast_sales_and_inventory = _consumption.forecast_sales_and_inventory
load_amazon_daily_sales_from_snapshot_raw = (
    _consumption.load_amazon_daily_sales_from_snapshot_raw
)

__all__ = [
    "AmazonForecastContext",
    "apply_consumption_with_events",
    "build_amazon_forecast_context",
    "estimate_daily_consumption",
    "forecast_sales_and_inventory",
    "load_amazon_daily_sales_from_snapshot_raw",
]
