"""Compatibility facade that mirrors the v5 analytics package."""

from scm_dashboard_v5.analytics.inventory import pivot_inventory_cost_from_raw
from scm_dashboard_v5.analytics.kpi import kpi_breakdown_per_sku
from scm_dashboard_v5.analytics.sales import (
    prepare_amazon_daily_sales,
    prepare_amazon_inventory_layers,
    prepare_amazon_sales_series,
)

__all__ = [
    "kpi_breakdown_per_sku",
    "pivot_inventory_cost_from_raw",
    "prepare_amazon_daily_sales",
    "prepare_amazon_inventory_layers",
    "prepare_amazon_sales_series",
]
