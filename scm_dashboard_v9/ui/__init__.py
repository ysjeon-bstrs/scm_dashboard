"""
UI 레이어의 공개 API

이 모듈은 Streamlit 기반 UI 컴포넌트를 재수출합니다.
차트, KPI 카드, 테이블, 어댑터 등을 포함합니다.
"""

from .adapters import handle_domain_errors
from .charts import (
    render_amazon_sales_vs_inventory,
    render_sku_summary_cards,
    render_step_chart,
)
from .kpi import (
    build_amazon_snapshot_kpis,
    render_amazon_snapshot_kpis,
    render_taekwang_stock_dashboard,
)
from .tables import (
    build_resource_name_map,
    render_inbound_and_wip_tables,
    render_inventory_table,
    render_lot_details,
)

__all__ = (
    # Charts
    "render_step_chart",
    "render_amazon_sales_vs_inventory",
    "render_sku_summary_cards",
    "render_amazon_snapshot_kpis",
    "render_taekwang_stock_dashboard",
    "build_amazon_snapshot_kpis",
    # Tables
    "render_inbound_and_wip_tables",
    "render_inventory_table",
    "render_lot_details",
    "build_resource_name_map",
    # Adapters
    "handle_domain_errors",
)
