"""KPI 렌더링 모듈.

기존 scm_dashboard_v9/ui/kpi.py를 여러 서브모듈로 분리했습니다.
기존 import 경로를 유지하기 위해 모든 함수를 re-export합니다.
"""

# Public API (외부에서 직접 사용하는 함수)
from .amazon_snapshot import build_amazon_snapshot_kpis, render_amazon_snapshot_kpis
from .cards import build_center_card as _build_center_card
from .cards import build_grid as _build_grid
from .cards import build_metric_card as _build_metric_card
from .cards import center_grid_layout as _center_grid_layout
from .cards import render_sku_summary_cards

# Internal utilities (하위 호환성을 위해 export)
from .formatters import calculate_coverage_days as _calculate_coverage_days
from .formatters import calculate_sellout_date as _calculate_sellout_date
from .formatters import escape as _escape
from .formatters import format_date as _format_date
from .formatters import format_days as _format_days
from .formatters import format_number as _format_number
from .formatters import should_show_in_transit as _should_show_in_transit
from .formatters import value_font_size as _value_font_size
from .metrics import compute_depletion_from_timeline, compute_depletion_metrics
from .metrics import extract_daily_demand as _extract_daily_demand
from .metrics import movement_breakdown_per_center as _movement_breakdown_per_center
from .shopee_snapshot import build_shopee_snapshot_kpis, render_shopee_snapshot_kpis
from .styles import inject_responsive_styles as _inject_responsive_styles
from .taekwang_stock import render_taekwang_stock_dashboard

__all__ = [
    "build_amazon_snapshot_kpis",
    "build_shopee_snapshot_kpis",
    "render_amazon_snapshot_kpis",
    "render_shopee_snapshot_kpis",
    "render_sku_summary_cards",
    "render_taekwang_stock_dashboard",
    "compute_depletion_from_timeline",
    "compute_depletion_metrics",
]
