"""차트 렌더링 모듈.

기존 scm_dashboard_v9/ui/charts.py를 여러 서브모듈로 분리했습니다.
기존 import 경로를 유지하기 위해 모든 함수를 re-export합니다.
"""

# Public API (외부에서 직접 사용하는 함수)
from .renderers import render_amazon_sales_vs_inventory, render_step_chart
from scm_dashboard_v9.ui.kpi import render_sku_summary_cards as _render_sku_summary_cards

# Color utilities
from .colors import (
    PALETTE,
    STEP_PALETTE,
    CENTER_SHADE,
    DEFAULT_SHADE_STEP,
    hex_to_rgb as _hex_to_rgb,
    rgb_to_hex as _rgb_to_hex,
    tint as _tint,
    shade_for as _shade_for,
    sku_colors as _sku_colors,
    sku_color_map as _sku_color_map,
    step_sku_color_map as _step_sku_color_map,
)

# Data utilities
from .data_utils import (
    safe_dataframe as _safe_dataframe,
    safe_series as _safe_series,
    as_naive_timestamp as _as_naive_timestamp,
    ensure_naive_index as _ensure_naive_index,
    normalize_inventory_frame as _normalize_inventory_frame,
    normalize_sales_frame as _normalize_sales_frame,
    coerce_cols as _coerce_cols,
    empty_sales_frame as _empty_sales_frame,
)

# Plotly helpers
from .plotly_helpers import (
    ensure_plotly_available as _ensure_plotly_available,
    to_plot_list as _to_plot_list,
    safe_add_bar as _safe_add_bar,
    safe_add_scatter as _safe_add_scatter,
)

# Filters
from .filters import (
    is_wip_center_name as _is_wip_center_name,
    drop_wip_centers as _drop_wip_centers,
    pick_amazon_centers as _pick_amazon_centers,
    contains_wip_center as _contains_wip_center,
)

# Sales calculation
from .sales import (
    sales_from_snapshot as _sales_from_snapshot,
    sales_forecast_ma as _sales_forecast_ma,
    sales_from_snapshot_raw as _sales_from_snapshot_raw,
    sales_forecast_from_inventory_projection as _sales_forecast_from_inventory_projection,
    sales_from_snapshot_decays as _sales_from_snapshot_decays,
)

# Inventory helpers
from .inventory import (
    total_inventory_series as _total_inventory_series,
    trim_sales_forecast_to_inventory as _trim_sales_forecast_to_inventory,
    inventory_matrix as _inventory_matrix,
    timeline_inventory_matrix as _timeline_inventory_matrix,
    clamped_forecast_series as _clamped_forecast_series,
)


def render_sku_summary_cards(*args: object, **kwargs: object):
    """KPI 카드 렌더러를 호환성을 위해 노출합니다."""
    return _render_sku_summary_cards(*args, **kwargs)


# Public exports (v5_main.py에서 사용)
__all__ = [
    "render_amazon_sales_vs_inventory",
    "render_step_chart",
    "render_sku_summary_cards",
    "PALETTE",
    "STEP_PALETTE",
]
