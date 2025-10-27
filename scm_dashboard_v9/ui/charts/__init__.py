"""차트 렌더링 모듈.

기존 scm_dashboard_v9/ui/charts.py를 여러 서브모듈로 분리했습니다.
기존 import 경로를 유지하기 위해 모든 함수를 re-export합니다.
"""

from scm_dashboard_v9.ui.kpi import (
    render_sku_summary_cards as _render_sku_summary_cards,
)

# Color utilities
from .colors import (
    CENTER_SHADE,
    DEFAULT_SHADE_STEP,
    PALETTE,
    STEP_PALETTE,
)
from .colors import hex_to_rgb as _hex_to_rgb
from .colors import rgb_to_hex as _rgb_to_hex
from .colors import shade_for as _shade_for
from .colors import sku_color_map as _sku_color_map
from .colors import sku_colors as _sku_colors
from .colors import step_sku_color_map as _step_sku_color_map
from .colors import tint as _tint

# Data utilities
from .data_utils import as_naive_timestamp as _as_naive_timestamp
from .data_utils import coerce_cols as _coerce_cols
from .data_utils import empty_sales_frame as _empty_sales_frame
from .data_utils import ensure_naive_index as _ensure_naive_index
from .data_utils import normalize_inventory_frame as _normalize_inventory_frame
from .data_utils import normalize_sales_frame as _normalize_sales_frame
from .data_utils import safe_dataframe as _safe_dataframe
from .data_utils import safe_series as _safe_series

# Filters
from .filters import contains_wip_center as _contains_wip_center
from .filters import drop_wip_centers as _drop_wip_centers
from .filters import is_wip_center_name as _is_wip_center_name
from .filters import pick_amazon_centers as _pick_amazon_centers

# Inventory helpers
from .inventory import clamped_forecast_series as _clamped_forecast_series
from .inventory import inventory_matrix as _inventory_matrix
from .inventory import timeline_inventory_matrix as _timeline_inventory_matrix
from .inventory import total_inventory_series as _total_inventory_series
from .inventory import (
    trim_sales_forecast_to_inventory as _trim_sales_forecast_to_inventory,
)

# Plotly helpers
from .plotly_helpers import ensure_plotly_available as _ensure_plotly_available
from .plotly_helpers import safe_add_bar as _safe_add_bar
from .plotly_helpers import safe_add_scatter as _safe_add_scatter
from .plotly_helpers import to_plot_list as _to_plot_list

# Public API (외부에서 직접 사용하는 함수)
from .renderers import render_amazon_sales_vs_inventory, render_step_chart

# Sales calculation
from .sales import (
    sales_forecast_from_inventory_projection as _sales_forecast_from_inventory_projection,
)
from .sales import sales_forecast_ma as _sales_forecast_ma
from .sales import sales_from_snapshot as _sales_from_snapshot
from .sales import sales_from_snapshot_decays as _sales_from_snapshot_decays
from .sales import sales_from_snapshot_raw as _sales_from_snapshot_raw


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
