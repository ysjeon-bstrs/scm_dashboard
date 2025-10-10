"""UI rendering helpers for the Streamlit dashboard."""

from __future__ import annotations

from importlib import import_module
from typing import Any

import streamlit as st

from .kpi import render_sku_summary_cards

__all__ = [
    "render_amazon_sales_vs_inventory",
    "render_step_chart",
    "render_sku_summary_cards",
]


def _load_charts_module() -> Any | None:
    """Return the charts module if its optional dependencies are available."""

    try:
        return import_module("scm_dashboard_v5.ui.charts")
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on runtime env
        missing = getattr(exc, "name", "")
        if missing and missing.split(".")[0] == "plotly":
            st.error(
                "Plotly 라이브러리가 설치되어야 Amazon 판매/재고 및 계단식 차트를 표시할 수 있습니다. "
                "requirements.txt에 'plotly>=5.0.0' 항목이 포함되어 있는지 확인해주세요.",
                icon="⚠️",
            )
            return None
        raise


def render_step_chart(*args: Any, **kwargs: Any) -> None:
    """Proxy to the step chart renderer, guarding against optional deps."""

    charts = _load_charts_module()
    if charts is None:
        return
    charts.render_step_chart(*args, **kwargs)


def render_amazon_sales_vs_inventory(*args: Any, **kwargs: Any) -> None:
    """Proxy to the Amazon sales vs. inventory renderer."""

    charts = _load_charts_module()
    if charts is None:
        return
    charts.render_amazon_sales_vs_inventory(*args, **kwargs)
