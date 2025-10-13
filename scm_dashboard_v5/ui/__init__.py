"""UI rendering helpers for the Streamlit dashboard."""

from __future__ import annotations

from importlib import import_module
from typing import Any

import streamlit as st

from .kpi import render_sku_summary_cards

__all__ = [
    "render_amazon_sales_vs_inventory",
    "render_amazon_panel",
    "render_step_chart",
    "render_sku_summary_cards",
]

_CHARTS_MODULE: Any | None = None
_CHARTS_IMPORT_ERROR: ImportError | None = None


def _should_suppress_import_error(exc: ImportError) -> bool:
    """Return True if the charts module failed because of optional deps."""

    missing = getattr(exc, "name", "") or ""
    missing_root = missing.split(".")[0]
    if missing_root.startswith("scm_dashboard_v5"):
        return False

    message = str(exc).lower()
    if "scm_dashboard_v5" in message and "plotly" not in message:
        return False

    return True


def _load_charts_module() -> Any | None:
    """Return the charts module if its optional dependencies are available."""

    global _CHARTS_MODULE, _CHARTS_IMPORT_ERROR

    if _CHARTS_MODULE is not None:
        return _CHARTS_MODULE

    if _CHARTS_IMPORT_ERROR is not None:
        _show_plotly_error(_CHARTS_IMPORT_ERROR)
        return None

    try:
        _CHARTS_MODULE = import_module("scm_dashboard_v5.ui.charts")
    except ImportError as exc:  # pragma: no cover - depends on runtime env
        if _should_suppress_import_error(exc):
            _CHARTS_IMPORT_ERROR = exc
            _show_plotly_error(exc)
            return None
        raise

    return _CHARTS_MODULE


def _show_plotly_error(exc: ImportError) -> None:
    """Display a user-friendly warning about missing Plotly dependencies."""

    missing = getattr(exc, "name", "") or "Plotly"
    message = str(exc)
    if message:
        detail = f"<details><summary>오류 세부정보</summary><pre>{message}</pre></details>"
    else:
        detail = ""

    guidance = (
        "Plotly 라이브러리가 설치되어야 Amazon 판매/재고 및 계단식 차트를 표시할 수 있습니다. "
        "requirements.txt에 'plotly>=5.0.0' 항목이 포함되어 있는지 확인해주세요.\n"
        f"'{missing}' 모듈을 불러오지 못했습니다. Plotly 및 관련 의존성이 올바르게 설치되었는지 확인해주세요."
    )

    st.error(f"{guidance}{detail}", icon="⚠️")


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


def render_amazon_panel(*args: Any, **kwargs: Any) -> None:
    """Proxy to the Amazon summary panel renderer."""

    charts = _load_charts_module()
    if charts is None:
        return
    charts.render_amazon_panel(*args, **kwargs)
