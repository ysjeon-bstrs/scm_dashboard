"""
v7 차트 공통 유틸

설명(한글):
- v5 구현과 톤을 맞추되, 필요한 최소 기능만 래핑 제공합니다.
"""

from __future__ import annotations

from typing import Dict, Sequence, Any, Iterable, List

import pandas as pd
from scm_dashboard_v5.ui.charts import (
    _safe_dataframe as _v5_safe_dataframe,
    _safe_add_bar as _v5_safe_add_bar,
    _safe_add_scatter as _v5_safe_add_scatter,
)


_PALETTE = [
    "#4E79A7","#F28E2B","#E15759","#76B7B2","#59A14F",
    "#EDC948","#B07AA1","#FF9DA7","#9C755F","#BAB0AC",
    "#1F77B4","#FF7F0E","#2CA02C","#D62728","#9467BD",
    "#8C564B","#E377C2","#7F7F7F","#BCBD22","#17BECF",
]


def sku_color_map(skus: Sequence[str]) -> Dict[str, str]:
    """SKU별 고정 색상 매핑을 반환한다."""
    mapping: Dict[str, str] = {}
    i = 0
    for sku in skus:
        if sku not in mapping:
            mapping[sku] = _PALETTE[i % len(_PALETTE)]
            i += 1
    return mapping


def to_plot_list(values: Any) -> List:
    """Plotly 입력을 위한 값들을 리스트로 정제한다."""
    if values is None:
        return []
    if isinstance(values, (pd.Index, pd.Series)):
        return values.dropna().tolist()
    if hasattr(values, "tolist"):
        return [v for v in values.tolist() if not pd.isna(v)]
    if isinstance(values, Iterable) and not isinstance(values, (str, bytes)):
        return [v for v in values if not pd.isna(v)]
    return [] if pd.isna(values) else [values]


def safe_dataframe(df: Any, *, index: Any = None, columns: Any = None, fill_value: float = 0.0, dtype: type = float) -> pd.DataFrame:
    """v5의 안전한 DataFrame 보조를 래핑한다."""
    return _v5_safe_dataframe(df, index=index, columns=columns, fill_value=fill_value, dtype=dtype)


def safe_add_bar(fig: Any, *, x: Any, y: Any, name: str, marker_color: Any, **kwargs: Any) -> None:
    """v5의 안전한 bar 추가 래퍼"""
    return _v5_safe_add_bar(fig, x=x, y=y, name=name, marker_color=marker_color, **kwargs)


def safe_add_scatter(fig: Any, *, x: Any, y: Any, name: str, line: Any | None = None, yaxis: str = "y", mode: str = "lines", **kwargs: Any) -> None:
    """v5의 안전한 scatter 추가 래퍼"""
    return _v5_safe_add_scatter(fig, x=x, y=y, name=name, line=line, yaxis=yaxis, mode=mode, **kwargs)


