"""Plotly 차트 렌더링 헬퍼 함수 모듈.

Plotly 미설치 환경 처리 및 안전한 trace 추가 함수를 제공합니다.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import streamlit as st

try:
    import plotly.graph_objects as go  # type: ignore
except ImportError as _plotly_err:
    go = None  # type: ignore[assignment]
    _PLOTLY_IMPORT_ERROR = _plotly_err
else:
    _PLOTLY_IMPORT_ERROR = None

_PLOTLY_WARNING_EMITTED = False


def ensure_plotly_available() -> bool:
    """Plotly가 설치되어 있는지 확인하고, 없으면 경고 메시지를 표시합니다.

    Returns:
        Plotly 사용 가능 여부
    """
    global _PLOTLY_WARNING_EMITTED
    if _PLOTLY_IMPORT_ERROR is None:
        return True
    if not _PLOTLY_WARNING_EMITTED:
        st.warning(
            "Plotly가 설치되어 있지 않아 차트를 렌더링할 수 없습니다. "
            "관리자에게 Plotly 설치를 요청하거나 requirements를 확인하세요.\n"
            f"원인: {_PLOTLY_IMPORT_ERROR}"
        )
        _PLOTLY_WARNING_EMITTED = True
    return False


def to_plot_list(values: Optional[Iterable]) -> List:
    """임의의 iterable 값을 Plotly API용 리스트로 변환합니다.

    None 값과 NaN 값을 제거하여 안전한 리스트를 반환합니다.

    Args:
        values: 변환할 값 (Series, Index, ndarray, list 등)

    Returns:
        정제된 리스트
    """
    import pandas as pd
    import numpy as np

    if values is None:
        return []

    if isinstance(values, (pd.Index, pd.Series)):
        cleaned = values.dropna().tolist()
    elif isinstance(values, np.ndarray):
        cleaned = [v for v in values.tolist() if not pd.isna(v)]
    elif isinstance(values, Iterable) and not isinstance(values, (str, bytes)):
        cleaned = [v for v in values if not pd.isna(v)]
    else:
        cleaned = [] if pd.isna(values) else [values]

    return cleaned


def safe_add_bar(
    fig: "go.Figure",
    *,
    x: Optional[Iterable],
    y: Optional[Iterable],
    name: str,
    marker_color: Optional[str],
    **kwargs: object,
) -> None:
    """Bar trace를 안전하게 추가합니다.

    데이터가 유효하지 않거나 에러가 발생하면 조용히 무시합니다.

    Args:
        fig: Plotly Figure 객체
        x: x축 데이터
        y: y축 데이터
        name: trace 이름
        marker_color: 막대 색상
        **kwargs: 추가 Plotly 옵션
    """
    xs = to_plot_list(x)
    ys = to_plot_list(y)

    if not xs or not ys:
        return

    if marker_color is None:
        return

    if len(xs) != len(ys):
        limit = min(len(xs), len(ys))
        xs = xs[:limit]
        ys = ys[:limit]

    try:
        fig.add_bar(x=xs, y=ys, name=name, marker_color=marker_color, **kwargs)
    except Exception:  # pragma: no cover
        return


def safe_add_scatter(
    fig: "go.Figure",
    *,
    x: Optional[Iterable],
    y: Optional[Iterable],
    name: str,
    line: Optional[Dict[str, object]] = None,
    yaxis: str = "y",
    mode: str = "lines",
    **kwargs: object,
) -> None:
    """Scatter trace를 안전하게 추가합니다.

    데이터가 유효하지 않거나 에러가 발생하면 조용히 무시합니다.

    Args:
        fig: Plotly Figure 객체
        x: x축 데이터
        y: y축 데이터
        name: trace 이름
        line: 선 스타일 딕셔너리
        yaxis: y축 이름 ("y" 또는 "y2")
        mode: 모드 ("lines", "markers", "lines+markers" 등)
        **kwargs: 추가 Plotly 옵션
    """
    xs = to_plot_list(x)
    ys = to_plot_list(y)

    if not xs or not ys:
        return

    try:
        fig.add_trace(
            go.Scatter(x=xs, y=ys, name=name, mode=mode, line=line, yaxis=yaxis, **kwargs)
        )
    except Exception:  # pragma: no cover
        return
