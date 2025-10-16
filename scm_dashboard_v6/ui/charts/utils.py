"""
v6 차트 공통 유틸 (임시 래퍼)

- 초기 단계에서는 v5의 구현을 래핑하여 동작을 유지한다.
- 점진적으로 본 파일로 공통 유틸(팔레트, 안전 가드, 변환)을 이전한다.
"""

from __future__ import annotations

from typing import Dict, Sequence, Iterable, List, Any

import pandas as pd


# 간단한 팔레트 정의 (v5 PALETTE와 동일 톤 유지)
_PALETTE = [
    "#4E79A7","#F28E2B","#E15759","#76B7B2","#59A14F",
    "#EDC948","#B07AA1","#FF9DA7","#9C755F","#BAB0AC",
    "#1F77B4","#FF7F0E","#2CA02C","#D62728","#9467BD",
    "#8C564B","#E377C2","#7F7F7F","#BCBD22","#17BECF",
]


def sku_color_map(skus: Sequence[str]) -> Dict[str, str]:
    """SKU별 고정 색상 매핑을 반환한다.

    - v6 단계에서 차트별 팔레트 정책을 재정의할 수 있도록 단순화된 구현을 제공한다.
    - 현재는 v5 톤과 동일한 팔레트를 순환 적용한다.
    """

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


