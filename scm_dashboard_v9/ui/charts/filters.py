"""센터 및 WIP 필터링 유틸리티 모듈.

Amazon 센터 추출, WIP 센터 감지 및 필터링 기능을 제공합니다.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence

import pandas as pd


def is_wip_center_name(value: object) -> bool:
    """주어진 센터 이름이 WIP/생산 센터인지 확인합니다.

    Args:
        value: 센터 이름

    Returns:
        WIP 센터이면 True
    """
    text = str(value or "")
    upper = text.upper()
    if "WIP" in upper or "PRODUCTION" in upper:
        return True
    return "생산" in text


def drop_wip_centers(df: pd.DataFrame, *, center_col: str = "center") -> pd.DataFrame:
    """DataFrame에서 WIP/생산 센터 행을 제거합니다.

    Args:
        df: 입력 DataFrame
        center_col: 센터 컬럼 이름

    Returns:
        WIP 센터가 제거된 DataFrame
    """
    if not isinstance(df, pd.DataFrame) or center_col not in df.columns:
        return df

    centers = df[center_col].astype(str)
    mask = centers.map(is_wip_center_name)
    if not mask.any():
        return df.copy()

    return df.loc[~mask].copy()


def pick_amazon_centers(all_centers: Iterable[str]) -> List[str]:
    """선택된 센터 중 Amazon 계열 센터만 추출합니다.

    센터 이름에 "AMZ" 또는 "AMAZON"이 포함된 센터를 찾습니다.

    Args:
        all_centers: 전체 센터 리스트

    Returns:
        Amazon 센터 리스트
    """
    out = []
    for c in all_centers:
        if not c:
            continue
        c_up = str(c).upper()
        if c_up.startswith("AMZ") or "AMAZON" in c_up:
            out.append(str(c))
    return out


def contains_wip_center(centers: Sequence[str]) -> bool:
    """선택된 센터에 WIP/태광 센터가 포함되어 있는지 확인합니다.

    Args:
        centers: 센터 리스트

    Returns:
        WIP 센터가 포함되어 있으면 True
    """
    for center in centers:
        norm = str(center).replace(" ", "").lower()
        if not norm:
            continue
        if norm == "wip":
            return True
        if "태광" in norm or "taekwang" in norm or "tae-kwang" in norm:
            return True
    return False
