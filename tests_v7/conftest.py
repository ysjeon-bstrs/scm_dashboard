"""
tests_v7 공통 픽스처

설명(한글):
- 골든 마스터 비교용 테스트 데이터 로딩 및 정규화 유틸을 제공합니다.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Callable

import os
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
GOLDEN_DIR = ROOT / "tests_v7" / "golden"

# CI 환경에서 패키지 임포트를 보장하기 위해 프로젝트 루트를 sys.path에 추가
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def golden_dir() -> Path:
    """골든 산출물 디렉터리 경로"""
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    return GOLDEN_DIR


@pytest.fixture(scope="session")
def normalize_df() -> Callable[[pd.DataFrame, list[str]], pd.DataFrame]:
    """
    비교 안정화를 위한 DataFrame 정규화 함수 팩토리

    - 날짜 normalize, 수치형 캐스팅, 컬럼 정렬, 정렬 키 적용
    """

    def _normalize(df: pd.DataFrame, sort_keys: list[str]) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=sort_keys)
        out = df.copy()
        # 날짜 후보 컬럼을 모두 normalize
        for col in ["date", "snapshot_date", "inbound_date", "arrival_date", "event_date"]:
            if col in out.columns:
                out[col] = pd.to_datetime(out[col], errors="coerce").dt.normalize()
        # 수치 컬럼 정리(가능한 경우만)
        for col in out.columns:
            if col.endswith("qty") or col.endswith("_ea") or col in {"stock_qty", "sales_ea", "qty_ea"}:
                try:
                    out[col] = pd.to_numeric(out[col], errors="coerce")
                except Exception:
                    pass
        # 정렬 키가 존재하는 경우만 값 유지
        for key in list(sort_keys):
            if key not in out.columns:
                out[key] = pd.NA
        out = out[sorted(out.columns)]
        out = out.sort_values([k for k in sort_keys if k in out.columns]).reset_index(drop=True)
        return out

    return _normalize


def _bool_env(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.lower() in {"1", "true", "yes", "on"}


UPDATE_GOLDEN = _bool_env("UPDATE_GOLDEN", False)


