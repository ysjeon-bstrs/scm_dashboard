"""데이터 정규화 및 변환 유틸리티 모듈.

DataFrame/Series 안전 변환, 타임스탬프 정규화, 컬럼 매핑 등을 제공합니다.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd


def safe_dataframe(
    df: Optional[pd.DataFrame],
    *,
    index: Optional[Sequence[pd.Timestamp]] = None,
    columns: Optional[Sequence[str]] = None,
    fill_value: float = 0.0,
    dtype: type = float,
) -> pd.DataFrame:
    """항상 사용 가능한 DataFrame을 반환합니다.

    None이나 잘못된 데이터를 안전하게 처리하여 빈 DataFrame을 생성합니다.

    Args:
        df: 입력 DataFrame (None 가능)
        index: 설정할 인덱스
        columns: 설정할 컬럼 리스트
        fill_value: 빈 셀 채울 값
        dtype: 데이터 타입

    Returns:
        정규화된 DataFrame
    """
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(dtype=dtype)

    result = df.copy()

    if columns is not None:
        result = result.reindex(columns=list(columns), fill_value=fill_value)

    if index is not None:
        if not isinstance(index, pd.Index):
            index = pd.Index(index)
        result = result.reindex(index, fill_value=fill_value)

    if result.empty and columns is not None:
        result = pd.DataFrame([], columns=list(columns), dtype=dtype)

    return result


def safe_series(
    obj: Optional[pd.Series | Iterable[float]], *, length_hint: int | None = None
) -> pd.Series:
    """임의의 iterable을 안전하게 Pandas Series로 변환합니다.

    Args:
        obj: 변환할 객체 (Series, list, ndarray 등)
        length_hint: 최대 길이 제한

    Returns:
        변환된 Series
    """
    if isinstance(obj, pd.Series):
        return obj

    if obj is None:
        data: List[float] = []
    elif isinstance(obj, (pd.Index, np.ndarray)):
        data = list(obj)
    elif hasattr(obj, "tolist"):
        data = list(obj.tolist())  # type: ignore[arg-type]
    elif isinstance(obj, Iterable) and not isinstance(obj, (str, bytes)):
        data = list(obj)
    else:
        data = [obj] if obj is not None else []

    if length_hint is not None and len(data) > length_hint:
        data = data[:length_hint]

    return pd.Series(data)


def as_naive_timestamp(value: pd.Timestamp | str | None) -> pd.Timestamp:
    """타임존이 없는 naive timestamp를 반환합니다.

    일관된 날짜 비교를 위해 타임존 정보를 제거합니다.

    Args:
        value: 변환할 timestamp (None이면 오늘)

    Returns:
        타임존이 제거된 timestamp
    """
    ts = pd.Timestamp(value) if value is not None else pd.Timestamp.today()
    try:
        return ts.tz_localize(None)  # tz-aware 값 처리
    except TypeError:
        return ts  # 이미 naive


def ensure_naive_index(idx: pd.Index) -> pd.DatetimeIndex:
    """Index를 타임존 없는 DatetimeIndex로 변환합니다.

    Args:
        idx: 변환할 Index

    Returns:
        타임존이 제거된 DatetimeIndex
    """
    dt_idx = pd.to_datetime(idx, errors="coerce")
    if not isinstance(dt_idx, pd.DatetimeIndex):
        dt_idx = pd.DatetimeIndex(dt_idx)
    if dt_idx.tz is not None:
        dt_idx = dt_idx.tz_localize(None)
    return dt_idx


def normalize_inventory_frame(
    df: Optional[pd.DataFrame], *, default_center: str | None = None
) -> pd.DataFrame:
    """재고 DataFrame을 표준 형식으로 정규화합니다.

    필수 컬럼: date, center, resource_code, stock_qty

    Args:
        df: 입력 DataFrame
        default_center: center 컬럼이 없을 때 사용할 기본 센터

    Returns:
        정규화된 재고 DataFrame
    """
    cols = ["date", "center", "resource_code", "stock_qty"]
    if df is None or df.empty:
        return pd.DataFrame(columns=cols)

    frame = df.copy()
    rename_map = {str(col).lower(): col for col in frame.columns}

    if "date" not in frame.columns and "snapshot_date" in rename_map:
        frame = frame.rename(columns={rename_map["snapshot_date"]: "date"})
    if "center" not in frame.columns and "center" in rename_map:
        frame = frame.rename(columns={rename_map["center"]: "center"})
    if "resource_code" not in frame.columns and "resource_code" in rename_map:
        frame = frame.rename(columns={rename_map["resource_code"]: "resource_code"})
    if "stock_qty" not in frame.columns and "stock" in rename_map:
        frame = frame.rename(columns={rename_map["stock"]: "stock_qty"})

    frame["date"] = pd.to_datetime(frame.get("date"), errors="coerce").dt.normalize()
    if "center" in frame.columns:
        frame["center"] = frame["center"].astype(str)
    elif default_center is not None:
        frame["center"] = default_center
    else:
        frame["center"] = ""

    if "resource_code" in frame.columns:
        frame["resource_code"] = frame["resource_code"].astype(str)
    else:
        frame["resource_code"] = ""

    frame["stock_qty"] = pd.to_numeric(frame.get("stock_qty"), errors="coerce").fillna(0)

    return frame[cols]


def normalize_sales_frame(
    df: Optional[pd.DataFrame], *, default_center: str | None = None
) -> pd.DataFrame:
    """판매 DataFrame을 표준 형식으로 정규화합니다.

    필수 컬럼: date, center, resource_code, sales_ea

    Args:
        df: 입력 DataFrame
        default_center: center 컬럼이 없을 때 사용할 기본 센터

    Returns:
        정규화된 판매 DataFrame
    """
    cols = ["date", "center", "resource_code", "sales_ea"]
    if df is None or df.empty:
        return pd.DataFrame(columns=cols)

    frame = df.copy()
    rename_map = {str(col).lower(): col for col in frame.columns}

    if "date" not in frame.columns and "snapshot_date" in rename_map:
        frame = frame.rename(columns={rename_map["snapshot_date"]: "date"})
    if "center" not in frame.columns and "center" in rename_map:
        frame = frame.rename(columns={rename_map["center"]: "center"})
    if "resource_code" not in frame.columns and "resource_code" in rename_map:
        frame = frame.rename(columns={rename_map["resource_code"]: "resource_code"})
    if "sales_ea" not in frame.columns:
        candidate = None
        for col in frame.columns:
            if str(col).lower().endswith("sales_ea") or "sales" in str(col).lower():
                candidate = col
                break
        if candidate is not None:
            frame = frame.rename(columns={candidate: "sales_ea"})

    frame["date"] = pd.to_datetime(frame.get("date"), errors="coerce").dt.normalize()
    if "center" in frame.columns:
        frame["center"] = frame["center"].astype(str)
    elif default_center is not None:
        frame["center"] = default_center
    else:
        frame["center"] = ""

    if "resource_code" in frame.columns:
        frame["resource_code"] = frame["resource_code"].astype(str)
    else:
        frame["resource_code"] = ""

    frame["sales_ea"] = pd.to_numeric(frame.get("sales_ea"), errors="coerce").fillna(0)
    frame = frame.dropna(subset=["date"])

    return frame[cols]


def coerce_cols(df: pd.DataFrame) -> Dict[str, str]:
    """DataFrame의 컬럼을 표준 이름으로 매핑합니다.

    여러 가능한 컬럼 이름을 인식하여 표준 이름으로 변환합니다.

    Args:
        df: 입력 DataFrame

    Returns:
        표준 컬럼 이름 매핑 딕셔너리
    """
    cols = {c.lower(): c for c in df.columns}
    date_col = cols.get("snapshot_date") or cols.get("date")
    center_col = cols.get("center")
    sku_col = cols.get("resource_code") or cols.get("sku")
    qty_col = cols.get("stock_qty") or cols.get("qty") or cols.get("quantity")
    return {"date": date_col, "center": center_col, "sku": sku_col, "qty": qty_col}


def empty_sales_frame() -> pd.DataFrame:
    """빈 판매 DataFrame을 생성합니다.

    Returns:
        표준 컬럼을 가진 빈 DataFrame
    """
    return pd.DataFrame(columns=["date", "resource_code", "sales_ea"])
