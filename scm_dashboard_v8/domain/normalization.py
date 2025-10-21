"""도메인 데이터 정규화 유틸리티."""

from __future__ import annotations

from typing import Iterable

import pandas as pd

DATE_COLUMNS = ("onboard_date", "arrival_date", "inbound_date", "event_date")


def normalize_dates(frame: pd.DataFrame, *, columns: Iterable[str] = DATE_COLUMNS) -> pd.DataFrame:
    """지정된 날짜 컬럼을 자정 기준으로 정규화한 DataFrame을 반환한다."""

    out = frame.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce").dt.normalize()
    return out


def normalize_moves(frame: pd.DataFrame) -> pd.DataFrame:
    """move 데이터의 핵심 컬럼 타입을 안정적으로 보정한다."""

    out = normalize_dates(frame)

    # ✅ 문자열 컬럼은 공백 대비를 위해 문자열형으로 통일한다.
    carrier_src = out["carrier_mode"] if "carrier_mode" in out.columns else pd.Series("", index=out.index)
    out["carrier_mode"] = carrier_src.astype(str).str.upper()

    resource_src = out["resource_code"] if "resource_code" in out.columns else pd.Series("", index=out.index)
    out["resource_code"] = resource_src.astype(str)

    from_src = out["from_center"] if "from_center" in out.columns else pd.Series("", index=out.index)
    out["from_center"] = from_src.astype(str)

    to_src = out["to_center"] if "to_center" in out.columns else pd.Series("", index=out.index)
    out["to_center"] = to_src.astype(str)

    qty_src = out["qty_ea"] if "qty_ea" in out.columns else pd.Series(0, index=out.index)
    out["qty_ea"] = pd.to_numeric(qty_src, errors="coerce").fillna(0)

    return out


def normalize_snapshot(frame: pd.DataFrame) -> pd.DataFrame:
    """스냅샷 데이터를 date/center/resource_code/stock_qty 컬럼으로 정규화한다."""

    out = frame.copy()
    date_col = None
    for candidate in ("date", "snapshot_date"):
        if candidate in out.columns:
            date_col = candidate
            break
    if not date_col:
        raise KeyError("snapshot 데이터에는 'date' 또는 'snapshot_date' 컬럼이 필요합니다.")

    out["date"] = pd.to_datetime(out[date_col], errors="coerce").dt.normalize()

    center_src = out["center"] if "center" in out.columns else pd.Series("", index=out.index)
    out["center"] = center_src.astype(str)

    resource_src = out["resource_code"] if "resource_code" in out.columns else pd.Series("", index=out.index)
    out["resource_code"] = resource_src.astype(str)

    stock_src = out["stock_qty"] if "stock_qty" in out.columns else pd.Series(0, index=out.index)
    out["stock_qty"] = pd.to_numeric(stock_src, errors="coerce")

    out = out.dropna(subset=["date"])
    return out[["date", "center", "resource_code", "stock_qty"]]
