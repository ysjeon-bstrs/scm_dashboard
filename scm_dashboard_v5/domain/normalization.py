"""Data normalization helpers for the SCM dashboard domain objects."""

from __future__ import annotations

from typing import Iterable

import pandas as pd

DATE_COLUMNS = ("onboard_date", "arrival_date", "inbound_date", "event_date")


def normalize_dates(frame: pd.DataFrame, *, columns: Iterable[str] = DATE_COLUMNS) -> pd.DataFrame:
    """Return a copy of *frame* with the specified date columns normalised to midnight."""

    out = frame.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce").dt.normalize()
    return out


def normalize_moves(frame: pd.DataFrame) -> pd.DataFrame:
    """Coerce core move columns to predictable dtypes."""

    out = normalize_dates(frame)
    out["carrier_mode"] = out.get("carrier_mode", "").astype(str).str.upper()
    out["resource_code"] = out.get("resource_code", "").astype(str)
    out["from_center"] = out.get("from_center", "").astype(str)
    out["to_center"] = out.get("to_center", "").astype(str)
    out["qty_ea"] = pd.to_numeric(out.get("qty_ea", 0), errors="coerce").fillna(0)
    return out


def normalize_snapshot(frame: pd.DataFrame) -> pd.DataFrame:
    """Standardise snapshot schema for downstream consumers."""

    out = frame.copy()
    date_col = None
    for candidate in ("date", "snapshot_date"):
        if candidate in out.columns:
            date_col = candidate
            break
    if not date_col:
        raise KeyError("snapshot frame must include a 'date' or 'snapshot_date' column")

    out["date"] = pd.to_datetime(out[date_col], errors="coerce").dt.normalize()
    out["center"] = out.get("center", "").astype(str)
    out["resource_code"] = out.get("resource_code", "").astype(str)
    out["stock_qty"] = pd.to_numeric(out.get("stock_qty", 0), errors="coerce")
    out = out.dropna(subset=["date"])
    return out[["date", "center", "resource_code", "stock_qty"]]
