"""Data preparation helpers for the SCM dashboard."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
import streamlit as st


def _flatten_candidates(candidates: Iterable) -> List[str]:
    names: List[str] = []
    for item in candidates:
        if isinstance(item, (list, tuple, set)):
            names.extend([str(x).strip() for x in item])
        else:
            names.append(str(item).strip())
    return names


def coalesce_columns(df: pd.DataFrame, candidates: Iterable, parse_date: bool = False) -> pd.Series:
    """Return the first available column from *candidates* in *df*."""
    all_names = _flatten_candidates(candidates)

    cols = [c for c in df.columns if str(c).strip() in all_names]
    if not cols:
        cols = [c for c in df.columns if any(name.lower() in str(c).lower() for name in all_names)]
    if not cols:
        cols = [
            c
            for c in df.columns
            if any(name.lower() in str(c).lower() or str(c).lower() in name.lower() for name in all_names)
        ]

    if not cols:
        return pd.Series(pd.NaT if parse_date else np.nan, index=df.index)

    sub = df[cols].copy()
    if parse_date:
        for c in cols:
            sub[c] = pd.to_datetime(sub[c], errors="coerce")
    return sub.bfill(axis=1).iloc[:, 0]


@st.cache_data(ttl=300)
def normalize_moves(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip() for c in df.columns]

    resource_code = coalesce_columns(df, [["resource_code", "상품코드", "RESOURCE_CODE", "sku", "SKU"]])
    qty_ea = coalesce_columns(df, [["qty_ea", "QTY_EA", "수량(EA)", "qty", "QTY", "quantity", "Quantity", "수량", "EA", "ea"]])
    carrier_mode = coalesce_columns(df, [["carrier_mode", "운송방법", "carrier mode", "운송수단"]])
    from_center = coalesce_columns(df, [["from_center", "출발창고", "from center"]])
    to_center = coalesce_columns(df, [["to_center", "도착창고", "to center"]])
    onboard_date = coalesce_columns(
        df,
        [["onboard_date", "배정일", "출발일", "H", "onboard", "depart_date"]],
        parse_date=True,
    )
    arrival_date = coalesce_columns(
        df,
        [["arrival_date", "도착일", "eta_date", "ETA", "arrival"]],
        parse_date=True,
    )
    inbound_date = coalesce_columns(df, [["inbound_date", "입고일", "입고완료일"]], parse_date=True)

    out = pd.DataFrame(
        {
            "resource_code": resource_code.astype(str).str.strip(),
            "qty_ea": pd.to_numeric(qty_ea.astype(str).str.replace(",", ""), errors="coerce").fillna(0).astype(int),
            "carrier_mode": carrier_mode.astype(str).str.strip(),
            "from_center": from_center.astype(str).str.strip(),
            "to_center": to_center.astype(str).str.strip(),
            "onboard_date": onboard_date,
            "arrival_date": arrival_date,
            "inbound_date": inbound_date,
        }
    )
    out["event_date"] = out["inbound_date"].where(out["inbound_date"].notna(), out["arrival_date"])
    for col in ["onboard_date", "arrival_date", "inbound_date", "event_date"]:
        out[col] = pd.to_datetime(out[col], errors="coerce").dt.normalize()
    return out


@st.cache_data(ttl=300)
def normalize_refined_snapshot(df_ref: pd.DataFrame) -> pd.DataFrame:
    cols = {str(c).strip().lower(): c for c in df_ref.columns}

    date_col = next((cols[k] for k in ["date", "날짜", "snapshot_date", "스냅샷일"] if k in cols), None)
    center_col = next((cols[k] for k in ["center", "센터", "창고", "warehouse"] if k in cols), None)
    resource_col = next((cols[k] for k in ["resource_code", "resource_cc", "sku", "상품코드", "product_code"] if k in cols), None)
    stock_col = next((cols[k] for k in ["stock_qty", "qty", "수량", "재고", "quantity"] if k in cols), None)
    name_col = next((cols[k] for k in ["resource_name", "품명", "상품명", "product_name"] if k in cols), None)

    missing = [
        n
        for n, v in {
            "date": date_col,
            "center": center_col,
            "resource_code": resource_col,
            "stock_qty": stock_col,
        }.items()
        if not v
    ]
    if missing:
        st.error(f"'snap_정제' 시트에 누락된 컬럼: {missing}")
        st.stop()

    result = df_ref.rename(
        columns={
            date_col: "date",
            center_col: "center",
            resource_col: "resource_code",
            stock_col: "stock_qty",
            **({name_col: "resource_name"} if name_col else {}),
        }
    ).copy()

    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.normalize()
    result["center"] = result["center"].astype(str)
    result["resource_code"] = result["resource_code"].astype(str)
    result["stock_qty"] = pd.to_numeric(result["stock_qty"], errors="coerce").fillna(0).astype(int)

    if "resource_name" in result.columns:
        result["resource_name"] = result["resource_name"].astype(str).str.strip().replace({"nan": "", "None": ""})

    return result.dropna(subset=["date", "center", "resource_code"])


def _parse_po_date(po_str: str) -> pd.Timestamp:
    if not isinstance(po_str, str):
        return pd.NaT
    match = re.search(r"[A-Za-z](\d{2})(\d{2})(\d{2})", po_str)
    if not match:
        return pd.NaT
    yy, mm, dd = match.groups()
    year = 2000 + int(yy)
    try:
        return pd.Timestamp(datetime(year, int(mm), int(dd)))
    except Exception:
        return pd.NaT


def load_wip_from_incoming(df_incoming: Optional[pd.DataFrame], default_center: str = "태광KR") -> pd.DataFrame:
    if df_incoming is None or df_incoming.empty:
        return pd.DataFrame()

    df_incoming.columns = [str(c).strip().lower() for c in df_incoming.columns]
    po_col = next((c for c in df_incoming.columns if c in ["po_no", "ponumber", "po"]), None)
    date_col = next((c for c in df_incoming.columns if "intended_push_date" in c or "입고" in c), None)
    sku_col = next((c for c in df_incoming.columns if c in ["product_code", "resource_code", "상품코드"]), None)
    qty_col = next((c for c in df_incoming.columns if c in ["quantity", "qty", "수량", "total_quantity"]), None)
    lot_col = next((c for c in df_incoming.columns if c in ["lot", "제조번호", "lot_no", "lotnumber"]), None)

    if not date_col or not sku_col or not qty_col:
        return pd.DataFrame()

    out = pd.DataFrame(
        {
            "resource_code": df_incoming[sku_col].astype(str).str.strip(),
            "to_center": default_center,
            "wip_ready": pd.to_datetime(df_incoming[date_col], errors="coerce"),
            "qty_ea": pd.to_numeric(df_incoming[qty_col].astype(str).str.replace(",", ""), errors="coerce").fillna(0).astype(int),
            "lot": df_incoming[lot_col].astype(str).str.strip() if lot_col else "",
        }
    )
    out["wip_start"] = df_incoming[po_col].map(_parse_po_date) if po_col else pd.NaT
    mask_na = out["wip_start"].isna() & out["wip_ready"].notna()
    out.loc[mask_na, "wip_start"] = out.loc[mask_na, "wip_ready"] - pd.to_timedelta(30, unit="D")

    out = out.dropna(subset=["resource_code", "wip_ready", "wip_start"]).reset_index(drop=True)
    return out[["resource_code", "to_center", "wip_start", "wip_ready", "qty_ea", "lot"]]


def merge_wip_as_moves(moves_df: pd.DataFrame, wip_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if wip_df is None or wip_df.empty:
        return moves_df
    wip_df_norm = wip_df.copy()
    wip_df_norm["wip_start"] = pd.to_datetime(wip_df_norm["wip_start"], errors="coerce").dt.normalize()
    wip_df_norm["wip_ready"] = pd.to_datetime(wip_df_norm["wip_ready"], errors="coerce").dt.normalize()
    wip_moves = pd.DataFrame(
        {
            "resource_code": wip_df_norm["resource_code"],
            "qty_ea": wip_df_norm["qty_ea"].astype(int),
            "carrier_mode": "WIP",
            "from_center": "WIP",
            "to_center": wip_df_norm["to_center"],
            "onboard_date": wip_df_norm["wip_start"],
            "arrival_date": wip_df_norm["wip_ready"],
            "inbound_date": pd.NaT,
            "event_date": wip_df_norm["wip_ready"],
            "lot": wip_df_norm.get("lot", ""),
        }
    )
    for col in ["onboard_date", "arrival_date", "event_date"]:
        wip_moves[col] = pd.to_datetime(wip_moves[col], errors="coerce").dt.normalize()
    return pd.concat([moves_df, wip_moves], ignore_index=True)


def normalize_center_name(center: str) -> Optional[str]:
    if center in ["", "nan", "None", "WIP", "In-Transit"]:
        return None
    if center in ["AcrossBUS", "어크로스비US"]:
        return "어크로스비US"
    return center
