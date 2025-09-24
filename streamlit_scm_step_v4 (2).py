
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import re
from datetime import datetime, timedelta
from io import BytesIO

st.set_page_config(page_title="SCM Step Dashboard v4", layout="wide")
st.title("📦 센터×SKU 재고 흐름 (계단식) 대시보드 — v4")

def _coalesce_columns(df, candidates, parse_date=False):
    all_names = []
    for item in candidates:
        if isinstance(item, (list, tuple, set)):
            all_names.extend([str(x).strip() for x in item])
        else:
            all_names.append(str(item).strip())
    cols = [c for c in df.columns if str(c).strip() in all_names]
    if not cols:
        return pd.Series(pd.NaT if parse_date else np.nan, index=df.index)
    sub = df[cols].copy()
    if parse_date:
        for c in cols:
            sub[c] = pd.to_datetime(sub[c], errors="coerce")
    out = sub.bfill(axis=1).iloc[:, 0]
    return out

@st.cache_data(ttl=300)
def load_from_excel(file):
    data = file.read() if hasattr(file, "read") else file
    bio = BytesIO(data) if isinstance(data, (bytes, bytearray)) else file
    xl = pd.ExcelFile(bio, engine="openpyxl")
    df_move = pd.read_excel(bio, sheet_name="SCM_통합", engine="openpyxl"); bio.seek(0)
    df_snap = pd.read_excel(bio, sheet_name="sample_snapshot", engine="openpyxl"); bio.seek(0)
    wip_df = None
    if "입고예정내역" in xl.sheet_names:
        wip_df = pd.read_excel(bio, sheet_name="입고예정내역", engine="openpyxl")
    return df_move, df_snap, wip_df

def normalize_snapshot(df_snap):
    df = df_snap.copy(); df.columns = [str(c).strip() for c in df.columns]
    rename_map = {}
    for c in df.columns:
        cs = str(c)
        if ("스냅샷" in cs and "일자" in cs) or cs.lower() in ["snapshot_date","date"]:
            rename_map[c] = "snapshot_date"
        elif cs.strip() in ["창고명","center"]:
            rename_map[c] = "center"
    df = df.rename(columns=rename_map)
    id_cols = ["snapshot_date", "center"]
    sku_cols = [c for c in df.columns if c not in id_cols and "스냅샷" not in str(c)]
    longy = df.melt(id_vars=id_cols, value_vars=sku_cols, var_name="resource_code", value_name="stock_qty")
    longy["snapshot_date"] = pd.to_datetime(longy["snapshot_date"], errors="coerce")
    longy["stock_qty"] = pd.to_numeric(longy["stock_qty"], errors="coerce").fillna(0).astype(int)
    return longy.dropna(subset=["snapshot_date","center","resource_code"])

def normalize_moves(df):
    df = df.copy(); df.columns = [str(c).strip() for c in df.columns]
    out = pd.DataFrame({
        "resource_code": _coalesce_columns(df, [["resource_code","상품코드","RESOURCE_CODE"]]).astype(str).str.strip(),
        "qty_ea": pd.to_numeric(_coalesce_columns(df, [["qty_ea","QTY_EA","수량(EA)"]]), errors="coerce").fillna(0).astype(int),
        "carrier_mode": _coalesce_columns(df, [["carrier_mode","운송방법","carrier mode"]]).astype(str).str.strip(),
        "from_center": _coalesce_columns(df, [["from_center","출발창고"]]).astype(str).str.strip(),
        "to_center": _coalesce_columns(df, [["to_center","도착창고"]]).astype(str).str.strip(),
        "onboard_date": _coalesce_columns(df, [["onboard_date","배정일","출발일","H"]], parse_date=True),
        "arrival_date": _coalesce_columns(df, [["arrival_date","도착일","eta_date","ETA"]], parse_date=True),
        "inbound_date": _coalesce_columns(df, [["inbound_date","입고일"]], parse_date=True),
    })
    out["event_date"] = out["inbound_date"].where(out["inbound_date"].notna(), out["arrival_date"])
    return out

def _parse_po_date(po_str):
    if not isinstance(po_str, str): return pd.NaT
    m = re.search(r"[A-Za-z](\d{2})(\d{2})(\d{2})", po_str); 
    if not m: return pd.NaT
    yy, mm, dd = m.groups(); year = 2000 + int(yy)
    try: return pd.Timestamp(datetime(year, int(mm), int(dd)))
    except: return pd.NaT

def load_wip_from_incoming(df_incoming, default_center="태광KR"):
    if df_incoming is None or df_incoming.empty: return pd.DataFrame()
    df = df_incoming.copy(); df.columns = [str(c).strip().lower() for c in df.columns]
    po_col   = next((c for c in df.columns if c in ["po_no","ponumber","po"]), None)
    date_col = next((c for c in df.columns if "intended_push_date" in c or "입고" in c), None)
    sku_col  = next((c for c in df.columns if c in ["product_code","resource_code","상품코드"]), None)
    qty_col  = next((c for c in df.columns if c in ["quantity","qty","수량","total_quantity"]), None)
    lot_col  = next((c for c in df.columns if c in ["lot","제조번호","lot_no","lotnumber"]), None)
    if not date_col or not sku_col or not qty_col:
        return pd.DataFrame()
    out = pd.DataFrame({
        "resource_code": df[sku_col].astype(str).str.strip(),
        "to_center": default_center,
        "wip_ready": pd.to_datetime(df[date_col], errors="coerce"),
        "qty_ea": pd.to_numeric(df[qty_col], errors="coerce").fillna(0).astype(int),
        "lot": df[lot_col].astype(str).str.strip() if lot_col else ""
    })
    out["wip_start"] = df[po_col].map(_parse_po_date) if po_col else pd.NaT
    mask_na = out["wip_start"].isna() & out["wip_ready"].notna()
    out.loc[mask_na, "wip_start"] = out.loc[mask_na, "wip_ready"] - pd.to_timedelta(10, unit="D")
    return out.dropna(subset=["resource_code","wip_ready","wip_start"])[["resource_code","to_center","wip_start","wip_ready","qty_ea","lot"]]

def merge_wip_as_moves(moves_df, wip_df):
    if wip_df is None or wip_df.empty: return moves_df
    wip_moves = pd.DataFrame({
        "resource_code": wip_df["resource_code"],
        "qty_ea": wip_df["qty_ea"].astype(int),
        "carrier_mode": "WIP",
        "from_center": "WIP",
        "to_center": wip_df["to_center"],
        "onboard_date": wip_df["wip_start"],
        "arrival_date": wip_df["wip_ready"],
        "inbound_date": pd.NaT,
        "event_date": wip_df["wip_ready"],
        "lot": wip_df.get("lot","")
    })
    return pd.concat([moves_df, wip_moves], ignore_index=True)

# --- App body omitted for brevity; for this task we only need WIP lot propagation and table columns ---
st.success("코드에 WIP lot(제조번호) 지원을 추가했습니다. 실제 본문은 기존 v4 코드에 이 함수들을 병합해 사용하세요.")
