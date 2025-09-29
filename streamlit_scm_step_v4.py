# dashboard_v4_clean.py
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import re
from datetime import datetime
from io import BytesIO
from typing import Dict, List, Optional, Tuple
import requests
from urllib.parse import quote

# =========================
# App setup
# =========================
st.set_page_config(page_title="SCM 재고 흐름 대시보드 — v4 (clean)", layout="wide")
st.title("📦 SCM 재고 흐름 대시보드 — v4 (clean)")

# 공개/비공개 어디서든 동작하도록: 실패는 조용히, 요청 시에만 자세히
GSHEET_ID = "1RYjKW2UDJ2kWJLAqQH26eqx2-r9Xb0_qE_hfwu9WIj8"

CENTER_COL = {
    "태광KR": "stock2",
    "AMZUS": "fba_available_stock",
    "품고KR": "poomgo_v2_available_stock",
    "SBSPH": "shopee_ph_available_stock",
    "SBSSG": "shopee_sg_available_stock",
    "SBSMY": "shopee_my_available_stock",
    "AcrossBUS": "acrossb_available_stock",
    "어크로스비US": "acrossb_available_stock",
}

# -------------------- Helpers --------------------
def _coalesce_columns(df: pd.DataFrame, candidates: List, parse_date: bool = False) -> pd.Series:
    all_names = []
    for item in candidates:
        if isinstance(item, (list, tuple, set)):
            all_names.extend([str(x).strip() for x in item])
        else:
            all_names.append(str(item).strip())
    cols = [c for c in df.columns if str(c).strip() in all_names]
    if not cols:
        cols = [c for c in df.columns if any(name.lower() in str(c).lower() for name in all_names)]
    if not cols:
        cols = [c for c in df.columns if any(name.lower() in str(c).lower() or str(c).lower() in name.lower() for name in all_names)]
    if not cols:
        return pd.Series(pd.NaT if parse_date else np.nan, index=df.index)
    sub = df[cols].copy()
    if parse_date:
        for c in cols:
            sub[c] = pd.to_datetime(sub[c], errors="coerce")
    return sub.bfill(axis=1).iloc[:, 0]

def gs_csv_public(sheet_name: str, noisy: bool=False) -> pd.DataFrame:
    """공개(Anyone with the link) 시트를 gviz CSV로 읽음. 실패 시 빈 DF(+옵션 에러 표시)."""
    try:
        url = f"https://docs.google.com/spreadsheets/d/{GSHEET_ID}/gviz/tq?tqx=out:csv&sheet={quote(sheet_name)}"
        return pd.read_csv(url)
    except Exception as e:
        if noisy:
            st.error(f"Google Sheets(gviz) 로드 실패: {e}")
            st.info("▶ 시트 공개 상태(Anyone with the link), 시트명(탭명), 헤더 1행 여부를 확인하세요.")
        return pd.DataFrame()

# -------------------- Loaders --------------------
@st.cache_data(ttl=300)
def load_from_excel(file):
    data = file.read() if hasattr(file, "read") else file
    bio = BytesIO(data) if isinstance(data, (bytes, bytearray)) else file
    xl = pd.ExcelFile(bio, engine="openpyxl")

    if "SCM_통합" not in xl.sheet_names:
        st.error("엑셀에 'SCM_통합' 시트가 필요합니다.")
        st.stop()
    df_move = pd.read_excel(bio, sheet_name="SCM_통합", engine="openpyxl")
    bio.seek(0)

    refined = next((s for s in xl.sheet_names if s in ["snap_정제","snap_refined","snap_refine","snap_ref"]), None)
    if refined is None:
        st.error("엑셀에 'snap_정제' 시트가 필요합니다.")
        st.stop()
    df_ref = pd.read_excel(bio, sheet_name=refined, engine="openpyxl")
    bio.seek(0)

    df_incoming = pd.read_excel(bio, sheet_name="입고예정내역", engine="openpyxl") if "입고예정내역" in xl.sheet_names else None
    bio.seek(0)

    df_snapraw = pd.read_excel(bio, sheet_name="snapshot_raw", engine="openpyxl") if "snapshot_raw" in xl.sheet_names else None
    if df_snapraw is not None and not df_snapraw.empty:
        st.session_state["snapshot_raw_df"] = df_snapraw
    else:
        st.session_state.pop("snapshot_raw_df", None)

    return df_move, df_ref, df_incoming, df_snapraw

@st.cache_data(ttl=300)
def normalize_refined_snapshot(df_ref: pd.DataFrame) -> pd.DataFrame:
    cols = {c.strip().lower(): c for c in df_ref.columns}
    date_col = next((cols[k] for k in ["date","날짜","snapshot_date","스냅샷일"] if k in cols), None)
    center_col = next((cols[k] for k in ["center","센터","창고","warehouse"] if k in cols), None)
    sku_col = next((cols[k] for k in ["resource_code","sku","상품코드","product_code"] if k in cols), None)
    qty_col = next((cols[k] for k in ["stock_qty","qty","수량","재고","quantity"] if k in cols), None)
    missing = [n for n,v in {"date":date_col,"center":center_col,"resource_code":sku_col,"stock_qty":qty_col}.items() if not v]
    if missing:
        st.error(f"'snap_정제' 시트에 누락된 컬럼: {missing}")
        st.stop()

    sr = df_ref.rename(columns={date_col:"date", center_col:"center", sku_col:"resource_code", qty_col:"stock_qty"}).copy()
    sr["date"] = pd.to_datetime(sr["date"], errors="coerce").dt.normalize()
    sr["center"] = sr["center"].astype(str)
    sr["resource_code"] = sr["resource_code"].astype(str)
    sr["stock_qty"] = pd.to_numeric(sr["stock_qty"], errors="coerce").fillna(0).astype(int)
    return sr.dropna(subset=["date","center","resource_code"])

def normalize_moves(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip() for c in df.columns]
    resource_code = _coalesce_columns(df, [["resource_code","상품코드","RESOURCE_CODE","sku","SKU"]])
    qty_ea       = _coalesce_columns(df, [["qty_ea","QTY_EA","수량(EA)","qty","QTY","quantity","Quantity","수량","EA","ea"]])
    carrier_mode = _coalesce_columns(df, [["carrier_mode","운송방법","carrier mode","운송수단"]])
    from_center  = _coalesce_columns(df, [["from_center","출발창고","from center"]])
    to_center    = _coalesce_columns(df, [["to_center","도착창고","to center"]])
    onboard_date = _coalesce_columns(df, [["onboard_date","배정일","출발일","H","onboard","depart_date"]], parse_date=True)
    arrival_date = _coalesce_columns(df, [["arrival_date","도착일","eta_date","ETA","arrival"]], parse_date=True)
    inbound_date = _coalesce_columns(df, [["inbound_date","입고일","입고완료일"]], parse_date=True)
    real_depart  = _coalesce_columns(df, [["실제 선적일","real_departure","AI"]], parse_date=True)
    out = pd.DataFrame({
        "resource_code": resource_code.astype(str).str.strip(),
        "qty_ea": pd.to_numeric(qty_ea.astype(str).str.replace(',',''), errors="coerce").fillna(0).astype(int),
        "carrier_mode": carrier_mode.astype(str).str.strip(),
        "from_center": from_center.astype(str).str.strip(),
        "to_center": to_center.astype(str).str.strip(),
        "onboard_date": onboard_date,
        "arrival_date": arrival_date,
        "inbound_date": inbound_date,
        "real_departure": real_depart,
    })
    out["event_date"] = out["inbound_date"].where(out["inbound_date"].notna(), out["arrival_date"])
    return out

def _parse_po_date(po_str: str) -> pd.Timestamp:
    if not isinstance(po_str, str): return pd.NaT
    m = re.search(r"[A-Za-z](\d{2})(\d{2})(\d{2})", po_str)
    if not m: return pd.NaT
    yy, mm, dd = m.groups()
    try:
        return pd.Timestamp(datetime(2000+int(yy), int(mm), int(dd)))
    except Exception:
        return pd.NaT

def load_wip_from_incoming(df_incoming: Optional[pd.DataFrame], default_center: str = "태광KR") -> pd.DataFrame:
    if df_incoming is None or df_incoming.empty:
        return pd.DataFrame()
    df_incoming.columns = [str(c).strip().lower() for c in df_incoming.columns]
    po_col   = next((c for c in df_incoming.columns if c in ["po_no","ponumber","po"]), None)
    date_col = next((c for c in df_incoming.columns if "intended_push_date" in c or "입고" in c), None)
    sku_col  = next((c for c in df_incoming.columns if c in ["product_code","resource_code","상품코드"]), None)
    qty_col  = next((c for c in df_incoming.columns if c in ["quantity","qty","수량","total_quantity"]), None)
    lot_col  = next((c for c in df_incoming.columns if c in ["lot","제조번호","lot_no","lotnumber"]), None)
    if not date_col or not sku_col or not qty_col:
        return pd.DataFrame()
    out = pd.DataFrame({
        "resource_code": df_incoming[sku_col].astype(str).str.strip(),
        "to_center": default_center,
        "wip_ready": pd.to_datetime(df_incoming[date_col], errors="coerce"),
        "qty_ea": pd.to_numeric(df_incoming[qty_col].astype(str).str.replace(',',''), errors="coerce").fillna(0).astype(int),
        "lot": df_incoming[lot_col].astype(str).str.strip() if lot_col else ""
    })
    out["wip_start"] = df_incoming[po_col].map(_parse_po_date) if po_col else pd.NaT
    m = out["wip_start"].isna() & out["wip_ready"].notna()
    out.loc[m, "wip_start"] = out.loc[m, "wip_ready"] - pd.to_timedelta(30, unit="D")
    return out.dropna(subset=["resource_code","wip_ready","wip_start"])[["resource_code","to_center","wip_start","wip_ready","qty_ea","lot"]]

def merge_wip_as_moves(moves_df: pd.DataFrame, wip_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if wip_df is None or wip_df.empty:
        return moves_df
    wip_moves = pd.DataFrame({
        "resource_code": wip_df["resource_code"],
        "qty_ea": wip_df["qty_ea"].astype(int),
        "carrier_mode": "WIP",
        "from_center": "WIP",
        "to_center": wip_df["to_center"],
        "onboard_date": wip_df["wip_start"],
        "arrival_date": wip_df["wip_ready"],
        "inbound_date": pd.NaT,
        "real_departure": pd.NaT,
        "event_date": wip_df["wip_ready"],
        "lot": wip_df.get("lot", "")
    })
    return pd.concat([moves_df, wip_moves], ignore_index=True)

# ===== 이후 Timeline, Forecast, KPI, 차트, 입고 예정 내역, 재고자산, LOT 상세 섹션은 동일하게 이어집니다 =====
# (답변 길이 제한으로 여기서 끊지만, 제가 이전 메시지에 드린 `dashboard_v4_clean.py` 전체 코드 블록과 완전히 동일합니다.)
