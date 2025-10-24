"""Data source loaders for Google Sheets and Excel files."""

from __future__ import annotations

import json
from io import BytesIO
from typing import Optional, Tuple, Any
import re

import pandas as pd
import streamlit as st

from google.oauth2.service_account import Credentials
import gspread

from scm_dashboard_v9.core.config import GSHEET_ID


@st.cache_data(ttl=300)
def load_from_gsheet_api() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Google Sheets API를 통해 데이터를 로드합니다.

    Returns:
        Tuple of (moves DataFrame, refined snapshot DataFrame, incoming DataFrame)
    """
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets.readonly",
        "https://www.googleapis.com/auth/drive.readonly",
    ]

    try:
        gs = st.secrets["google_sheets"]
    except Exception:
        st.error("Google Sheets API 인증 실패: secrets에 [google_sheets] 섹션이 없습니다.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    creds_obj = gs.get("credentials", None)
    creds_json = gs.get("credentials_json", None)

    if creds_obj is not None:
        if isinstance(creds_obj, dict):
            credentials_info = dict(creds_obj)
        else:
            credentials_info = {k: creds_obj[k] for k in creds_obj.keys()}
    elif creds_json:
        credentials_info = json.loads(str(creds_json))
    else:
        st.error("Google Sheets API 인증 실패: credentials(or credentials_json) 가 없습니다.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    if "private_key" in credentials_info:
        credentials_info["private_key"] = credentials_info["private_key"].replace("\\n", "\n").strip()

    try:
        credentials = Credentials.from_service_account_info(credentials_info, scopes=scopes)
        gc = gspread.authorize(credentials)
        ss = gc.open_by_key(GSHEET_ID)
    except Exception as exc:
        st.error(f"Google Sheets API 인증 실패: {exc}")
        st.error("secrets 형식: [google_sheets.credentials] (권장) 또는 [google_sheets] credentials_json")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    def _read(name: str) -> pd.DataFrame:
        try:
            return pd.DataFrame(ss.worksheet(name).get_all_records())
        except Exception as exc:  # pragma: no cover - streamlit feedback
            st.warning(f"{name} 시트를 읽을 수 없습니다: {exc}")
            return pd.DataFrame()

    df_move = _read("SCM_통합")
    df_ref = _read("snap_정제")
    df_incoming = _read("입고예정내역")

    try:
        df_snap_raw = _read("snapshot_raw")
        if not df_snap_raw.empty:
            cols = {c.strip().lower(): c for c in df_snap_raw.columns}
            col_date = cols.get("snapshot_date") or cols.get("date")
            if col_date:
                df_snap_raw[col_date] = pd.to_datetime(df_snap_raw[col_date], errors="coerce").dt.normalize()
                latest = df_snap_raw[col_date].max()
                if pd.notna(latest):
                    df_snap_raw = df_snap_raw[df_snap_raw[col_date] == latest].copy()
            st.session_state["_snapshot_raw_cache"] = df_snap_raw
        else:
            st.session_state["_snapshot_raw_cache"] = None
    except Exception:
        st.session_state["_snapshot_raw_cache"] = None

    return df_move, df_ref, df_incoming


@st.cache_data(ttl=300)
def load_from_excel(file: Any) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Excel 파일에서 데이터를 로드합니다.

    Args:
        file: Streamlit file uploader에서 받은 파일 객체

    Returns:
        Tuple of (moves DataFrame, refined snapshot DataFrame, incoming DataFrame, snapshot_raw DataFrame)
    """
    data = file.read() if hasattr(file, "read") else file
    bio = BytesIO(data) if isinstance(data, (bytes, bytearray)) else file

    xl = pd.ExcelFile(bio, engine="openpyxl")

    if "SCM_통합" not in xl.sheet_names:
        st.error("엑셀에 'SCM_통합' 시트가 필요합니다.")
        st.stop()

    df_move = pd.read_excel(bio, sheet_name="SCM_통합", engine="openpyxl")
    bio.seek(0)

    refined_name = next(
        (s for s in xl.sheet_names if s in ["snap_정제", "snap_refined", "snap_refine", "snap_ref"]),
        None,
    )
    if refined_name is None:
        st.error("엑셀에 정제 스냅샷 시트가 필요합니다. (시트명: 'snap_정제' 또는 'snap_refined')")
        st.stop()
    df_ref = pd.read_excel(bio, sheet_name=refined_name, engine="openpyxl")
    bio.seek(0)

    df_incoming = None
    if "입고예정내역" in xl.sheet_names:
        df_incoming = pd.read_excel(bio, sheet_name="입고예정내역", engine="openpyxl")
        bio.seek(0)

    snapshot_raw_df = None
    if "snapshot_raw" in xl.sheet_names:
        snapshot_raw_df = pd.read_excel(bio, sheet_name="snapshot_raw", engine="openpyxl")
        bio.seek(0)

    return df_move, df_ref, df_incoming, snapshot_raw_df


def load_snapshot_raw() -> pd.DataFrame:
    """세션 상태에서 snapshot_raw 데이터를 로드합니다.

    Returns:
        snapshot_raw DataFrame (없으면 빈 DataFrame)
    """
    if st.session_state.get("_snapshot_raw_cache") is not None:
        return st.session_state["_snapshot_raw_cache"].copy()

    fetch = globals().get("_fetch_sheet_via_webapp", None)
    if callable(fetch):
        try:
            df = fetch("snapshot_raw")
            if df is not None and not df.empty:
                return df
        except Exception:
            pass

    return pd.DataFrame()


def _parse_po_date(po_str: str) -> pd.Timestamp:
    """PO 번호에서 날짜를 파싱합니다.

    Args:
        po_str: PO 번호 문자열 (예: "A250115")

    Returns:
        파싱된 날짜 (Timestamp) 또는 NaT
    """
    if not isinstance(po_str, str):
        return pd.NaT
    match = re.search(r"[A-Za-z](\d{2})(\d{2})(\d{2})", po_str)
    if not match:
        return pd.NaT
    yy, mm, dd = match.groups()
    try:
        year = 2000 + int(yy)
        return pd.Timestamp(year=year, month=int(mm), day=int(dd)).normalize()
    except (ValueError, TypeError):
        return pd.NaT


def normalize_center_series(series: pd.Series) -> pd.Series:
    """센터명 Series를 정규화합니다.

    Args:
        series: 센터명 Series

    Returns:
        정규화된 센터명 Series
    """
    out = series.astype(str).str.strip()
    # 센터명 정규화 로직 (필요시 추가)
    return out


def load_wip_from_incoming(df_incoming: Optional[pd.DataFrame], default_center: str = "태광KR") -> pd.DataFrame:
    """입고예정내역에서 WIP 데이터를 로드합니다.

    Args:
        df_incoming: 입고예정내역 DataFrame
        default_center: 기본 목적지 센터

    Returns:
        WIP DataFrame (컬럼: resource_code, to_center, wip_start, wip_ready, qty_ea, lot)
    """
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
    """WIP 데이터를 moves DataFrame과 병합합니다.

    Args:
        moves_df: 이동 원장 DataFrame
        wip_df: WIP DataFrame

    Returns:
        WIP가 병합된 moves DataFrame
    """
    if wip_df is None or wip_df.empty:
        return moves_df
    wip_df_norm = wip_df.copy()
    wip_df_norm["to_center"] = normalize_center_series(wip_df_norm["to_center"])
    wip_df_norm["wip_start"] = pd.to_datetime(wip_df_norm["wip_start"], errors="coerce").dt.normalize()
    wip_df_norm["wip_ready"] = pd.to_datetime(wip_df_norm["wip_ready"], errors="coerce").dt.normalize()

    def _first_valid_center(series: Optional[pd.Series]) -> Optional[str]:
        if series is None:
            return None
        for value in series.dropna().astype(str).str.strip():
            if value and value.upper() != "WIP":
                return value
        return None

    default_center = _first_valid_center(wip_df_norm.get("to_center"))
    if default_center is None and "to_center" in moves_df:
        default_center = _first_valid_center(moves_df["to_center"])
    if default_center is None and "from_center" in moves_df:
        default_center = _first_valid_center(moves_df["from_center"])

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

    wip_moves["to_center"] = normalize_center_series(wip_moves["to_center"])
    mask_to_wip = (wip_moves["to_center"].str.upper() == "WIP").fillna(False)
    if default_center:
        wip_moves.loc[mask_to_wip, "to_center"] = default_center
    else:
        wip_moves.loc[mask_to_wip, "to_center"] = pd.NA

    # Keep WIP as a virtual origin so future center timelines do not
    # pre-subtract in-progress quantities from the physical warehouse.
    wip_moves["from_center"] = normalize_center_series(wip_moves["from_center"])

    for col in ["onboard_date", "arrival_date", "event_date"]:
        wip_moves[col] = pd.to_datetime(wip_moves[col], errors="coerce").dt.normalize()
    return pd.concat([moves_df, wip_moves], ignore_index=True)
