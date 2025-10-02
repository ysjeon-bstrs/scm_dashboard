"""Data source loaders used by the SCM dashboard."""

from __future__ import annotations

import json
from io import BytesIO
from typing import Optional, Tuple

import pandas as pd
import streamlit as st

from google.oauth2.service_account import Credentials
import gspread

from .config import GSHEET_ID


@st.cache_data(ttl=300)
def load_from_gsheet_api() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
def load_from_excel(file) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
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


@st.cache_data(ttl=300)
def load_snapshot_raw() -> pd.DataFrame:
    if st.session_state.get("_snapshot_raw_cache") is not None:
        return st.session_state["_snapshot_raw_cache"]

    fetch = globals().get("_fetch_sheet_via_webapp", None)
    if callable(fetch):
        try:
            df = fetch("snapshot_raw")
            if df is not None and not df.empty:
                return df
        except Exception:
            pass

    return pd.DataFrame()
