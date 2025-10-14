
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import re
from datetime import datetime, timedelta
from io import BytesIO
from typing import Dict, List, Optional, Tuple
import gspread
from google.oauth2.service_account import Credentials

from center_alias import normalize_center_series, normalize_center_value

# =========================
# Global configuration
# =========================
st.set_page_config(page_title="글로벌 대시보드 — v4", layout="wide")
st.title("📦 SCM 재고 흐름 대시보드 — v4")

# 데이터 소스 상태(excel | csv | gsheet). UI의 입력 탭에서 세팅함
if "_data_source" not in st.session_state:
    st.session_state["_data_source"] = None  # 아직 미정
# on-demand로 사용할 snapshot_raw 캐시 (엑셀에서 읽은 경우 저장)
if "_snapshot_raw_cache" not in st.session_state:
    st.session_state["_snapshot_raw_cache"] = None

# 원본 스프레드시트 ID (Google Sheets 탭에서만 사용)
GSHEET_ID = "1RYjKW2UDJ2kWJLAqQH26eqx2-r9Xb0_qE_hfwu9WIj8"

# 센터별 원본 컬럼 매핑
CENTER_COL = {
    "태광KR": "stock2",
    "AMZUS": "fba_available_stock",
    "품고KR": "poomgo_v2_available_stock",
    "SBSPH": "shopee_ph_available_stock",
    "SBSSG": "shopee_sg_available_stock",
    "SBSMY": "shopee_my_available_stock",
    "AcrossBUS": "acrossb_available_stock",
    "어크로스비US": "acrossb_available_stock",  # 별칭 통일
}

# -------------------- Small helpers --------------------
def _coalesce_columns(df: pd.DataFrame, candidates: List, parse_date: bool = False) -> pd.Series:
    """
    df에서 후보 컬럼들 중 첫 번째 유효 컬럼을 찾아 값을 반환.
    parse_date=True면 datetime으로 파싱.
    """
    all_names = []
    for item in candidates:
        if isinstance(item, (list, tuple, set)):
            all_names.extend([str(x).strip() for x in item])
        else:
            all_names.append(str(item).strip())

    # 정확 일치 → 대소문자 무시 포함 검색 → 부분 일치
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
    out = sub.bfill(axis=1).iloc[:, 0]
    return out


# -------------------- Google Sheets (API authentication) loader --------------------
@st.cache_data(ttl=300)
def load_from_gsheet_api():
    import json
    from google.oauth2.service_account import Credentials
    import gspread

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets.readonly",
        "https://www.googleapis.com/auth/drive.readonly",
    ]

    # -- secrets 3형식 지원 --
    try:
        gs = st.secrets["google_sheets"]
    except Exception as e:
        st.error("Google Sheets API 인증 실패: secrets에 [google_sheets] 섹션이 없습니다.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    creds_obj = gs.get("credentials", None)
    creds_json = gs.get("credentials_json", None)

    if creds_obj is not None:
        # 중첩/인라인 테이블
        if isinstance(creds_obj, dict):
            credentials_info = dict(creds_obj)
        else:
            # Streamlit Secrets 객체 → dict
            credentials_info = {k: creds_obj[k] for k in creds_obj.keys()}
    elif creds_json:
        # 멀티라인 JSON 문자열
        credentials_info = json.loads(str(creds_json))
    else:
        st.error("Google Sheets API 인증 실패: credentials(or credentials_json) 가 없습니다.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # 개행 복구(인라인 테이블 대비)
    if "private_key" in credentials_info:
        credentials_info["private_key"] = credentials_info["private_key"].replace("\\n", "\n").strip()

    try:
        credentials = Credentials.from_service_account_info(credentials_info, scopes=scopes)
        gc = gspread.authorize(credentials)
        ss = gc.open_by_key(GSHEET_ID)
    except Exception as e:
        st.error(f"Google Sheets API 인증 실패: {e}")
        st.error("secrets 형식: [google_sheets.credentials] (권장) 또는 [google_sheets] credentials_json")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # 시트 읽기
    def _read(name):
        try:
            return pd.DataFrame(ss.worksheet(name).get_all_records())
        except Exception as e:
            st.warning(f"{name} 시트를 읽을 수 없습니다: {e}")
            return pd.DataFrame()

    # 기존 3개 시트
    df_move = _read("SCM_통합")
    df_ref = _read("snap_정제")
    df_incoming = _read("입고예정내역")

    # 🔹 snapshot_raw(선택)도 시도해서 읽고, 읽히면 세션 캐시에 저장
    try:
        df_snap_raw = _read("snapshot_raw")
        if not df_snap_raw.empty:
            # 메모 절감을 위해 최신 스냅샷만 보관 (옵션)
            cols = {c.strip().lower(): c for c in df_snap_raw.columns}
            col_date = cols.get("snapshot_date") or cols.get("date")
            if col_date:
                df_snap_raw[col_date] = pd.to_datetime(df_snap_raw[col_date], errors="coerce").dt.normalize()
                latest = df_snap_raw[col_date].max()
                if pd.notna(latest):
                    df_snap_raw = df_snap_raw[df_snap_raw[col_date] == latest].copy()
            st.session_state["_snapshot_raw_cache"] = df_snap_raw  # ✅ 캐시에 저장
        else:
            st.session_state["_snapshot_raw_cache"] = None
    except Exception:
        # 없거나 권한 없으면 조용히 패스 (로트 상세는 자동으로 미표시)
        st.session_state["_snapshot_raw_cache"] = None

    return df_move, df_ref, df_incoming

# -------------------- Loaders --------------------
@st.cache_data(ttl=300)
def load_from_excel(file):
    """
    필수: SCM_통합, snap_정제
    선택: 입고예정내역, snapshot_raw
    """
    data = file.read() if hasattr(file, "read") else file
    bio = BytesIO(data) if isinstance(data, (bytes, bytearray)) else file

    xl = pd.ExcelFile(bio, engine="openpyxl")

    if "SCM_통합" not in xl.sheet_names:
        st.error("엑셀에 'SCM_통합' 시트가 필요합니다.")
        st.stop()

    df_move = pd.read_excel(bio, sheet_name="SCM_통합", engine="openpyxl")
    bio.seek(0)

    refined_name = next((s for s in xl.sheet_names if s in ["snap_정제","snap_refined","snap_refine","snap_ref"]), None)
    if refined_name is None:
        st.error("엑셀에 정제 스냅샷 시트가 필요합니다. (시트명: 'snap_정제' 또는 'snap_refined')")
        st.stop()
    df_ref = pd.read_excel(bio, sheet_name=refined_name, engine="openpyxl")
    bio.seek(0)

    df_incoming = None
    if "입고예정내역" in xl.sheet_names:
        df_incoming = pd.read_excel(bio, sheet_name="입고예정내역", engine="openpyxl")
        bio.seek(0)

    # snapshot_raw도 있으면 캐시에 보관 (재고자산 계산용)
    snapshot_raw_df = None
    if "snapshot_raw" in xl.sheet_names:
        snapshot_raw_df = pd.read_excel(bio, sheet_name="snapshot_raw", engine="openpyxl")
        bio.seek(0)

    return df_move, df_ref, df_incoming, snapshot_raw_df


@st.cache_data(ttl=300)
def load_snapshot_raw():
    """
    재고자산 계산용 원본 스냅샷.
    - EXCEL/CSV를 사용 중이면 업로드 파일의 snapshot_raw(있을 때만) 사용
      (없으면 빈 DF, 에러 메시지 출력하지 않음)
    - Apps Script 프록시(_fetch_sheet_via_webapp)가 있으면 그쪽을 우선 사용
    """
    # 1) 업로드 캐시가 있으면 최우선
    if st.session_state.get("_snapshot_raw_cache") is not None:
        return st.session_state["_snapshot_raw_cache"]

    # 2) Apps Script 프록시 제공 시 사용
    fetch = globals().get("_fetch_sheet_via_webapp", None)
    if callable(fetch):
        try:
            df = fetch("snapshot_raw")
            if df is not None and not df.empty:
                return df
        except Exception:
            pass

    # 3) 그 외에는 조용히 빈 DF
    return pd.DataFrame()

# -------------------- Normalizers --------------------
def normalize_refined_snapshot(df_ref: pd.DataFrame) -> pd.DataFrame:
    cols = {str(c).strip().lower(): c for c in df_ref.columns}

    # 날짜/센터/코드/수량 기본 컬럼 탐색 (alias 확대)
    date_col     = next((cols[k] for k in ["date","날짜","snapshot_date","스냅샷일"] if k in cols), None)
    center_col   = next((cols[k] for k in ["center","센터","창고","warehouse"] if k in cols), None)
    resource_col = next((cols[k] for k in ["resource_code","resource_cc","sku","상품코드","product_code"] if k in cols), None)
    stock_col    = next((cols[k] for k in ["stock_qty","qty","수량","재고","quantity"] if k in cols), None)

    # (신규) 품명 컬럼 탐색
    name_col     = next((cols[k] for k in ["resource_name","품명","상품명","product_name"] if k in cols), None)

    missing = [n for n,v in {"date":date_col,"center":center_col,"resource_code":resource_col,"stock_qty":stock_col}.items() if not v]
    if missing:
        st.error(f"'snap_정제' 시트에 누락된 컬럼: {missing}")
        st.stop()

    result = df_ref.rename(columns={
        date_col: "date",
        center_col: "center",
        resource_col: "resource_code",
        stock_col: "stock_qty",
        **({name_col: "resource_name"} if name_col else {})
    }).copy()

    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.normalize()
    result["center"] = normalize_center_series(result["center"])
    result["resource_code"] = result["resource_code"].astype(str)
    result["stock_qty"] = pd.to_numeric(result["stock_qty"], errors="coerce").fillna(0).astype(int)

    if "resource_name" in result.columns:
        # 문자열 정리 (NaN 방지)
        result["resource_name"] = result["resource_name"].astype(str).str.strip().replace({"nan": "", "None": ""})

    return result.dropna(subset=["date","center","resource_code"])

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
    out = pd.DataFrame({
        "resource_code": resource_code.astype(str).str.strip(),
        "qty_ea": pd.to_numeric(qty_ea.astype(str).str.replace(',',''), errors="coerce").fillna(0).astype(int),
        "carrier_mode": carrier_mode.astype(str).str.strip(),
        "from_center": normalize_center_series(from_center),
        "to_center": normalize_center_series(to_center),
        "onboard_date": onboard_date,
        "arrival_date": arrival_date,
        "inbound_date": inbound_date,
    })
    out["event_date"] = out["inbound_date"].where(out["inbound_date"].notna(), out["arrival_date"])
    for col in ["onboard_date", "arrival_date", "inbound_date", "event_date"]:
        out[col] = pd.to_datetime(out[col], errors="coerce").dt.normalize()
    return out

# PO 번호 → 날짜 파싱
def _parse_po_date(po_str: str) -> pd.Timestamp:
    if not isinstance(po_str, str):
        return pd.NaT
    m = re.search(r"[A-Za-z](\d{2})(\d{2})(\d{2})", po_str)
    if not m:
        return pd.NaT
    yy, mm, dd = m.groups()
    year = 2000 + int(yy)
    try:
        return pd.Timestamp(datetime(year, int(mm), int(dd)))
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
    mask_na = out["wip_start"].isna() & out["wip_ready"].notna()
    out.loc[mask_na, "wip_start"] = out.loc[mask_na, "wip_ready"] - pd.to_timedelta(30, unit="D")

    out = out.dropna(subset=["resource_code","wip_ready","wip_start"]).reset_index(drop=True)
    return out[["resource_code","to_center","wip_start","wip_ready","qty_ea","lot"]]

def merge_wip_as_moves(moves_df: pd.DataFrame, wip_df: Optional[pd.DataFrame]) -> pd.DataFrame:
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

    wip_moves = pd.DataFrame({
        "resource_code": wip_df_norm["resource_code"],
        "qty_ea": wip_df_norm["qty_ea"].astype(int),
        "carrier_mode": "WIP",
        "from_center": "WIP",
        "to_center": wip_df_norm["to_center"],
        "onboard_date": wip_df_norm["wip_start"],
        "arrival_date": wip_df_norm["wip_ready"],
        "inbound_date": pd.NaT,
        "event_date": wip_df_norm["wip_ready"],
        "lot": wip_df_norm.get("lot", "")
    })

    wip_moves["to_center"] = normalize_center_series(wip_moves["to_center"])
    mask_to_wip = (wip_moves["to_center"].str.upper() == "WIP").fillna(False)
    if default_center:
        wip_moves.loc[mask_to_wip, "to_center"] = default_center
    else:
        wip_moves.loc[mask_to_wip, "to_center"] = pd.NA

    wip_moves["from_center"] = normalize_center_series(wip_moves["from_center"])
    mask_from_wip = (wip_moves["from_center"].str.upper() == "WIP").fillna(False)
    wip_moves.loc[mask_from_wip, "from_center"] = (
        wip_moves.loc[mask_from_wip, "to_center"].fillna(default_center)
    )

    for col in ["onboard_date", "arrival_date", "event_date"]:
        wip_moves[col] = pd.to_datetime(wip_moves[col], errors="coerce").dt.normalize()
    return pd.concat([moves_df, wip_moves], ignore_index=True)

# -------------------- 소비(소진) 추세 + 이벤트 가중치 --------------------
@st.cache_data(ttl=3600)
def estimate_daily_consumption(snap_long: pd.DataFrame,
                               centers_sel: List[str], skus_sel: List[str],
                               asof_dt: pd.Timestamp,
                               lookback_days: int = 28) -> Dict[Tuple[str, str], float]:
    snap = snap_long.rename(columns={"snapshot_date":"date"}).copy()
    snap["date"] = pd.to_datetime(snap["date"], errors="coerce").dt.normalize()
    start = (asof_dt - pd.Timedelta(days=int(lookback_days)-1)).normalize()

    hist = snap[(snap["date"] >= start) & (snap["date"] <= asof_dt) &
                (snap["center"].isin(centers_sel)) &
                (snap["resource_code"].isin(skus_sel))]

    rates = {}
    if hist.empty:
        return rates

    for (ct, sku), g in hist.groupby(["center","resource_code"]):
        ts = (g.sort_values("date")
                .set_index("date")["stock_qty"]
                .asfreq("D").ffill())
        if ts.dropna().shape[0] < max(7, lookback_days//2):
            continue
        x = np.arange(len(ts), dtype=float)
        y = ts.values.astype(float)
        rate_reg = max(0.0, -np.polyfit(x, y, 1)[0])  # 감소만
        dec = (-np.diff(y)).clip(min=0)
        rate_dec = dec.mean() if len(dec) else 0.0
        rate = max(rate_reg, rate_dec)
        if rate > 0:
            rates[(ct, sku)] = float(rate)
    return rates

@st.cache_data(ttl=1800)
def apply_consumption_with_events(
    timeline: pd.DataFrame,
    snap_long: pd.DataFrame,
    centers_sel: List[str], skus_sel: List[str],
    start_dt: pd.Timestamp, end_dt: pd.Timestamp,
    lookback_days: int = 28,
    events: Optional[List[Dict]] = None
) -> pd.DataFrame:
    out = timeline.copy()
    if out.empty:
        return out
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()

    snap_cols = {c.lower(): c for c in snap_long.columns}
    date_col = snap_cols.get("date") or snap_cols.get("snapshot_date")
    if date_col is None:
        raise KeyError("snap_long에는 'date' 또는 'snapshot_date' 컬럼이 필요합니다.")

    latest_snap = pd.to_datetime(snap_long[date_col]).max().normalize()
    cons_start = max(latest_snap + pd.Timedelta(days=1), start_dt)
    if cons_start > end_dt:
        return out

    idx = pd.date_range(cons_start, end_dt, freq="D")
    uplift = pd.Series(1.0, index=idx)
    if events:
        for e in events:
            s = pd.to_datetime(e.get("start"), errors="coerce")
            t = pd.to_datetime(e.get("end"), errors="coerce")
            u = min(3.0, max(-1.0, float(e.get("uplift", 0.0))))
            if pd.notna(s) and pd.notna(t):
                s = s.normalize(); t = t.normalize()
                s = max(s, idx[0]); t = min(t, idx[-1])
                if s <= t:
                    uplift.loc[s:t] = uplift.loc[s:t] * (1.0 + u)

    rates = estimate_daily_consumption(snap_long, centers_sel, skus_sel, latest_snap, int(lookback_days))

    chunks: list[pd.DataFrame] = []
    for (ct, sku), g in out.groupby(["center","resource_code"]):
        if ct in ("In-Transit", "WIP"):
            chunks.append(g)
            continue

        rate = float(rates.get((ct, sku), 0.0))
        if rate <= 0:
            chunks.append(g)
            continue

        g = g.sort_values("date").copy()
        mask = g["date"] >= cons_start
        if not mask.any():
            chunks.append(g)
            continue

        daily = g.loc[mask, "date"].map(uplift).fillna(1.0).values * rate
        stk = g.loc[mask, "stock_qty"].astype(float).values
        for i in range(len(stk)):
            dec = daily[i]
            stk[i:] = np.maximum(0.0, stk[i:] - dec)
        g.loc[mask, "stock_qty"] = stk
        chunks.append(g)

    if not chunks:
        return out

    out = pd.concat(chunks, ignore_index=True)
    # 더 강력한 NaN 처리
    out["stock_qty"] = pd.to_numeric(out["stock_qty"], errors="coerce")
    out["stock_qty"] = out["stock_qty"].fillna(0)
    out["stock_qty"] = out["stock_qty"].replace([np.inf, -np.inf], 0)
    out["stock_qty"] = out["stock_qty"].round().clip(lower=0).astype(int)
    return out


# -------------------- Timeline --------------------
def build_timeline(snap_long: pd.DataFrame, moves: pd.DataFrame, 
                   centers_sel: List[str], skus_sel: List[str],
                   start_dt: pd.Timestamp, end_dt: pd.Timestamp, 
                   horizon_days: int = 0, today: Optional[pd.Timestamp] = None,
                   lag_days: int = 7) -> pd.DataFrame:
    if today is None:
        today = pd.Timestamp.today().normalize()
    horizon_end = end_dt + pd.Timedelta(days=horizon_days)
    full_dates = pd.date_range(start_dt, horizon_end, freq="D")

    # 이동건 복사
    mv_all = moves.copy()

    base = snap_long[
        snap_long["center"].isin(centers_sel) &
        snap_long["resource_code"].isin(skus_sel)
    ].copy().rename(columns={"snapshot_date":"date"})
    if base.empty:
        return pd.DataFrame(columns=["date","center","resource_code","stock_qty"])
    base = base[(base["date"] >= start_dt) & (base["date"] <= end_dt)]

    lines = []

    # 1) 실제 센터 라인
    for (ct, sku), grp in base.groupby(["center","resource_code"]):
        grp = grp.sort_values("date")
        last_dt = grp["date"].max()

        if horizon_days > 0:
            proj_dates = pd.date_range(last_dt + pd.Timedelta(days=1), horizon_end, freq="D")
            proj_df = pd.DataFrame({"date": proj_dates, "center": ct,
                                    "resource_code": sku, "stock_qty": np.nan})
            ts = pd.concat([grp[["date","center","resource_code","stock_qty"]], proj_df], ignore_index=True)
        else:
            ts = grp[["date","center","resource_code","stock_qty"]].copy()

        ts = ts.sort_values("date")
        ts["stock_qty"] = ts["stock_qty"].ffill()

        mv = mv_all[mv_all["resource_code"] == sku].copy()

        # 출고(-) 이벤트
        eff_minus = (
            mv[(mv["from_center"].astype(str) == str(ct)) &
               (mv["onboard_date"].notna()) &
               (mv["onboard_date"] > last_dt)]
            .groupby("onboard_date", as_index=False)["qty_ea"].sum()
            .rename(columns={"onboard_date":"date","qty_ea":"delta"})
        )
        eff_minus["delta"] *= -1

        # 입고(+) 이벤트 (예측 입고일 계산)
        mv_center = mv[(mv["to_center"].astype(str) == str(ct))].copy()
        if not mv_center.empty:
            # 예측 입고일 계산 (이동중 종료일과 동일한 로직)
            pred_inbound = pd.Series(pd.NaT, index=mv_center.index, dtype="datetime64[ns]")
            
            # 1) inbound가 있으면 그 날 입고
            mask_inb = mv_center["inbound_date"].notna()
            pred_inbound.loc[mask_inb] = mv_center.loc[mask_inb, "inbound_date"]
            
            # 2) inbound 없고 arrival 있는 경우
            mask_arr = (~mask_inb) & mv_center["arrival_date"].notna()
            if mask_arr.any():
                # 2-1) arrival이 과거면: arrival + N일에 입고(가정)
                past_arr = mask_arr & (mv_center["arrival_date"] <= today)
                pred_inbound.loc[past_arr] = (
                    mv_center.loc[past_arr, "arrival_date"] + pd.Timedelta(days=int(lag_days))
                )
                
                # 2-2) arrival이 미래면: arrival에 입고
                fut_arr = mask_arr & (mv_center["arrival_date"] > today)
                pred_inbound.loc[fut_arr] = mv_center.loc[fut_arr, "arrival_date"]
            
            mv_center["pred_inbound_date"] = pred_inbound
            
            eff_plus = (
                mv_center[(mv_center["pred_inbound_date"].notna()) &
                          (mv_center["pred_inbound_date"] > last_dt)]
                .groupby("pred_inbound_date", as_index=False)["qty_ea"].sum()
                .rename(columns={"pred_inbound_date":"date","qty_ea":"delta"})
            )
        else:
            eff_plus = pd.DataFrame(columns=["date","delta"])

        # 벡터화된 처리: 날짜별 증감(Delta) 시리즈로 변경
        eff_all = pd.concat([eff_minus, eff_plus], ignore_index=True)
        if not eff_all.empty:
            # 날짜별로 그룹화하여 합계 계산
            delta_series = eff_all.groupby("date")["delta"].sum()
            # 날짜 인덱스에 맞춰 reindex하고 누적합 계산
            delta_series = delta_series.reindex(ts["date"], fill_value=0).fillna(0)
            # 누적합 대신 직접 더하기 (더 안전)
            for i, (date, delta) in enumerate(delta_series.items()):
                if delta != 0:
                    ts.loc[ts["date"] >= date, "stock_qty"] = ts.loc[ts["date"] >= date, "stock_qty"] + delta

        ts["stock_qty"] = ts["stock_qty"].fillna(0).replace([np.inf, -np.inf], 0).clip(lower=0)
        lines.append(ts)

        # (보강) WIP 완료 물량을 해당 도착 센터 라인에 반영
        wip_complete = moves[
            (moves["resource_code"] == sku) &
            (moves["carrier_mode"].astype(str).str.upper() == "WIP") &
            (moves["to_center"] == ct) &
            (moves["event_date"].notna())
        ].copy()
        if not wip_complete.empty:
            wip_add = (
                wip_complete.groupby("event_date", as_index=False)["qty_ea"].sum()
                .rename(columns={"event_date":"date","qty_ea":"delta"})
            )
            # 벡터화된 WIP 처리
            wip_delta_series = wip_add.groupby("date")["delta"].sum()
            wip_delta_series = wip_delta_series.reindex(ts["date"], fill_value=0).fillna(0)
            # 누적합 대신 직접 더하기 (더 안전)
            for date, delta in wip_delta_series.items():
                if delta != 0:
                    ts.loc[ts["date"] >= date, "stock_qty"] = ts.loc[ts["date"] >= date, "stock_qty"] + delta
            ts["stock_qty"] = ts["stock_qty"].fillna(0).replace([np.inf, -np.inf], 0).clip(lower=0)
            lines[-1] = ts  # 갱신

    # 2) In-Transit & WIP 가상 라인
    moves_str = mv_all.copy()
    moves_str["from_center"] = moves_str["from_center"].astype(str)
    moves_str["to_center"] = moves_str["to_center"].astype(str)
    moves_str["carrier_mode"] = moves_str["carrier_mode"].astype(str).str.upper()

    mv_sel = moves_str[
        moves_str["resource_code"].isin(skus_sel) &
        (moves_str["from_center"].isin(centers_sel) | 
         moves_str["to_center"].isin(centers_sel) | 
         (moves_str["carrier_mode"] == "WIP"))
    ]

    for sku, g in mv_sel.groupby("resource_code"):
        # --- Non-WIP In-Transit (벡터화) ----
        g_nonwip = g[g["carrier_mode"] != "WIP"]
        if not g_nonwip.empty:
            g_selected = g_nonwip[g_nonwip["to_center"].isin(centers_sel)]
            if not g_selected.empty:
                idx = pd.date_range(start_dt, horizon_end, freq="D")
                today_norm = (today or pd.Timestamp.today()).normalize()

                # 종료일 계산: inbound(있으면) / arrival(미래면) / 그 외 today+1
                end_eff = pd.Series(pd.NaT, index=g_selected.index, dtype="datetime64[ns]")

                # 1) inbound가 있으면 그 날 종료
                mask_inb = g_selected["inbound_date"].notna()
                end_eff.loc[mask_inb] = g_selected.loc[mask_inb, "inbound_date"]

                # 2) inbound 없고 arrival 있는 경우
                mask_arr = (~mask_inb) & g_selected["arrival_date"].notna()
                if mask_arr.any():
                    # 2-1) arrival이 과거면: arrival + N일에 종료(가정)
                    past_arr = mask_arr & (g_selected["arrival_date"] <= today_norm)
                    end_eff.loc[past_arr] = (
                        g_selected.loc[past_arr, "arrival_date"] + pd.Timedelta(days=int(lag_days))
                    )

                    # 2-2) arrival이 미래면: arrival에 종료
                    fut_arr = mask_arr & (g_selected["arrival_date"] > today_norm)
                    end_eff.loc[fut_arr] = g_selected.loc[fut_arr, "arrival_date"]

                # 3) 그래도 비어있으면: today+1 (화면상 오늘까지만 이동중으로 보이도록)
                end_eff = end_eff.fillna(min(today_norm + pd.Timedelta(days=1), idx[-1] + pd.Timedelta(days=1)))

                # starts/ends 델타 만들어 누적합
                g_selected_with_end = g_selected.copy()
                g_selected_with_end["end_date"] = end_eff

                starts = (g_selected_with_end
                          .dropna(subset=["onboard_date"])
                          .groupby("onboard_date")["qty_ea"].sum())
                ends = (g_selected_with_end
                        .groupby("end_date")["qty_ea"].sum() * -1)

                delta = (starts.rename_axis("date").to_frame("delta")
                           .add(ends.rename_axis("date").to_frame("delta"), fill_value=0)["delta"]
                           .sort_index())

                s = delta.reindex(idx, fill_value=0).cumsum().clip(lower=0)

                # carry(기간 시작 이전 출발해 아직 안 끝난 건) 처리
                carry_mask = (
                    g_selected["onboard_date"].notna() &
                    (g_selected["onboard_date"] < idx[0]) &
                    (end_eff > idx[0])
                )
                carry = int(g_selected.loc[carry_mask, "qty_ea"].sum())
                if carry:
                    s = (s + carry).clip(lower=0)

                if s.any():
                    lines.append(pd.DataFrame({
                        "date": s.index, "center": "In-Transit",
                        "resource_code": sku, "stock_qty": s.values.astype(int)
                    }))

        # --- WIP ---
        g_wip = g[g["carrier_mode"] == "WIP"]
        if not g_wip.empty:
            # 벡터화된 WIP 처리
            s = pd.Series(0, index=pd.to_datetime(full_dates))
            
            # onboard +, event - 의 누적 효과를 연속 값으로 변환
            add_onboard = (
                g_wip[g_wip["onboard_date"].notna()]
                .groupby("onboard_date", as_index=False)["qty_ea"].sum()
                .rename(columns={"onboard_date":"date","qty_ea":"delta"})
            )
            add_event = (
                g_wip[g_wip["event_date"].notna()]
                .groupby("event_date", as_index=False)["qty_ea"].sum()
                .rename(columns={"event_date":"date","qty_ea":"delta"})
            )
            add_event["delta"] *= -1
            deltas = pd.concat([add_onboard, add_event], ignore_index=True)
            
            if not deltas.empty:
                # 날짜별로 그룹화하여 합계 계산
                delta_series = deltas.groupby("date")["delta"].sum()
                # 날짜 인덱스에 맞춰 reindex하고 직접 더하기
                delta_series = delta_series.reindex(s.index, fill_value=0).fillna(0)
                for date, delta in delta_series.items():
                    if delta != 0:
                        s.loc[s.index >= date] = s.loc[s.index >= date] + delta
                
                vdf = pd.DataFrame({"date": s.index, "center": "WIP",
                                    "resource_code": sku, "stock_qty": s.values})
                vdf["stock_qty"] = vdf["stock_qty"].fillna(0).replace([np.inf, -np.inf], 0).clip(lower=0)
                lines.append(vdf)

    if not lines:
        return pd.DataFrame(columns=["date","center","resource_code","stock_qty"])

    out = pd.concat(lines, ignore_index=True)
    out = out[(out["date"] >= start_dt) & (out["date"] <= horizon_end)]
    
    # 최종 NaN 처리
    out["stock_qty"] = pd.to_numeric(out["stock_qty"], errors="coerce")
    out["stock_qty"] = out["stock_qty"].fillna(0)
    out["stock_qty"] = out["stock_qty"].replace([np.inf, -np.inf], 0)
    out["stock_qty"] = out["stock_qty"].clip(lower=0).astype(int)
    
    return out

# -------------------- 비용(재고자산) 피벗 --------------------
def pivot_inventory_cost_from_raw(snap_raw: pd.DataFrame,
                                  _latest_dt: pd.Timestamp,
                                  centers: list[str]) -> pd.DataFrame:
    if snap_raw is None or snap_raw.empty:
        return pd.DataFrame(columns=["resource_code"] + [f"{c}_재고자산" for c in centers])

    df = snap_raw.copy()
    cols = {str(c).strip().lower(): c for c in df.columns}
    col_date = cols.get("snapshot_date") or cols.get("date")
    col_sku  = cols.get("resource_code") or cols.get("sku") or cols.get("상품코드") or cols.get("option1")
    col_cogs = cols.get("cogs")

    if not all([col_date, col_sku, col_cogs]):
        return pd.DataFrame(columns=["resource_code"] + [f"{c}_재고자산" for c in centers])

    df[col_date] = pd.to_datetime(df[col_date], errors="coerce").dt.normalize()
    df[col_sku]  = df[col_sku].astype(str)
    df[col_cogs] = pd.to_numeric(df[col_cogs], errors="coerce").fillna(0).clip(lower=0)

    # 최신 스냅샷(오늘 우선 → 없으면 가장 최근)
    today = pd.Timestamp.today().normalize()
    sub = df[df[col_date] == today].copy()
    if sub.empty:
        latest_date = df[col_date].max()
        sub = df[df[col_date] == latest_date].copy()
    if sub.empty:
        return pd.DataFrame(columns=["resource_code"] + [f"{c}_재고자산" for c in centers])

    cost_cols = {}
    for ct in centers:
        src_col = CENTER_COL.get(ct)
        if not src_col or src_col not in sub.columns:
            continue
        qty = pd.to_numeric(sub[src_col], errors="coerce").fillna(0).clip(lower=0)
        cost = (sub[col_cogs] * qty)
        g = sub[[col_sku]].copy()
        g[f"{ct}_재고자산"] = cost
        cost_cols[ct] = g.groupby(col_sku, as_index=False)[f"{ct}_재고자산"].sum()

    if not cost_cols:
        return pd.DataFrame(columns=["resource_code"] + [f"{c}_재고자산" for c in centers])

    base = pd.DataFrame({"resource_code": pd.unique(sub[col_sku])})
    for ct, g in cost_cols.items():
        base = base.merge(g.rename(columns={col_sku: "resource_code"}), on="resource_code", how="left")
    num_cols = [c for c in base.columns if c.endswith("_재고자산")]
    base[num_cols] = base[num_cols].fillna(0).round().astype(int)
    return base

# ==================== Tabs for inputs ====================
tab1, tab2 = st.tabs(["엑셀 업로드", "Google Sheets"])

with tab1:
    xfile = st.file_uploader("엑셀 업로드 (.xlsx)", type=["xlsx"], key="excel")
    if xfile is not None:
        df_move, df_ref, df_incoming, snap_raw_df = load_from_excel(xfile)
        st.session_state["_data_source"] = "excel"
        st.session_state["_snapshot_raw_cache"] = snap_raw_df  # snapshot_raw 있으면 저장

        moves_raw = normalize_moves(df_move)
        snap_long = normalize_refined_snapshot(df_ref)

        try:
            wip_df = load_wip_from_incoming(df_incoming)
            moves = merge_wip_as_moves(moves_raw, wip_df)
            st.success(f"WIP {len(wip_df)}건 반영 완료" if wip_df is not None and not wip_df.empty else "WIP 없음")
        except Exception as e:
            moves = moves_raw
            st.warning(f"WIP 불러오기 실패: {e}")

with tab2:
    st.info("Google Sheets API를 사용하여 데이터를 로드합니다.")
    st.caption("서비스 계정 키 파일을 사용하여 인증합니다.")
    
    if st.button("Google Sheets에서 데이터 로드", type="primary"):
        try:
            df_move, df_ref, df_incoming = load_from_gsheet_api()
            
            # 데이터가 비어있는지 확인
            if df_move.empty or df_ref.empty:
                st.error("❌ Google Sheets API로 데이터를 불러올 수 없습니다. 서비스 계정 권한을 확인해주세요.")
                st.stop()
            
            st.session_state["_data_source"] = "gsheet"

            moves_raw = normalize_moves(df_move)
            snap_long = normalize_refined_snapshot(df_ref)
            try:
                wip_df = load_wip_from_incoming(df_incoming)
                moves = merge_wip_as_moves(moves_raw, wip_df)
                st.success(f"✅ Google Sheets 로드 완료! WIP {len(wip_df)}건 반영" if wip_df is not None and not wip_df.empty else "✅ Google Sheets 로드 완료! WIP 없음")
            except Exception as e:
                moves = moves_raw
                st.warning(f"⚠️ WIP 불러오기 실패: {e}")
        except Exception as e:
            st.error(f"❌ Google Sheets 데이터 로드 중 오류가 발생했습니다: {e}")
            st.info("💡 해결 방법:\n- 서비스 계정 키 파일이 올바른지 확인\n- 스프레드시트에 서비스 계정 이메일이 공유되어 있는지 확인\n- 시트명이 정확한지 확인 (SCM_통합, snap_정제)")

# 초기 자동 로드(없을 때만): Google Sheets API 시도 → 실패하면 안내
if "snap_long" not in locals():
    try:
        df_move, df_ref, df_incoming = load_from_gsheet_api()
        if not df_move.empty and not df_ref.empty:
            st.session_state["_data_source"] = "gsheet"
            moves = normalize_moves(df_move)
            snap_long = normalize_refined_snapshot(df_ref)
            try:
                wip_df = load_wip_from_incoming(df_incoming)
                moves = merge_wip_as_moves(moves, wip_df)
            except Exception:
                pass
            st.success("✅ Google Sheets에서 데이터 로드됨 (필요 시 엑셀 업로드 탭 사용 가능)")
        else:
            st.info("엑셀 업로드 또는 Google Sheets에서 데이터를 로드하면 필터/차트가 나타납니다.")
            st.stop()
    except Exception:
        st.info("엑셀 업로드 또는 Google Sheets에서 데이터를 로드하면 필터/차트가 나타납니다.")
        st.stop()

# -------------------- Filters --------------------
centers_snap = set(snap_long["center"].dropna().astype(str).unique().tolist())
centers_moves = set(moves["from_center"].dropna().astype(str).unique().tolist() + 
                   moves["to_center"].dropna().astype(str).unique().tolist())

def normalize_center_name(center):
    return normalize_center_value(center)

all_centers = set()
for center in centers_snap | centers_moves:
    normalized = normalize_center_name(center)
    if normalized:
        all_centers.add(normalized)

centers = sorted(list(all_centers))
skus = sorted(snap_long["resource_code"].dropna().astype(str).unique().tolist())

today = pd.Timestamp.today().normalize()
PAST_DAYS = 42
FUTURE_DAYS = 60

snap_min = pd.to_datetime(snap_long["date"]).min().normalize()
snap_max = pd.to_datetime(snap_long["date"]).max().normalize()

bound_min = max(today - pd.Timedelta(days=PAST_DAYS), snap_min)
bound_max = min(today + pd.Timedelta(days=FUTURE_DAYS), snap_max + pd.Timedelta(days=60))

def _init_range():
    if "date_range" not in st.session_state:
        st.session_state.date_range = (max(today - pd.Timedelta(days=20), bound_min),
                                       min(today + pd.Timedelta(days=20), bound_max))
    if "horizon_days" not in st.session_state:
        st.session_state.horizon_days = 20

def _apply_horizon_to_range():
    h = int(st.session_state.horizon_days)
    h = max(0, min(h, FUTURE_DAYS))
    st.session_state.horizon_days = h
    start = max(today - pd.Timedelta(days=h), bound_min)
    end   = min(today + pd.Timedelta(days=h), bound_max)
    st.session_state.date_range = (start, end)

def _clamp_range(r):
    s, e = pd.Timestamp(r[0]).normalize(), pd.Timestamp(r[1]).normalize()
    s = max(min(s, bound_max), bound_min)
    e = max(min(e, bound_max), bound_min)
    if e < s:
        e = s
    return (s, e)

_init_range()

st.sidebar.header("필터")
centers_sel = st.sidebar.multiselect("센터 선택", centers, default=(["태광KR"] if "태광KR" in centers else centers[:1]))
skus_sel = st.sidebar.multiselect("SKU 선택", skus, default=([s for s in ["BA00022","BA00021"] if s in skus] or skus[:2]))

st.sidebar.subheader("기간 설정")
st.sidebar.number_input("미래 전망 일수", min_value=0, max_value=FUTURE_DAYS, step=1,
                        key="horizon_days", on_change=_apply_horizon_to_range)

date_range = st.sidebar.slider("기간",
    min_value=bound_min.to_pydatetime(),
    max_value=bound_max.to_pydatetime(),
    value=tuple(d.to_pydatetime() for d in _clamp_range(st.session_state.date_range)),
    format="YYYY-MM-DD")

start_dt = pd.Timestamp(date_range[0]).normalize()
end_dt   = pd.Timestamp(date_range[1]).normalize()
_latest_snap = snap_long["date"].max()
proj_days_for_build = max(0, int((end_dt - _latest_snap).days))

st.sidebar.header("표시 옵션")
show_prod = st.sidebar.checkbox("생산중(미완료) 표시", value=True)
show_transit = st.sidebar.checkbox("이동중 표시", value=True)
use_cons_forecast = st.sidebar.checkbox("추세 기반 재고 예측", value=True)
lookback_days = st.sidebar.number_input("추세 계산 기간(일)", min_value=7, max_value=56, value=28, step=7)

# 입고 반영 가정 옵션
st.sidebar.subheader("입고 반영 가정")
lag_days = st.sidebar.number_input("입고 반영 리드타임(일) – inbound 미기록 시 arrival+N", 
                                   min_value=0, max_value=21, value=7, step=1)

with st.sidebar.expander("프로모션 가중치(+%)", expanded=False):
    enable_event = st.checkbox("가중치 적용", value=False)
    ev_start = st.date_input("시작일")
    ev_end   = st.date_input("종료일")
    ev_pct   = st.number_input("가중치(%)", min_value=-100.0, max_value=300.0, value=30.0, step=5.0)
events = [{"start": pd.Timestamp(ev_start).strftime("%Y-%m-%d"),
           "end":   pd.Timestamp(ev_end).strftime("%Y-%m-%d"),
           "uplift": ev_pct/100.0}] if enable_event else []

# -------------------- KPIs (SKU별 분해) --------------------
st.subheader("요약 KPI")

# 스냅샷 날짜 컬럼 이름 호환('date' 또는 'snapshot_date')
_snap_date_col = "date" if "date" in snap_long.columns else "snapshot_date"
_latest_dt = pd.to_datetime(snap_long[_snap_date_col]).max().normalize()
_latest_dt_str = _latest_dt.strftime("%Y-%m-%d")

# 품명 매핑(선택)
_name_col = None
for cand in ["resource_name", "상품명", "품명"]:
    if cand in snap_long.columns:
        _name_col = cand
        break
_name_map = {}
if _name_col:
    name_rows = (snap_long[snap_long[_snap_date_col] == _latest_dt]
                    .dropna(subset=["resource_code"])[["resource_code", _name_col]]
                    .drop_duplicates())
    _name_map = dict(zip(name_rows["resource_code"].astype(str), name_rows[_name_col].astype(str)))

# moves 가공
_today = pd.Timestamp.today().normalize()
mv = moves.copy()
mv["carrier_mode"] = mv["carrier_mode"].astype(str).str.upper()
mv["resource_code"] = mv["resource_code"].astype(str)

def _kpi_breakdown_per_sku(snap_long, mv, centers_sel, skus_sel, today):
    # 현재 재고(최신 스냅샷, 선택 센터 합계)
    cur = (snap_long[
        (snap_long[_snap_date_col] == _latest_dt) &
        (snap_long["center"].isin(centers_sel)) &
        (snap_long["resource_code"].astype(str).isin(skus_sel))
    ].groupby("resource_code", as_index=True)["stock_qty"].sum())

    # 이동중: 예측 종료일 기준으로 오늘 이후까지 이동중인 건만
    mv_kpi = mv.copy()
    if not mv_kpi.empty:
        # 예측 종료일 계산
        pred_end = pd.Series(pd.NaT, index=mv_kpi.index, dtype="datetime64[ns]")
        
        # 1) inbound가 있으면 그 날 종료
        mask_inb = mv_kpi["inbound_date"].notna()
        pred_end.loc[mask_inb] = mv_kpi.loc[mask_inb, "inbound_date"]
        
        # 2) inbound 없고 arrival 있는 경우
        mask_arr = (~mask_inb) & mv_kpi["arrival_date"].notna()
        if mask_arr.any():
            # 2-1) arrival이 과거면: arrival + N일에 종료(가정)
            past_arr = mask_arr & (mv_kpi["arrival_date"] <= today)
            pred_end.loc[past_arr] = (
                mv_kpi.loc[past_arr, "arrival_date"] + pd.Timedelta(days=int(lag_days))
            )
            
            # 2-2) arrival이 미래면: arrival에 종료
            fut_arr = mask_arr & (mv_kpi["arrival_date"] > today)
            pred_end.loc[fut_arr] = mv_kpi.loc[fut_arr, "arrival_date"]
        
        # 3) 그래도 비어있으면: today+1
        pred_end = pred_end.fillna(today + pd.Timedelta(days=1))
        mv_kpi["pred_end_date"] = pred_end
    
    it = (mv_kpi[
        (mv_kpi["carrier_mode"] != "WIP") &
        (mv_kpi["to_center"].isin(centers_sel)) &
        (mv_kpi["resource_code"].isin(skus_sel)) &
        (mv_kpi["onboard_date"].notna()) &
        (mv_kpi["onboard_date"] <= today) &
        (today < mv_kpi["pred_end_date"])  # 오늘 이후까지 이동중인 건만
    ].groupby("resource_code", as_index=True)["qty_ea"].sum())

    # WIP: SKU×날짜별 (onboard +, event -) 누적합을 오늘까지 계산한 잔량
    w = mv[
        (mv["carrier_mode"] == "WIP") &
        (mv["to_center"].isin(centers_sel)) &
        (mv["resource_code"].isin(skus_sel))
    ].copy()
    if w.empty:
        wip = pd.Series(0, index=pd.Index(skus_sel, name="resource_code"))
    else:
        add = (w.dropna(subset=["onboard_date"])
                .set_index(["resource_code","onboard_date"])["qty_ea"])
        rem = (w.dropna(subset=["event_date"])
                .set_index(["resource_code","event_date"])["qty_ea"] * -1)
        flow = pd.concat([add, rem]).groupby(level=[0,1]).sum()
        flow = flow[flow.index.get_level_values(1) <= today]  # 오늘까지만
        wip = (flow.groupby(level=0).cumsum()
                    .groupby(level=0).last()
                    .clip(lower=0))

    out = pd.DataFrame({
        "current": cur,
        "in_transit": it,
        "wip": wip
    }).reindex(skus_sel).fillna(0).astype(int)
    return out

kpi_df = _kpi_breakdown_per_sku(snap_long, mv, centers_sel, skus_sel, _today)

# SKU별 KPI 카드 렌더링
def _chunks(lst, n):  # 줄바꿈을 위해 2~4개씩 끊어서 출력
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

for group in _chunks(skus_sel, 2):  # 한 줄에 2개씩 보기 좋게
    cols = st.columns(len(group))
    for i, sku in enumerate(group):
        with cols[i].container(border=True):
            name = _name_map.get(sku, "")
            if name:
                st.markdown(f"**{name}**  \n`{sku}`")
            else:
                st.markdown(f"`{sku}`")
            c1, c2, c3 = st.columns(3)
            c1.metric("현재 재고", f"{kpi_df.loc[sku,'current']:,}")
            c2.metric("이동중", f"{kpi_df.loc[sku,'in_transit']:,}")
            c3.metric("생산중", f"{kpi_df.loc[sku,'wip']:,}")

# (선택) 전체 합계도 같이 보고 싶으면 아래 4줄을 해제
# total = kpi_df.sum()
# t1, t2, t3 = st.columns(3)
# t1.metric("선택 SKU 현재 재고 합계", f"{total['current']:,}")
# t2.metric("선택 SKU 이동중 합계", f"{total['in_transit']:,}"); t3.metric("선택 SKU 생산중 합계", f"{total['wip']:,}")

st.divider()

# -------------------- Step Chart --------------------
st.subheader("계단식 재고 흐름")
timeline = build_timeline(snap_long, moves, centers_sel, skus_sel,
                          start_dt, end_dt, horizon_days=proj_days_for_build, today=today,
                          lag_days=int(lag_days))

if use_cons_forecast and not timeline.empty:
    timeline = apply_consumption_with_events(
        timeline, snap_long, centers_sel, skus_sel,
        start_dt, end_dt, lookback_days=int(lookback_days), events=events
    )

if timeline.empty:
    st.info("선택 조건에 해당하는 타임라인 데이터가 없습니다.")
else:
    vis_df = timeline[(timeline["date"] >= start_dt) & (timeline["date"] <= end_dt)].copy()
    vis_df["center"] = vis_df["center"].str.replace(r"^In-Transit.*$", "이동중", regex=True)
    vis_df.loc[vis_df["center"] == "WIP", "center"] = "생산중"
    if "태광KR" not in centers_sel:
        vis_df = vis_df[vis_df["center"] != "생산중"]
    if not show_prod:
        vis_df = vis_df[vis_df["center"] != "생산중"]
    if not show_transit:
        vis_df = vis_df[~vis_df["center"].str.startswith("이동중")]
    
    # 재고량이 0보다 큰 데이터만 표시
    vis_df = vis_df[vis_df["stock_qty"] > 0]
    vis_df["label"] = vis_df["resource_code"] + " @ " + vis_df["center"]

    fig = px.line(vis_df, x="date", y="stock_qty", color="label", line_shape="hv",
                  title="선택한 SKU × 센터(및 이동중/생산중) 계단식 재고 흐름", render_mode="svg")
    fig.update_layout(hovermode="x unified", xaxis_title="날짜", yaxis_title="재고량(EA)",
                      legend_title_text="SKU @ Center / 이동중(점선) / 생산중(점선)",
                      margin=dict(l=20, r=20, t=60, b=20))

    if start_dt <= today <= end_dt:
        fig.add_vline(x=today, line_width=1, line_dash="solid", line_color="rgba(255, 0, 0, 0.4)")
        fig.add_annotation(x=today, y=1.02, xref="x", yref="paper", text="오늘",
                           showarrow=False, font=dict(size=12, color="#555"), align="center", yanchor="bottom")

    fig.update_yaxes(tickformat=",.0f")
    fig.update_traces(hovertemplate="날짜: %{x|%Y-%m-%d}<br>재고: %{y:,.0f} EA<br>%{fullData.name}<extra></extra>")

    # 선 스타일
    PALETTE = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
        "#c49c94", "#f7b6d3", "#c7c7c7", "#dbdb8d", "#9edae5",
        "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
        "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac"
    ]
    line_colors = {}
    color_idx = 0
    for tr in fig.data:
        name = tr.name or ""
        if " @ " in name and name not in line_colors:
            line_colors[name] = PALETTE[color_idx % len(PALETTE)]
            color_idx += 1
    for i, tr in enumerate(fig.data):
        name = tr.name or ""
        if " @ " not in name:
            continue
        sku, kind = name.split(" @ ", 1)
        line_color = line_colors.get(name, PALETTE[0])
        if kind == "이동중":
            fig.data[i].update(line=dict(color=line_color, dash="dot", width=1.2), opacity=0.9)
        elif kind == "생산중":
            fig.data[i].update(line=dict(color=line_color, dash="dash", width=1.0), opacity=0.8)
        else:
            fig.data[i].update(line=dict(color=line_color, dash="solid", width=1.5), opacity=1.0)

    chart_key = (f"stepchart|centers={','.join(centers_sel)}|skus={','.join(skus_sel)}|"
                 f"{start_dt:%Y%m%d}-{end_dt:%Y%m%d}|h{int(st.session_state.horizon_days)}|"
                 f"prod{int(show_prod)}|tran{int(show_transit)}")
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False}, key=chart_key)

# -------------------- Upcoming Arrivals (fixed) --------------------
st.subheader("입고 예정 내역 (선택 센터/SKU)")
window_start = start_dt
window_end   = end_dt

# 1) 운송(비 WIP) — 아직 입고완료되지 않은 건만
arr_transport = mv[
    (mv["carrier_mode"] != "WIP") &
    (mv["to_center"].isin(centers_sel)) &
    (mv["resource_code"].isin(skus_sel)) &
    (mv["inbound_date"].isna())                    # ✅ 입고완료 제외
].copy()

# 도착(예정)일: arrival_date(= ETA/도착일) 우선, 없으면 onboard_date 보조
arr_transport["display_date"] = arr_transport["arrival_date"].fillna(arr_transport["onboard_date"])
arr_transport = arr_transport[arr_transport["display_date"].notna()]
arr_transport = arr_transport[
    (arr_transport["display_date"] >= window_start) &
    (arr_transport["display_date"] <= window_end)
]

# 2) WIP — 태광KR일 때만, wip_ready(event_date) 기준
arr_wip = pd.DataFrame()
if "태광KR" in centers_sel:
    arr_wip = mv[
        (mv["carrier_mode"] == "WIP") &
        (mv["to_center"] == "태광KR") &
        (mv["resource_code"].isin(skus_sel)) &
        (mv["event_date"].notna()) &
        (mv["event_date"] >= window_start) &
        (mv["event_date"] <= window_end)
    ].copy()
    arr_wip["display_date"] = arr_wip["event_date"]

# 3) 병합 + 표 렌더
upcoming = pd.concat([arr_transport, arr_wip], ignore_index=True)

# 예상 입고일 추가
mv_view = mv.copy()
if not mv_view.empty:
    # 예측 입고일 계산
    pred_inbound = pd.Series(pd.NaT, index=mv_view.index, dtype="datetime64[ns]")
    
    # 1) inbound가 있으면 그 날 입고
    mask_inb = mv_view["inbound_date"].notna()
    pred_inbound.loc[mask_inb] = mv_view.loc[mask_inb, "inbound_date"]
    
    # 2) inbound 없고 arrival 있는 경우
    mask_arr = (~mask_inb) & mv_view["arrival_date"].notna()
    if mask_arr.any():
        # 2-1) arrival이 과거면: arrival + N일에 입고(가정)
        past_arr = mask_arr & (mv_view["arrival_date"] <= today)
        pred_inbound.loc[past_arr] = (
            mv_view.loc[past_arr, "arrival_date"] + pd.Timedelta(days=int(lag_days))
        )
        
        # 2-2) arrival이 미래면: arrival에 입고
        fut_arr = mask_arr & (mv_view["arrival_date"] > today)
        pred_inbound.loc[fut_arr] = mv_view.loc[fut_arr, "arrival_date"]
    
    mv_view["pred_inbound_date"] = pred_inbound

upcoming = upcoming.merge(
    mv_view[["resource_code","onboard_date","pred_inbound_date"]],
    on=["resource_code","onboard_date"], how="left"
)

# 품명 붙이기 (있을 때만)
if _name_map:
    upcoming["resource_name"] = upcoming["resource_code"].map(_name_map).fillna("")

if upcoming.empty:
    st.caption("도착 예정 없음 (오늘 이후 / 선택 기간)")
else:
    upcoming["days_to_arrival"] = (upcoming["display_date"].dt.normalize() - today).dt.days
    upcoming["days_to_inbound"] = (upcoming["pred_inbound_date"].dt.normalize() - today).dt.days
    upcoming = upcoming.sort_values(["display_date","to_center","resource_code","qty_ea"],
                                    ascending=[True, True, True, False])
    cols = ["display_date","days_to_arrival","to_center","resource_code","resource_name","qty_ea",
            "carrier_mode","onboard_date","pred_inbound_date","days_to_inbound","lot"]
    cols = [c for c in cols if c in upcoming.columns]
    st.dataframe(upcoming[cols].head(1000), use_container_width=True, height=300)
    st.caption("※ days_to_arrival가 음수(–)로 보이면: 화물은 '도착'했으나 인바운드(입고완료) 등록 전 상태입니다.")
    st.caption("※ pred_inbound_date: 예상 입고일 (도착일 + 리드타임), days_to_inbound: 예상 입고까지 남은 일수")

# -------------------- 선택 센터 현재 재고 (전체 SKU) --------------------
st.subheader(f"선택 센터 현재 재고 (스냅샷 {_latest_dt_str} / 전체 SKU)")

cur = snap_long[(snap_long["date"] == _latest_dt) & (snap_long["center"].isin(centers_sel))].copy()
pivot = (cur.groupby(["resource_code","center"], as_index=False)["stock_qty"].sum()
           .pivot(index="resource_code", columns="center", values="stock_qty").fillna(0).astype(int))
pivot["총합"] = pivot.sum(axis=1)

col1, col2 = st.columns([2,1])
with col1:
    q = st.text_input(
        "SKU 필터 — 품목번호 검색 시 해당 SKU의 센터별 제조번호(LOT) 확인",
        "",
        key="sku_filter_text"
    )
with col2:
    sort_by = st.selectbox("정렬 기준", ["총합"] + list(pivot.columns.drop("총합")), index=0)

col1, col2 = st.columns(2)
with col1:
    hide_zero = st.checkbox("총합=0 숨기기", value=True)
with col2:
    show_cost = st.checkbox("재고자산(제조원가) 표시", value=False)

view = pivot.copy()
if q.strip():
    view = view[view.index.str.contains(q.strip(), case=False, regex=False)]
if hide_zero:
    view = view[view["총합"] > 0]
view = view.sort_values(by=sort_by, ascending=False)

base_df = view.reset_index().rename(columns={"resource_code":"SKU"})
if _name_map:
    base_df.insert(1, "품명", base_df["SKU"].map(_name_map).fillna(""))

if show_cost:
    snap_raw_df = load_snapshot_raw()
    cost_pivot = pivot_inventory_cost_from_raw(snap_raw_df, _latest_dt, centers_sel)
    if cost_pivot.empty:
        st.warning("재고자산 계산을 위한 'snapshot_raw' 데이터를 불러올 수 없어 수량만 표시합니다. (엑셀에 'snapshot_raw' 시트가 있으면 자동 사용됩니다)")
        show_df = base_df
    else:
        merged = base_df.merge(cost_pivot.rename(columns={"resource_code":"SKU"}), on="SKU", how="left")
        cost_cols2 = [c for c in merged.columns if c.endswith("_재고자산")] + (["총 재고자산"] if "총 재고자산" in merged.columns else [])
        if cost_cols2:
            merged[cost_cols2] = merged[cost_cols2].fillna(0).astype(int)
            for col in cost_cols2:
                merged[col] = merged[col].apply(lambda x: f"{x:,}원")
        qty_center_cols = [c for c in merged.columns if c not in ["SKU","품명","총합"] + cost_cols2]
        ordered = ["SKU"] + (["품명"] if "품명" in merged.columns else []) + qty_center_cols + (["총합"] if "총합" in merged.columns else []) + cost_cols2
        show_df = merged[ordered]
else:
    show_df = base_df

# 수량 포맷팅
qty_cols = [c for c in show_df.columns if c not in ["SKU"] and not c.endswith("_재고자산") and c != "총 재고자산"]
for col in qty_cols:
    if col in show_df.columns:
        show_df[col] = show_df[col].apply(lambda x: f"{x:,}" if pd.notna(x) and isinstance(x, (int, float)) else x)

st.dataframe(show_df, use_container_width=True, height=380)

csv_bytes = show_df.to_csv(index=False).encode("utf-8-sig")
st.download_button("현재 표 CSV 다운로드", data=csv_bytes,
                   file_name=f"centers_{'-'.join(centers_sel)}_snapshot_{_latest_dt_str}.csv", mime="text/csv")

st.caption("※ 이 표는 **선택된 센터 전체 SKU**의 최신 스냅샷 재고입니다. 상단 'SKU 선택'과 무관하게 모든 SKU가 포함됩니다.")

# === 로트 상세: SKU가 1개일 때 자동 표시 ===
filtered_df = show_df if "SKU" in show_df.columns else view.reset_index().rename(columns={"resource_code":"SKU"})
visible_skus = filtered_df["SKU"].dropna().astype(str).unique().tolist()

if len(visible_skus) == 1:
    lot_sku = visible_skus[0]
    snap_raw_df = load_snapshot_raw()
    # lot 상세 테이블 만들기
    if snap_raw_df is None or snap_raw_df.empty:
        # _latest_dt_str를 여기서 계산
        latest_dt_str = pd.to_datetime(snap_long["date"]).max().strftime("%Y-%m-%d")
        st.markdown(f"### 로트 상세 (스냅샷 {latest_dt_str} / **{', '.join(centers_sel)}** / **{lot_sku}**)")
        st.caption("해당 조건의 로트 상세가 없습니다. (snapshot_raw 없음)")
    else:
        sr = snap_raw_df.copy()
        cols_map = {c.strip().lower(): c for c in sr.columns}
        col_date = cols_map.get("snapshot_date") or cols_map.get("date")
        col_sku  = cols_map.get("resource_code") or cols_map.get("sku") or cols_map.get("상품코드")
        col_lot  = cols_map.get("lot")
        used_centers = [ct for ct in centers_sel if CENTER_COL.get(ct) in sr.columns]
        if not all([col_date, col_sku, col_lot]) or not used_centers:
            st.caption("해당 조건의 로트 상세가 없습니다.")
        else:
            sr[col_date] = pd.to_datetime(sr[col_date], errors="coerce")
            sub = sr[(sr[col_date].dt.normalize()==_latest_dt.normalize()) & (sr[col_sku].astype(str)==str(lot_sku))].copy()
            if sub.empty:
                st.caption("해당 조건의 로트 상세가 없습니다.")
            else:
                for ct in used_centers:
                    c = CENTER_COL[ct]
                    sub[c] = pd.to_numeric(sub[c], errors="coerce").fillna(0).clip(lower=0)
                out = pd.DataFrame({"lot": sub[col_lot].astype(str).fillna("(no lot)")})
                for ct in used_centers:
                    out[ct] = sub.groupby(col_lot)[CENTER_COL[ct]].transform("sum")
                out = out.drop_duplicates()
                out["합계"] = out[used_centers].sum(axis=1)
                out = out[out["합계"] > 0]
                # _latest_dt_str를 여기서 계산
                latest_dt_str = pd.to_datetime(snap_long["date"]).max().strftime("%Y-%m-%d")
                st.markdown(f"### 로트 상세 (스냅샷 {latest_dt_str} / **{', '.join(centers_sel)}** / **{lot_sku}**)")
                if out.empty:
                    st.caption("해당 조건의 로트 상세가 없습니다.")
                else:
                    st.dataframe(out[["lot"] + used_centers + ["합계"]].sort_values("합계", ascending=False).reset_index(drop=True),
                                 use_container_width=True, height=320)
