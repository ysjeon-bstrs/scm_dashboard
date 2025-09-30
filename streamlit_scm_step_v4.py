
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
    """
    Google Sheets API를 사용하여 인증된 방식으로 데이터를 가져옵니다.
    Streamlit Cloud secrets에서 인증 정보를 읽습니다.
    """
    try:
        # Streamlit secrets에서 인증 정보 로드
        import json
        
        # 인증 범위 설정
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets.readonly",
            "https://www.googleapis.com/auth/drive.readonly"
        ]
        
        # Streamlit secrets에서 credentials 정보 가져오기
        try:
            credentials_info = json.loads(st.secrets["google_sheets"]["credentials"])
            credentials = Credentials.from_service_account_info(credentials_info, scopes=scopes)
        except Exception as e:
            # secrets가 없으면 로컬 파일 시도 (개발 환경용)
            credentials_file = "python-spreadsheet-409212-3df25e0dc166.json"
            credentials = Credentials.from_service_account_file(credentials_file, scopes=scopes)
        
        gc = gspread.authorize(credentials)
        
        # 스프레드시트 열기
        spreadsheet = gc.open_by_key(GSHEET_ID)
        
        # 각 시트에서 데이터 가져오기
        df_move = pd.DataFrame()
        df_ref = pd.DataFrame()
        df_incoming = pd.DataFrame()
        
        # SCM_통합 시트
        try:
            worksheet = spreadsheet.worksheet("SCM_통합")
            data = worksheet.get_all_records()
            df_move = pd.DataFrame(data)
        except Exception as e:
            st.warning(f"SCM_통합 시트를 읽을 수 없습니다: {e}")
        
        # snap_정제 시트
        try:
            worksheet = spreadsheet.worksheet("snap_정제")
            data = worksheet.get_all_records()
            df_ref = pd.DataFrame(data)
        except Exception as e:
            st.warning(f"snap_정제 시트를 읽을 수 없습니다: {e}")
        
        # 입고예정내역 시트
        try:
            worksheet = spreadsheet.worksheet("입고예정내역")
            data = worksheet.get_all_records()
            df_incoming = pd.DataFrame(data)
        except Exception as e:
            st.warning(f"입고예정내역 시트를 읽을 수 없습니다: {e}")
        
        return df_move, df_ref, df_incoming
        
    except Exception as e:
        st.error(f"Google Sheets API 연결 실패: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

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
    cols = {c.strip().lower(): c for c in df_ref.columns}
    date_col = next((cols[k] for k in ["date","날짜","snapshot_date","스냅샷일"] if k in cols), None)
    center_col = next((cols[k] for k in ["center","센터","창고","warehouse"] if k in cols), None)
    resource_col = next((cols[k] for k in ["resource_code","sku","상품코드","product_code"] if k in cols), None)
    stock_col = next((cols[k] for k in ["stock_qty","qty","수량","재고","quantity"] if k in cols), None)

    missing = [n for n,v in {"date":date_col,"center":center_col,"resource_code":resource_col,"stock_qty":stock_col}.items() if not v]
    if missing:
        st.error(f"'snap_정제' 시트에 누락된 컬럼: {missing}")
        st.stop()

    result = df_ref.rename(columns={date_col: "date",
                                    center_col:"center",
                                    resource_col:"resource_code",
                                    stock_col:"stock_qty"}).copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.normalize()
    result["center"] = result["center"].astype(str)
    result["resource_code"] = result["resource_code"].astype(str)
    result["stock_qty"] = pd.to_numeric(result["stock_qty"], errors="coerce").fillna(0).astype(int)
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
        "from_center": from_center.astype(str).str.strip(),
        "to_center": to_center.astype(str).str.strip(),
        "onboard_date": onboard_date,
        "arrival_date": arrival_date,
        "inbound_date": inbound_date,
    })
    out["event_date"] = out["inbound_date"].where(out["inbound_date"].notna(), out["arrival_date"])
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
        "lot": wip_df.get("lot", "")
    })
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
                   horizon_days: int = 0, today: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    if today is None:
        today = pd.Timestamp.today().normalize()
    horizon_end = end_dt + pd.Timedelta(days=horizon_days)
    full_dates = pd.date_range(start_dt, horizon_end, freq="D")

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

        mv = moves[moves["resource_code"] == sku].copy()

        eff_minus = (
            mv[(mv["from_center"].astype(str) == str(ct)) &
               (mv["onboard_date"].notna()) &
               (mv["onboard_date"] > last_dt)]
            .groupby("onboard_date", as_index=False)["qty_ea"].sum()
            .rename(columns={"onboard_date":"date","qty_ea":"delta"})
        )
        eff_minus["delta"] *= -1

        # 입고(+) 이벤트 (inbound_date만 인정 - arrival은 미반영)
        eff_plus = (
            mv[(mv["to_center"].astype(str) == str(ct)) &
               (mv["inbound_date"].notna()) &
               (mv["inbound_date"] > last_dt)]
            .groupby("inbound_date", as_index=False)["qty_ea"].sum()
            .rename(columns={"inbound_date":"date","qty_ea":"delta"})
        )

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
    moves_str = moves.copy()
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
        # --- Non-WIP In-Transit (벡터화 + carry-over) ----
        g_nonwip = g[g["carrier_mode"] != "WIP"]
        if not g_nonwip.empty:
            g_selected = g_nonwip[g_nonwip["to_center"].isin(centers_sel)]
            if not g_selected.empty:
                idx = pd.date_range(start_dt, horizon_end, freq="D")
                today_norm = pd.Timestamp.today().normalize()

                # 유효 종료일: inbound > 미래 arrival > (기타) 오늘+1
                end_eff = pd.Series(pd.NaT, index=g_selected.index, dtype="datetime64[ns]")
                mask_inb = g_selected["inbound_date"].notna()
                end_eff.loc[mask_inb] = g_selected.loc[mask_inb, "inbound_date"]

                mask_arr_future = (~mask_inb) & g_selected["arrival_date"].notna() & (g_selected["arrival_date"] > today_norm)
                end_eff.loc[mask_arr_future] = g_selected.loc[mask_arr_future, "arrival_date"]

                end_eff = end_eff.fillna(min(today_norm + pd.Timedelta(days=1), idx[-1] + pd.Timedelta(days=1)))

                # ① 기간 시작 이전에 출발했고, 시작 시점에도 아직 이동중(= 종료>시작)인 물량 → 초기잔
                carry_mask = (
                    g_selected["onboard_date"].notna() &
                    (g_selected["onboard_date"] < idx[0]) &
                    (end_eff > idx[0])
                )
                carry = int(g_selected.loc[carry_mask, "qty_ea"].sum())

                # ② 시작일 이후의 출발 이벤트만 델타로 (이전 출발분은 carry로 처리했으므로 중복 방지)
                starts = (g_selected[g_selected["onboard_date"] >= idx[0]]
                          .groupby("onboard_date")["qty_ea"].sum())

                # ③ 모든 종료 이벤트(마이너스)
                ends = (g_selected.assign(end_date=end_eff)
                        .groupby("end_date")["qty_ea"].sum() * -1)

                delta = (starts.rename_axis("date").to_frame("delta")
                           .add(ends.rename_axis("date").to_frame("delta"), fill_value=0)["delta"]
                           .sort_index())

                s = delta.reindex(idx, fill_value=0).cumsum().clip(lower=0)
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
                                  latest_dt: pd.Timestamp,
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
            st.session_state["_snapshot_raw_cache"] = None

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
            st.session_state["_snapshot_raw_cache"] = None
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
    if center in ["", "nan", "None", "WIP", "In-Transit"]:
        return None
    if center in ["AcrossBUS", "어크로스비US"]:
        return "어크로스비US"
    return center

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

with st.sidebar.expander("프로모션 가중치(+%)", expanded=False):
    enable_event = st.checkbox("가중치 적용", value=False)
    ev_start = st.date_input("시작일")
    ev_end   = st.date_input("종료일")
    ev_pct   = st.number_input("가중치(%)", min_value=-100.0, max_value=300.0, value=30.0, step=5.0)
events = [{"start": pd.Timestamp(ev_start).strftime("%Y-%m-%d"),
           "end":   pd.Timestamp(ev_end).strftime("%Y-%m-%d"),
           "uplift": ev_pct/100.0}] if enable_event else []

# -------------------- KPIs --------------------
st.subheader("요약 KPI")
latest_dt = snap_long["date"].max()
latest_dt_str = pd.to_datetime(latest_dt).strftime("%Y-%m-%d")

latest_rows = snap_long[(snap_long["date"] == latest_dt) & (snap_long["center"].isin(centers_sel))]
sku_totals = {sku: int(latest_rows[latest_rows["resource_code"]==sku]["stock_qty"].sum()) for sku in skus_sel} if not latest_rows.empty else {sku:0 for sku in skus_sel}

today = pd.Timestamp.today().normalize()
moves_typed = moves.copy()
moves_typed["carrier_mode"] = moves_typed["carrier_mode"].astype(str).str.upper()

# KPI 이동중 재고: 도착했지만 인바운드 미등록도 포함
in_transit_mask = (
    (moves_typed["carrier_mode"] != "WIP") &
    (moves_typed["to_center"].isin(centers_sel)) &
    (moves_typed["resource_code"].isin(skus_sel)) &
    (moves_typed["onboard_date"].notna()) &
    (moves_typed["onboard_date"] <= today) &
    (moves_typed["inbound_date"].isna())   # arrival 여부와 무관
)
in_transit_total = int(moves_typed[in_transit_mask]["qty_ea"].sum())


wip_moves = moves_typed[
    (moves_typed["carrier_mode"] == "WIP") &
    (moves_typed["to_center"].isin(centers_sel)) &
    (moves_typed["resource_code"].isin(skus_sel))
]
if not wip_moves.empty:
    on = (wip_moves.dropna(subset=["onboard_date"]).groupby("onboard_date", as_index=True)["qty_ea"].sum())
    ev = (wip_moves.dropna(subset=["event_date"]).groupby("event_date", as_index=True)["qty_ea"].sum() * -1)
    wip_flow = pd.concat([on, ev]).groupby(level=0).sum().sort_index()
    wip_cum = wip_flow[wip_flow.index <= today].cumsum()
    wip_today = int(wip_cum.iloc[-1]) if not wip_cum.empty else 0
else:
    wip_today = 0

def _chunk(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

for group in _chunk(skus_sel, 4):
    cols = st.columns(len(group))
    for i, sku in enumerate(group):
        cols[i].metric(f"{sku} 현재 재고(스냅샷 {latest_dt_str})", f"{sku_totals.get(sku, 0):,}")

k_it, k_wip = st.columns(2)
k_it.metric("이동 중 재고", f"{in_transit_total:,}")
k_wip.metric("생산중(미완료)", f"{wip_today:,}")

st.divider()

# -------------------- Step Chart --------------------
st.subheader("계단식 재고 흐름")
timeline = build_timeline(snap_long, moves, centers_sel, skus_sel,
                          start_dt, end_dt, horizon_days=proj_days_for_build, today=today)

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

# -------------------- Upcoming Arrivals --------------------
st.subheader("입고 예정 내역 (선택 센터/SKU)")
window_start = start_dt
window_end   = end_dt

arr_transport_conditions = [
    (moves_typed["carrier_mode"] != "WIP"),
    (moves_typed["to_center"].isin(centers_sel)),
    (moves_typed["resource_code"].isin(skus_sel))
]

# 날짜 조건 (event_date → arrival_date → eta_date)
date_conditions = [
    (moves_typed["event_date"].notna()) & (moves_typed["event_date"] >= window_start) & (moves_typed["event_date"] <= window_end)
]
if "arrival_date" in moves_typed.columns:
    date_conditions.append(
        (moves_typed["event_date"].isna()) & (moves_typed["arrival_date"].notna()) &
        (moves_typed["arrival_date"] >= window_start) & (moves_typed["arrival_date"] <= window_end)
    )
if "eta_date" in moves_typed.columns:
    arrival_isna = moves_typed["arrival_date"].isna() if "arrival_date" in moves_typed.columns else True
    date_conditions.append(
        (moves_typed["event_date"].isna()) & arrival_isna & (moves_typed["eta_date"].notna()) &
        (moves_typed["eta_date"] >= window_start) & (moves_typed["eta_date"] <= window_end)
    )

date_condition = date_conditions[0]
for cond in date_conditions[1:]:
    date_condition = date_condition | cond
arr_transport_conditions.append(date_condition)

arr_transport = moves_typed[arr_transport_conditions[0]]
for cond in arr_transport_conditions[1:]:
    arr_transport = arr_transport[cond]

arr_wip = pd.DataFrame()
if "태광KR" in centers_sel:
    arr_wip = moves_typed[
        (moves_typed["event_date"].notna()) &
        (moves_typed["event_date"] >= window_start) & (moves_typed["event_date"] <= window_end) &
        (moves_typed["carrier_mode"] == "WIP") &
        (moves_typed["to_center"] == "태광KR") &
        (moves_typed["resource_code"].isin(skus_sel))
    ]

upcoming = pd.concat([arr_transport, arr_wip], ignore_index=True)

if not upcoming.empty:
    st.info(f"📊 총 {len(upcoming)}건의 입고 예정 내역이 있습니다.")
else:
    st.info("📊 선택된 조건에 해당하는 입고 예정 내역이 없습니다.")

if upcoming.empty:
    st.caption("도착 예정 없음 (오늘 이후 / 선택 기간)")
else:
    display_date = upcoming["event_date"]
    if "arrival_date" in upcoming.columns:
        display_date = display_date.fillna(upcoming["arrival_date"])
    if "eta_date" in upcoming.columns:
        display_date = display_date.fillna(upcoming["eta_date"])
    upcoming["display_date"] = pd.to_datetime(display_date)
    upcoming["days_to_arrival"] = (upcoming["display_date"] - today).dt.days
    upcoming = upcoming.sort_values(["display_date","to_center","resource_code","qty_ea"],
                                    ascending=[True,True,True,False])
    cols = ["display_date","days_to_arrival","to_center","resource_code","qty_ea","carrier_mode","onboard_date","lot"]
    cols = [c for c in cols if c in upcoming.columns]
    st.dataframe(upcoming[cols].head(1000), use_container_width=True, height=300)
    
    # ✅ 안내 문구 추가
    st.caption("※ days_to_arrival가 음수(–)로 보이면: 화물은 '도착'했으나 인바운드(입고완료) 등록 전 상태입니다.")

# -------------------- 선택 센터 현재 재고 (전체 SKU) --------------------
st.subheader(f"선택 센터 현재 재고 (스냅샷 {latest_dt_str} / 전체 SKU)")

cur = snap_long[(snap_long["date"] == latest_dt) & (snap_long["center"].isin(centers_sel))].copy()
pivot = (cur.groupby(["resource_code","center"], as_index=False)["stock_qty"].sum()
           .pivot(index="resource_code", columns="center", values="stock_qty").fillna(0).astype(int))
pivot["총합"] = pivot.sum(axis=1)

col1, col2 = st.columns([2,1])
with col1:
    q = st.text_input(
        "SKU 필터(포함 검색) — 검색 시 해당 SKU의 센터별 제조번호(LOT) 확인",
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

if show_cost:
    snap_raw_df = load_snapshot_raw()
    cost_pivot = pivot_inventory_cost_from_raw(snap_raw_df, latest_dt, centers_sel)
    if cost_pivot.empty:
        st.warning("재고자산 계산을 위한 'snapshot_raw' 데이터를 불러올 수 없어 수량만 표시합니다. (엑셀에 'snapshot_raw' 시트가 있으면 자동 사용됩니다)")
        show_df = view.reset_index().rename(columns={"resource_code":"SKU"})
    else:
        cost_cols = [c for c in cost_pivot.columns if c.endswith("_재고자산")]
        cost_pivot["총 재고자산"] = cost_pivot[cost_cols].sum(axis=1).astype(int)
        merged = (view.reset_index().rename(columns={"resource_code":"SKU"})
                    .merge(cost_pivot.rename(columns={"resource_code":"SKU"}), on="SKU", how="left"))
        cost_cols2 = [c for c in merged.columns if c.endswith("_재고자산")] + (["총 재고자산"] if "총 재고자산" in merged.columns else [])
        if cost_cols2:
            merged[cost_cols2] = merged[cost_cols2].fillna(0).astype(int)
            for col in cost_cols2:
                merged[col] = merged[col].apply(lambda x: f"{x:,}원" if pd.notna(x) else "0원")
        qty_center_cols = [c for c in merged.columns if c not in ["SKU","총합"] + cost_cols2]
        ordered = ["SKU"] + qty_center_cols + (["총합"] if "총합" in merged.columns else []) + cost_cols2
        merged = merged[ordered]
        show_df = merged
else:
    show_df = view.reset_index().rename(columns={"resource_code":"SKU"})

# 수량 포맷팅
qty_cols = [c for c in show_df.columns if c not in ["SKU"] and not c.endswith("_재고자산") and c != "총 재고자산"]
for col in qty_cols:
    if col in show_df.columns:
        show_df[col] = show_df[col].apply(lambda x: f"{x:,}" if pd.notna(x) and isinstance(x, (int, float)) else x)

st.dataframe(show_df, use_container_width=True, height=380)

csv_bytes = show_df.to_csv(index=False).encode("utf-8-sig")
st.download_button("현재 표 CSV 다운로드", data=csv_bytes,
                   file_name=f"centers_{'-'.join(centers_sel)}_snapshot_{latest_dt_str}.csv", mime="text/csv")

st.caption("※ 이 표는 **선택된 센터 전체 SKU**의 최신 스냅샷 재고입니다. 상단 'SKU 선택'과 무관하게 모든 SKU가 포함됩니다.")

# === 로트 상세: SKU가 1개일 때 자동 표시 ===
filtered_df = show_df if "SKU" in show_df.columns else view.reset_index().rename(columns={"resource_code":"SKU"})
visible_skus = filtered_df["SKU"].dropna().astype(str).unique().tolist()

if len(visible_skus) == 1:
    lot_sku = visible_skus[0]
    snap_raw_df = load_snapshot_raw()
    # lot 상세 테이블 만들기
    if snap_raw_df is None or snap_raw_df.empty:
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
            sub = sr[(sr[col_date].dt.normalize()==latest_dt.normalize()) & (sr[col_sku].astype(str)==str(lot_sku))].copy()
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
                st.markdown(f"### 로트 상세 (스냅샷 {latest_dt_str} / **{', '.join(centers_sel)}** / **{lot_sku}**)")
                if out.empty:
                    st.caption("해당 조건의 로트 상세가 없습니다.")
                else:
                    st.dataframe(out[["lot"] + used_centers + ["합계"]].sort_values("합계", ascending=False).reset_index(drop=True),
                                 use_container_width=True, height=320)
