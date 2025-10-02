
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import re
from datetime import datetime, timedelta
from io import BytesIO
from typing import Dict, List, Optional, Tuple
import json
import requests
import os
import time

# === Apps Script Web App 프록시 설정 ===
def _get_proxy_cfg():
    """WebApp 프록시 설정 로드"""
    if "gs_proxy" in st.secrets:
        cfg = st.secrets["gs_proxy"]
        return dict(
            webapp_url = cfg.get("webapp_url", "").strip(),
            sheet_id   = cfg.get("sheet_id", "").strip(),
            token      = cfg.get("token", "").strip(),
            timeout_s  = int(cfg.get("timeout_s", 15)),
        )
    # 시크릿이 없다면 환경변수로도 지원
    return dict(
        webapp_url = os.getenv("GS_WEBAPP_URL","").strip(),
        sheet_id   = os.getenv("GS_SHEET_ID","").strip(),
        token      = os.getenv("GS_TOKEN","").strip(),
        timeout_s  = int(os.getenv("GS_TIMEOUT","15")),
    )

def _fetch_sheet_via_webapp(tab_name: str) -> pd.DataFrame:
    """Apps Script Web App을 통해 시트 데이터 가져오기"""
    cfg = _get_proxy_cfg()
    if not (cfg["webapp_url"] and cfg["sheet_id"] and cfg["token"]):
        st.error("WebApp 프록시 설정이 누락되었습니다. (webapp_url/sheet_id/token)")
        return pd.DataFrame()

    params = {
        "id": cfg["sheet_id"],
        "sheet": tab_name,
        "token": cfg["token"],
        "_": str(int(time.time()))  # 캐시 무력화
    }

    # 간단 백오프 재시도
    last_err = None
    for i in range(3):
        try:
            r = requests.get(cfg["webapp_url"], params=params, timeout=cfg["timeout_s"])
            r.raise_for_status()
            data = r.json()
            if not data.get("ok"):
                raise RuntimeError(f"Proxy error: {data.get('error')}")
            rows = data.get("rows", [])
            return pd.DataFrame(rows)
        except Exception as e:
            last_err = e
            time.sleep(0.6 * (i+1))
    st.error(f"프록시 호출 실패: {last_err}")
    return pd.DataFrame()

def gs_csv(sheet_name: str) -> pd.DataFrame:
    """Apps Script Web App을 사용하여 시트 데이터 로드"""
    return _fetch_sheet_via_webapp(sheet_name)

# 센터별 원본 컬럼 매핑
CENTER_COL = {
    "태광KR": "stock2",
    "AMZUS": "fba_available_stock",
    "품고KR": "poomgo_v2_available_stock",
    "SBSPH": "shopee_ph_available_stock",
    "SBSSG": "shopee_sg_available_stock",
    "SBSMY": "shopee_my_available_stock",
    "AcrossBUS": "acrossb_available_stock",
}

@st.cache_data(ttl=300)
def load_from_gsheet():
    df_move = gs_csv("SCM_통합")             # 이동 원장
    df_ref  = gs_csv("snap_정제")           # 정제 스냅샷 (long: date|center|resource_code|stock_qty)
    # 입고 예정은 있으면 사용, 없으면 빈 DF
    try:
        df_incoming = gs_csv("입고예정내역")
    except Exception:
        df_incoming = pd.DataFrame()
    return df_move, df_ref, df_incoming

@st.cache_data(ttl=300)
def load_snapshot_raw():
    # 로트 상세용 on-demand
    try:
        return gs_csv("snapshot_raw")
    except Exception:
        return pd.DataFrame()

def build_lot_detail(snap_raw: pd.DataFrame, date_latest: pd.Timestamp, sku: str, centers: list[str]) -> pd.DataFrame:
    if snap_raw is None or snap_raw.empty:
        return pd.DataFrame(columns=["lot"] + centers + ["합계"])

    sr = snap_raw.copy()
    # 날짜/헤더 유연 인식
    cols = {c.strip().lower(): c for c in sr.columns}
    col_date = cols.get("snapshot_date") or cols.get("date")
    col_sku  = cols.get("resource_code") or cols.get("sku") or cols.get("상품코드")
    col_lot  = cols.get("lot")

    if not all([col_date, col_sku, col_lot]):
        return pd.DataFrame(columns=["lot"] + centers + ["합계"])

    # 날짜 정규화
    sr[col_date] = pd.to_datetime(sr[col_date], errors="coerce")
    # 필터: 최신일 + SKU
    sub = sr[(sr[col_date].dt.normalize()==date_latest.normalize()) & (sr[col_sku].astype(str)==str(sku))].copy()

    if sub.empty:
        return pd.DataFrame(columns=["lot"] + centers + ["합계"])

    # 센터별 수량 컬럼(없으면 건너뜀)
    used_centers = []
    for ct in centers:
        if CENTER_COL.get(ct) in sr.columns:
            used_centers.append(ct)
    if not used_centers:
        return pd.DataFrame(columns=["lot"] + centers + ["합계"])

    # 숫자화 + 음수 0 클립
    for ct in used_centers:
        c = CENTER_COL[ct]
        sub[c] = pd.to_numeric(sub[c], errors="coerce").fillna(0).clip(lower=0)

    # lot별 합계
    out = pd.DataFrame({"lot": sub[col_lot].astype(str).fillna("(no lot)")})
    for ct in used_centers:
        out[ct] = sub.groupby(col_lot)[CENTER_COL[ct]].transform("sum")
    out = out.drop_duplicates()
    out["합계"] = out[used_centers].sum(axis=1)
    # 선택 센터 외는 0 열 추가(보기용)
    for ct in centers:
        if ct not in used_centers:
            out[ct] = 0
    # 수량 0인 로트 제거
    out = out[out["합계"] > 0]
    
    # 정렬
    return out[["lot"] + centers + ["합계"]].sort_values("합계", ascending=False).reset_index(drop=True)

def pivot_inventory_cost_from_raw(snap_raw: pd.DataFrame,
                                  latest_dt: pd.Timestamp,
                                  centers: list[str]) -> pd.DataFrame:
    """
    snapshot_raw의 최신일(latest_dt)에서 lot별 COGS × 각 센터 수량을 합산해
    SKU×센터 비용 피벗을 반환. (단위: 원가 기준 금액)
    """
    if snap_raw is None or snap_raw.empty:
        return pd.DataFrame(columns=["resource_code"] + [f"{c}_비용" for c in centers])

    df = snap_raw.copy()
    cols = {str(c).strip().lower(): c for c in df.columns}

    col_date = cols.get("snapshot_date") or cols.get("date")
    col_sku  = cols.get("resource_code") or cols.get("sku") or cols.get("상품코드")
    col_cogs = cols.get("cogs")  # 제조원가
    if not all([col_date, col_sku, col_cogs]):
        return pd.DataFrame(columns=["resource_code"] + [f"{c}_비용" for c in centers])

    # 타입 정규화
    df[col_date] = pd.to_datetime(df[col_date], errors="coerce").dt.normalize()
    df[col_sku]  = df[col_sku].astype(str)
    df[col_cogs] = pd.to_numeric(df[col_cogs], errors="coerce").fillna(0).clip(lower=0)

    # 최신 스냅샷만
    sub = df[df[col_date] == latest_dt.normalize()].copy()
    if sub.empty:
        return pd.DataFrame(columns=["resource_code"] + [f"{c}_비용" for c in centers])

    # 센터별 비용 계산: sum_lot( cogs × qty_center )
    cost_cols = {}
    for ct in centers:
        src_col = CENTER_COL.get(ct)
        if not src_col or src_col not in sub.columns:
            continue
        qty = pd.to_numeric(sub[src_col], errors="coerce").fillna(0).clip(lower=0)
        cost = (sub[col_cogs] * qty)  # 단순 COGS × 수량
        # SKU별 합계
        g = sub[[col_sku]].copy()
        g[f"{ct}_재고자산"] = cost
        cost_cols[ct] = g.groupby(col_sku, as_index=False)[f"{ct}_재고자산"].sum()

    if not cost_cols:
        return pd.DataFrame(columns=["resource_code"] + [f"{c}_재고자산" for c in centers])

    # 병합: SKU 기준으로 모든 센터 비용 붙이기
    base = pd.DataFrame({"resource_code": pd.unique(sub[col_sku])})
    for ct, g in cost_cols.items():
        base = base.merge(g.rename(columns={col_sku: "resource_code"}), on="resource_code", how="left")
    # 누락 0, 정수 반올림
    num_cols = [c for c in base.columns if c.endswith("_재고자산")]
    base[num_cols] = base[num_cols].fillna(0).round().astype(int)
    return base

st.set_page_config(page_title="글로벌 대시보드 — v4", layout="wide")

st.title("📦 SCM 재고 흐름 대시보드 — v4")

st.caption("현재 재고는 항상 **스냅샷 기준(snap_정제)**입니다. 이동중 / 생산중 라인은 예측용 가상 라인입니다. ‘생산중(미완료)’ 그래프는 **태광KR 센터 선택 시에만** 표시됩니다.")

# -------------------- Helpers --------------------
def _coalesce_columns(df: pd.DataFrame, candidates: List, parse_date: bool = False) -> pd.Series:
    all_names = []
    for item in candidates:
        if isinstance(item, (list, tuple, set)):
            all_names.extend([str(x).strip() for x in item])
        else:
            all_names.append(str(item).strip())
    
    # 1단계: 정확한 매칭 시도
    cols = [c for c in df.columns if str(c).strip() in all_names]
    
    # 2단계: 대소문자 무시 매칭 시도
    if not cols:
        cols = [c for c in df.columns if any(name.lower() in str(c).lower() for name in all_names)]
    
    # 3단계: 부분 문자열 매칭 시도
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

@st.cache_data(ttl=300)
def load_from_excel(file):
    data = file.read() if hasattr(file, "read") else file
    bio = BytesIO(data) if isinstance(data, (bytes, bytearray)) else file

    xl = pd.ExcelFile(bio, engine="openpyxl")
    # 필수: 이동데이터 시트(이름 유지), 정제 스냅샷 시트(여러 후보명 지원), 입고예정(선택)
    need = {"SCM_통합": None}
    for s in xl.sheet_names:
        if s == "SCM_통합":
            need["SCM_통합"] = s

    # 정제 스냅샷 후보명
    refined_name = next((s for s in xl.sheet_names if s in ["snap_정제","snap_refined","snap_refine","snap_ref"]), None)
    if refined_name is None:
        st.error("엑셀에 정제 스냅샷 시트가 필요합니다. (시트명: 'snap_정제' 또는 'snap_refined')")
        st.stop()

    df_move = pd.read_excel(bio, sheet_name=need["SCM_통합"], engine="openpyxl")
    bio.seek(0)
    df_ref = pd.read_excel(bio, sheet_name=refined_name, engine="openpyxl")
    bio.seek(0)

    wip_df = None
    if "입고예정내역" in xl.sheet_names:
        wip_df = pd.read_excel(bio, sheet_name="입고예정내역", engine="openpyxl")
    return df_move, df_ref, wip_df

def normalize_refined_snapshot(df_ref: pd.DataFrame) -> pd.DataFrame:
    """정제 스냅샷 시트 → 필요한 타입 보정"""
    # 컬럼명 매핑 (복사 없이)
    cols = {c.strip(): c for c in df_ref.columns}
    req = ["date","center","resource_code","stock_qty"]
    miss = [c for c in req if c not in cols]
    if miss:
        st.error(f"'snap_정제' 시트에 누락된 컬럼: {miss}")
        st.stop()
    # 이름 정규화 및 데이터 타입 변환 (한 번에 처리)
    result = df_ref.rename(columns={cols["date"]:"date",
                                   cols["center"]:"center",
                                   cols["resource_code"]:"resource_code",
                                   cols["stock_qty"]:"stock_qty"})
    
    # 벡터화된 연산으로 성능 향상
    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.normalize()
    result["center"] = result["center"].astype(str)
    result["resource_code"] = result["resource_code"].astype(str)
    result["stock_qty"] = pd.to_numeric(result["stock_qty"], errors="coerce").fillna(0).astype(int)
    
    return result.dropna(subset=["date","center","resource_code"])


def normalize_moves(df: pd.DataFrame) -> pd.DataFrame:
    # 컬럼명 정규화 (in-place)
    df.columns = [str(c).strip() for c in df.columns]

    resource_code = _coalesce_columns(df, [["resource_code","상품코드","RESOURCE_CODE","sku","SKU"]])
    qty_ea       = _coalesce_columns(df, [
        ["qty_ea","QTY_EA","수량(EA)","qty","QTY","quantity","Quantity","수량","EA","ea"]
    ])
    carrier_mode = _coalesce_columns(df, [["carrier_mode","운송방법","carrier mode","운송수단"]])
    from_center  = _coalesce_columns(df, [["from_center","출발창고","from center"]])
    to_center    = _coalesce_columns(df, [["to_center","도착창고","to center"]])
    onboard_date = _coalesce_columns(df, [["onboard_date","배정일","출발일","H","onboard","depart_date"]], parse_date=True)
    arrival_date = _coalesce_columns(df, [["arrival_date","도착일","eta_date","ETA","arrival"]], parse_date=True)
    inbound_date = _coalesce_columns(df, [["inbound_date","입고일","입고완료일"]], parse_date=True)
    real_depart  = _coalesce_columns(df, [["실제 선적일","real_departure","AI"]], parse_date=True)

    out = pd.DataFrame({
        "resource_code": resource_code.astype(str).str.strip(),
        "qty_ea": pd.to_numeric(qty_ea.astype(str).str.replace(',', ''), errors="coerce").fillna(0).astype(int),
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
    """
    예: 'T250812-0001' -> 2025-08-12
    규칙: 영문 1자 + YYMMDD + '-' ...
    """
    if not isinstance(po_str, str):
        return pd.NaT
    m = re.search(r"[A-Za-z](\d{2})(\d{2})(\d{2})", po_str)
    if not m:
        return pd.NaT
    yy, mm, dd = m.groups()
    year = 2000 + int(yy)  # 20xx 가정
    try:
        return pd.Timestamp(datetime(year, int(mm), int(dd)))
    except Exception:
        return pd.NaT

def load_wip_from_incoming(df_incoming: Optional[pd.DataFrame], default_center: str = "태광KR") -> pd.DataFrame:
    """
    '입고예정내역' 시트 정규화:
      - wip_start = po_no(발주번호)에서 파싱한 날짜(없으면 wip_ready-10일)
      - wip_ready = intended_push_date
      - qty_ea    = quantity / total_quantity
      - to_center = 태광KR (고정)
      - lot       = 제조번호(선택)
    """
    if df_incoming is None or df_incoming.empty:
        return pd.DataFrame()

    df = df_incoming.copy()
    df.columns = [str(c).strip() for c in df.columns]
    low = {c.lower(): c for c in df.columns}

    # --- robust column pickers ---
    po_col   = next((low[k] for k in low if k in ["po_no","ponumber","po"]), None)
    sku_col  = next((low[k] for k in low if k in ["product_code","resource_code","상품코드"]), None)

    # 수량: quantity/qty/수량/total_quantity 中 "숫자로 잘 파싱되는" 컬럼만 채택
    qty_col = None
    for key in ["quantity","qty","수량","total_quantity","입고수량"]:
        if key in low:
            s = pd.to_numeric(df[low[key]].astype(str).str.replace(",",""), errors="coerce")
            if s.notna().sum() >= max(1, int(len(s)*0.6)):
                qty_col = low[key]; break

    # 날짜: '입고예정', '예정일', 'ready', 'push', 'intended_push' 포함 &
    #      "수량/qty" 같은 단어 포함 컬럼은 제외 &
    #      "날짜로 60% 이상 파싱"되는 컬럼만 채택
    date_col = None
    
    # 디버깅: 사용 가능한 컬럼들 확인
    st.write("🔍 사용 가능한 컬럼들:", df.columns.tolist())
    
    for p in ["입고예정", "예정일", "ready", "push", "intended_push", "입고일", "eta"]:
        cands = [c for c in df.columns
                 if (p.lower() in c.lower())
                 and not any(x in c.lower() for x in ["수량","qty","quantity"])]
        st.write(f"🔍 '{p}' 패턴 매칭 컬럼들:", cands)
        
        for c in cands:
            s = pd.to_datetime(df[c], errors="coerce")
            success_rate = s.notna().sum() / len(s) if len(s) > 0 else 0
            st.write(f"🔍 컬럼 '{c}' 파싱 성공률: {success_rate:.2%} ({s.notna().sum()}/{len(s)})")
            st.write(f"🔍 컬럼 '{c}' 샘플 값들:", df[c].head().tolist())
            
            if s.notna().sum() >= max(1, int(len(s)*0.6)):
                date_col = c
                st.write(f"✅ 날짜 컬럼 선택: '{c}'")
                break
        if date_col: break
    
    # 디버깅: 최종 선택된 컬럼들 확인
    st.write(f"🔍 최종 선택된 컬럼들 - date_col: '{date_col}', sku_col: '{sku_col}', qty_col: '{qty_col}'")
    
    if date_col:
        st.write(f"🔍 선택된 날짜 컬럼 '{date_col}'의 샘플 값들:", df[date_col].head().tolist())
        st.write(f"🔍 선택된 날짜 컬럼 '{date_col}'의 파싱 결과:", pd.to_datetime(df[date_col], errors="coerce").head().tolist())
    if not (date_col and sku_col and qty_col):
        return pd.DataFrame()

    out = pd.DataFrame({
        "resource_code": df[sku_col].astype(str).str.strip(),
        "to_center": default_center,
        "wip_ready": pd.to_datetime(df[date_col], errors="coerce").dt.normalize(),
        "qty_ea": pd.to_numeric(df[qty_col].astype(str).str.replace(',',''), errors="coerce").fillna(0).astype(int),
        "lot": df[low.get("lot")] if "lot" in low else ""
    })

    # wip_start: PO에서 파싱 or wip_ready-10일
    def _parse_po_date(po):
        m = re.search(r"[A-Za-z](\d{2})(\d{2})(\d{2})", str(po))
        return (pd.Timestamp(2000+int(m.group(1)), int(m.group(2)), int(m.group(3)))
                if m else pd.NaT)
    out["wip_start"] = df[po_col].map(_parse_po_date) if po_col else pd.NaT
    # 발주일이 못 읽혔으면(예외 케이스) 최소한 wip_ready - 30일로 보정
    mask_na = out["wip_start"].isna() & out["wip_ready"].notna()
    out.loc[mask_na, "wip_start"] = out.loc[mask_na, "wip_ready"] - pd.to_timedelta(30, unit="D")

    # 유효값만
    out = out.dropna(subset=["resource_code","wip_ready","wip_start"]).reset_index(drop=True)
    return out[["resource_code","to_center","wip_start","wip_ready","qty_ea","lot"]]

def merge_wip_as_moves(moves_df: pd.DataFrame, wip_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if wip_df is None or wip_df.empty:
        return moves_df
    # 디버깅: wip_df 상태 확인
    st.write(f"🔍 merge_wip_as_moves - wip_df 상태: {len(wip_df)}건")
    if not wip_df.empty:
        st.write("wip_df 샘플:", wip_df[["resource_code", "wip_start", "wip_ready", "qty_ea"]].head())
        st.write("wip_ready 상태:", wip_df["wip_ready"].isna().sum(), "개가 NaT")
        st.write("wip_ready 값들:", wip_df["wip_ready"].tolist())
    
    # 디버깅: wip_df의 wip_ready 상태 확인
    st.write(f"🔍 merge_wip_as_moves - wip_ready 상태: {wip_df['wip_ready'].isna().sum()}개가 NaT")
    st.write("wip_ready 샘플:", wip_df["wip_ready"].head().tolist())
    
    # 디버깅: wip_df의 BA00021 데이터 확인
    ba00021_wip_df = wip_df[wip_df["resource_code"] == "BA00021"]
    st.write(f"🔍 merge_wip_as_moves - BA00021 wip_df: {len(ba00021_wip_df)}건")
    if not ba00021_wip_df.empty:
        st.write("BA00021 wip_df 샘플:", ba00021_wip_df[["resource_code", "wip_start", "wip_ready", "qty_ea"]].head())
        st.write("BA00021 wip_ready 상태:", ba00021_wip_df["wip_ready"].isna().sum(), "개가 NaT")
        st.write("BA00021 wip_ready 값들:", ba00021_wip_df["wip_ready"].tolist())
    
    wip_moves = pd.DataFrame({
        "resource_code": wip_df["resource_code"],
        "qty_ea": wip_df["qty_ea"].astype(int),
        "carrier_mode": "WIP",
        "from_center": "WIP",
        "to_center": wip_df["to_center"],
        "onboard_date": wip_df["wip_start"].dt.normalize(),
        "arrival_date": wip_df["wip_ready"].dt.normalize(),
        "inbound_date": pd.NaT,
        "real_departure": pd.NaT,
        "event_date": wip_df["wip_ready"].dt.normalize(),
        "lot": wip_df.get("lot", "")
    })
    
    # 디버깅: wip_moves의 event_date 상태 확인
    st.write(f"🔍 merge_wip_as_moves - wip_moves event_date 상태: {wip_moves['event_date'].isna().sum()}개가 NaT")
    st.write("wip_moves event_date 샘플:", wip_moves["event_date"].head().tolist())
    
    # 날짜 정규화 (일 단위로 통일)
    wip_moves["onboard_date"] = pd.to_datetime(wip_moves["onboard_date"]).dt.normalize()
    wip_moves["arrival_date"] = pd.to_datetime(wip_moves["arrival_date"]).dt.normalize()
    wip_moves["event_date"] = pd.to_datetime(wip_moves["event_date"]).dt.normalize()
    
    # 디버깅: moves_df와 wip_moves의 컬럼 구조 확인
    st.write(f"🔍 merge_wip_as_moves - moves_df 컬럼들: {moves_df.columns.tolist()}")
    st.write(f"🔍 merge_wip_as_moves - wip_moves 컬럼들: {wip_moves.columns.tolist()}")
    
    # 디버깅: moves_df의 event_date 상태 확인
    if "event_date" in moves_df.columns:
        st.write(f"🔍 merge_wip_as_moves - moves_df event_date 상태: {moves_df['event_date'].isna().sum()}개가 NaT")
    else:
        st.write("🔍 merge_wip_as_moves - moves_df에 event_date 컬럼이 없음!")
    
    # 디버깅: wip_moves의 event_date 상태 확인
    st.write(f"🔍 merge_wip_as_moves - wip_moves event_date 상태: {wip_moves['event_date'].isna().sum()}개가 NaT")
    
    result = pd.concat([moves_df, wip_moves], ignore_index=True)
    
    # 디버깅: 병합 결과의 event_date 상태 확인
    st.write(f"🔍 merge_wip_as_moves - 병합 결과 event_date 상태: {result['event_date'].isna().sum()}개가 NaT")
    if "BA00021" in result["resource_code"].values:
        ba00021_result = result[result["resource_code"] == "BA00021"]
        st.write(f"🔍 merge_wip_as_moves - BA00021 병합 결과: {len(ba00021_result)}건")
        st.write("BA00021 병합 결과 event_date 상태:", ba00021_result["event_date"].isna().sum(), "개가 NaT")
        st.write("BA00021 병합 결과 event_date 값들:", ba00021_result["event_date"].head().tolist())
    
    return result

# ===== 소비(소진) 추세 + 이벤트 가중치 =====

@st.cache_data(ttl=3600)  # 1시간 캐시 (하루 1회 갱신에 적합)
def estimate_daily_consumption(snap_long: pd.DataFrame,
                               centers_sel: List[str], skus_sel: List[str],
                               asof_dt: pd.Timestamp,
                               lookback_days: int = 28) -> Dict[Tuple[str, str], float]:
    """
    최근 lookback_days 동안 스냅샷으로 SKU×센터별 일일 소진량(개/일)을 추정.
    회귀 기울기와 감소분 평균의 max를 사용하여 '입고 후 평평' 구간에서도 안정적 추정.
    """
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
        # 관측 최소 보장
        if ts.dropna().shape[0] < max(7, lookback_days//2):
            continue
        
        x = np.arange(len(ts), dtype=float)
        y = ts.values.astype(float)
        
        # 방법1: 회귀 기울기(감소만)
        rate_reg = max(0.0, -np.polyfit(x, y, 1)[0])
        
        # 방법2: 일/일 감소분 평균
        dec = (-np.diff(y)).clip(min=0)
        rate_dec = dec.mean() if len(dec) else 0.0
        
        # 두 방법의 max 사용
        rate = max(rate_reg, rate_dec)
        if rate > 0:
            rates[(ct, sku)] = float(rate)
    
    return rates

@st.cache_data(ttl=1800)  # 30분 캐시
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

    # 날짜 컬럼 자동 감지 (date / snapshot_date 모두 지원)
    snap_cols = {c.lower(): c for c in snap_long.columns}
    date_col = snap_cols.get("date") or snap_cols.get("snapshot_date")
    if date_col is None:
        raise KeyError("snap_long에는 'date' 또는 'snapshot_date' 컬럼이 필요합니다.")

    latest_snap = pd.to_datetime(snap_long[date_col]).max().normalize()
    cons_start = max(latest_snap + pd.Timedelta(days=1), start_dt)
    if cons_start > end_dt:
        return out

    # 이벤트 계수 (없으면 1.0)
    idx = pd.date_range(cons_start, end_dt, freq="D")
    uplift = pd.Series(1.0, index=idx)
    if events:
        for e in events:
            s = pd.to_datetime(e.get("start"), errors="coerce")
            t = pd.to_datetime(e.get("end"), errors="coerce")
            u = min(3.0, max(-1.0, float(e.get("uplift", 0.0))))  # -100% ~ +300% 방어
            if pd.notna(s) and pd.notna(t):
                s = s.normalize(); t = t.normalize()
                s = max(s, idx[0]); t = min(t, idx[-1])
                if s <= t:
                    uplift.loc[s:t] = uplift.loc[s:t] * (1.0 + u)

    # 일일 소진량 추정 (보강 버전: 회귀 vs 감소분 평균의 max)
    rates = estimate_daily_consumption(snap_long, centers_sel, skus_sel, latest_snap, int(lookback_days))

    chunks: list[pd.DataFrame] = []  # ← 반드시 루프 밖에서 초기화
    for (ct, sku), g in out.groupby(["center","resource_code"]):
        # 가상 라인은 소진 미적용
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
        # 누적 차감(하한 0)
        for i in range(len(stk)):
            dec = daily[i]
            stk[i:] = np.maximum(0.0, stk[i:] - dec)
        g.loc[mask, "stock_qty"] = stk
        chunks.append(g)

    # 그룹이 없거나 전부 제외되었다면 원본 반환
    if not chunks:
        return out

    out = pd.concat(chunks, ignore_index=True)
    # 소수 정리(표시/일관성)
    out["stock_qty"] = (
        pd.to_numeric(out["stock_qty"], errors="coerce")
        .round()
        .clip(lower=0)
        .astype(int)
    )
    return out




@st.cache_data(ttl=1800)  # 30분 캐시
def build_timeline_new(snap_long: pd.DataFrame, moves: pd.DataFrame, 
                       centers_sel: List[str], skus_sel: List[str],
                       start_dt: pd.Timestamp, end_dt: pd.Timestamp, 
                       horizon_days: int = 0) -> pd.DataFrame:
    """
    반환: date, center, resource_code, stock_qty
    - 실제 센터 라인: 스냅샷 + (출발:onboard -, 도착:event +)
    - In-Transit 가상 라인: onboard +, event -
    """
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

        eff_plus = (
            mv[(mv["to_center"].astype(str) == str(ct)) &
               (mv["event_date"].notna()) &
               (mv["event_date"] > last_dt)]
            .groupby("event_date", as_index=False)["qty_ea"].sum()
            .rename(columns={"event_date":"date","qty_ea":"delta"})
        )

        eff_all = pd.concat([eff_minus, eff_plus], ignore_index=True).sort_values("date")
        for d, delta in zip(eff_all["date"], eff_all["delta"]):
            ts.loc[ts["date"] >= d, "stock_qty"] = ts.loc[ts["date"] >= d, "stock_qty"] + delta

        ts["stock_qty"] = ts["stock_qty"].clip(lower=0)
        lines.append(ts)

    # 2) In-Transit 가상 라인 (Non-WIP / WIP 분리)
    # 미리 타입 변환하여 반복 연산 최적화
    moves_str = moves.copy()
    
    # 디버깅: moves 원본 데이터의 event_date 상태 확인
    if "BA00021" in skus_sel:
        ba00021_moves_orig = moves[moves["resource_code"] == "BA00021"]
        st.write(f"🔍 build_timeline - moves 원본 BA00021: {len(ba00021_moves_orig)}건")
        if not ba00021_moves_orig.empty:
            st.write("moves 원본 샘플:", ba00021_moves_orig[["resource_code", "carrier_mode", "event_date"]].head())
            st.write("moves 원본 event_date 상태:", ba00021_moves_orig["event_date"].isna().sum(), "개가 NaT")
            st.write("moves 원본 event_date 값들:", ba00021_moves_orig["event_date"].head().tolist())
    
    moves_str["from_center"] = moves_str["from_center"].astype(str)
    moves_str["to_center"] = moves_str["to_center"].astype(str)
    moves_str["carrier_mode"] = moves_str["carrier_mode"].astype(str).str.upper()
    
    # 디버깅: moves 데이터 처리 전 상태 확인
    if "BA00021" in skus_sel:
        ba00021_moves = moves_str[moves_str["resource_code"] == "BA00021"]
        st.write(f"🔍 build_timeline - moves 처리 전 BA00021: {len(ba00021_moves)}건")
        if not ba00021_moves.empty:
            st.write("moves 처리 전 샘플:", ba00021_moves[["resource_code", "carrier_mode", "event_date"]].head())
            st.write("event_date 상태:", ba00021_moves["event_date"].isna().sum(), "개가 NaT")
    
    # 디버깅: moves_str 상태 확인
    if "BA00021" in skus_sel:
        ba00021_moves = moves_str[moves_str["resource_code"] == "BA00021"]
        st.write(f"🔍 build_timeline - BA00021 moves_str: {len(ba00021_moves)}건")
        if not ba00021_moves.empty:
            st.write("moves_str 샘플:", ba00021_moves[["resource_code", "carrier_mode", "event_date"]].head())
            st.write("event_date 상태:", ba00021_moves["event_date"].isna().sum(), "개가 NaT")
        
        # WIP 데이터 확인
        wip_moves = moves_str[moves_str["carrier_mode"] == "WIP"]
        st.write(f"🔍 build_timeline - WIP moves: {len(wip_moves)}건")
        if not wip_moves.empty:
            st.write("WIP moves 샘플:", wip_moves[["resource_code", "carrier_mode", "event_date"]].head())
    
    mv_sel = moves_str[
        moves_str["resource_code"].isin(skus_sel) &
        (moves_str["from_center"].isin(centers_sel) | 
         moves_str["to_center"].isin(centers_sel) | 
         (moves_str["carrier_mode"] == "WIP"))
    ]
    
    # 디버깅: mv_sel 필터링 결과 확인
    if "BA00021" in skus_sel:
        ba00021_mv_sel = mv_sel[mv_sel["resource_code"] == "BA00021"]
        st.write(f"🔍 build_timeline - mv_sel BA00021: {len(ba00021_mv_sel)}건")
        if not ba00021_mv_sel.empty:
            st.write("mv_sel 샘플:", ba00021_mv_sel[["resource_code", "carrier_mode", "event_date"]].head())
            st.write("event_date 상태:", ba00021_mv_sel["event_date"].isna().sum(), "개가 NaT")

    for sku, g in mv_sel.groupby("resource_code"):
        # Non-WIP In-Transit (선택된 센터로 이동중인 재고만)
        g_nonwip = g[g["carrier_mode"] != "WIP"]
        if not g_nonwip.empty:
            # 선택된 센터로 이동중인 재고만 필터링
            g_selected = g_nonwip[g_nonwip["to_center"].isin(centers_sel)]
            if not g_selected.empty:
                add_onboard = (
                    g_selected[g_selected["onboard_date"].notna()]
                    .groupby("onboard_date", as_index=False)["qty_ea"].sum()
                    .rename(columns={"onboard_date":"date","qty_ea":"delta"})
                )
                add_event = (
                    g_selected[g_selected["event_date"].notna()]
                    .groupby("event_date", as_index=False)["qty_ea"].sum()
                    .rename(columns={"event_date":"date","qty_ea":"delta"})
                )
                add_event["delta"] *= -1
                deltas = pd.concat([add_onboard, add_event], ignore_index=True)
                if not deltas.empty:
                    s = pd.Series(0, index=pd.to_datetime(full_dates))
                    for d, v in deltas.groupby("date")["delta"].sum().items():
                        d = pd.to_datetime(d)
                        if d < full_dates[0]:
                            s.iloc[:] = s.iloc[:] + v
                        else:
                            s.loc[s.index >= d] = s.loc[s.index >= d] + v
                    vdf = pd.DataFrame({
                        "date": s.index,
                        "center": "In-Transit",
                        "resource_code": sku,
                        "stock_qty": s.values.clip(min=0)
                    })
                    lines.append(vdf)

        # --- WIP 별도 라인 (생산중) ---
        g_wip = g[g["carrier_mode"].astype(str).str.upper() == "WIP"]
        
        # 디버깅: BA00021 WIP 원본 데이터 확인
        if sku == "BA00021":
            st.write(f"🔍 BA00021 WIP 원본 데이터: {len(g_wip)}건")
            if not g_wip.empty:
                st.write("WIP 원본 샘플:", g_wip[["resource_code", "onboard_date", "event_date", "qty_ea"]].head())
                st.write("event_date 상태:", g_wip["event_date"].isna().sum(), "개가 NaT")
                st.write("event_date 값들:", g_wip["event_date"].tolist())
        
        if not g_wip.empty:
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
            # ✅ 완료일은 음수(감소)
            add_event["delta"] *= -1
            
            # 디버깅: BA00021 WIP 데이터 확인
            if sku == "BA00021":
                st.write(f"🔍 BA00021 WIP 시작 데이터: {len(add_onboard)}건")
                st.write("WIP 시작 샘플:", add_onboard.head())
                st.write(f"🔍 BA00021 WIP 완료 데이터: {len(add_event)}건")
                st.write("WIP 완료 샘플:", add_event.head())

            # 날짜를 일 단위로 정규화
            add_onboard["date"] = pd.to_datetime(add_onboard["date"]).dt.normalize()
            add_event["date"]   = pd.to_datetime(add_event["date"]).dt.normalize()
            deltas = pd.concat([add_onboard, add_event], ignore_index=True)
            if not deltas.empty:
                # 날짜 범위 전체 0으로 시작해 계단 누적
                s = pd.Series(0, index=pd.to_datetime(full_dates))
                for d, v in deltas.groupby("date")["delta"].sum().items():
                    d = pd.to_datetime(d)
                    if d < full_dates[0]:
                        s.iloc[:] = s.iloc[:] + v
                    else:
                        s.loc[s.index >= d] = s.loc[s.index >= d] + v

                vdf = pd.DataFrame({
                    "date": s.index,
                    "center": "WIP",
                    "resource_code": sku,
                    "stock_qty": s.values.clip(min=0)
                })
                lines.append(vdf)
                
                # --- WIP 완료 물량을 센터 라인에 누적(+1) 반영 ---
                wip_done = g_wip[g_wip["event_date"].notna()].copy()
                wip_done["event_date"] = pd.to_datetime(wip_done["event_date"]).dt.normalize()
                if not wip_done.empty:
                    # 디버깅: WIP 완료 데이터 확인
                    if sku == "BA00021":
                        st.write(f"🔍 BA00021 WIP 완료 데이터: {len(wip_done)}건")
                        st.write("WIP 완료 샘플:", wip_done[["resource_code", "to_center", "event_date", "qty_ea"]].head())
                    
                    add_to_center = (
                        wip_done.groupby(["to_center","event_date"], as_index=False)["qty_ea"].sum()
                        .rename(columns={"to_center":"center","event_date":"date","qty_ea":"delta"})
                    )
                    
                    # 디버깅: 센터 입고 데이터 확인
                    if sku == "BA00021":
                        st.write(f"🔍 BA00021 센터 입고 데이터: {len(add_to_center)}건")
                        st.write("센터 입고 샘플:", add_to_center.head())
                    # lines 안의 각 '센터 라인(ts_line)'에 날짜>=d 지점부터 누적 가산
                    for idx_line, ts_line in enumerate(lines):
                        if ts_line.empty: 
                            continue
                        if ts_line["resource_code"].iloc[0] != sku:
                            continue
                        ct_name = ts_line["center"].iloc[0]
                        if ct_name in ("In-Transit", "WIP"):    # 가상 라인 제외
                            continue

                        inc = add_to_center[add_to_center["center"].astype(str) == str(ct_name)]
                        if inc.empty:
                            continue

                        ts = ts_line.sort_values("date").copy()
                        for d, q in zip(pd.to_datetime(inc["date"]), inc["delta"]):
                            ts.loc[ts["date"] >= d, "stock_qty"] = ts.loc[ts["date"] >= d, "stock_qty"] + int(q)
                        ts["stock_qty"] = ts["stock_qty"].clip(lower=0)
                        lines[idx_line] = ts

    if not lines:
        return pd.DataFrame(columns=["date","center","resource_code","stock_qty"])

    out = pd.concat(lines, ignore_index=True)
    out = out[(out["date"] >= start_dt) & (out["date"] <= horizon_end)]
    return out

# -------------------- Tabs for inputs --------------------
tab1, tab2, tab3 = st.tabs(["엑셀 업로드", "CSV 수동 업로드", "Google Sheets"])

with tab1:
    xfile = st.file_uploader("엑셀 업로드 (.xlsx)", type=["xlsx"], key="excel")
    if xfile is not None:
        df_move, df_ref, df_incoming = load_from_excel(xfile)
        moves_raw = normalize_moves(df_move)
        snap_long = normalize_refined_snapshot(df_ref)


        # WIP 불러오기 및 병합
        try:
            wip_df = load_wip_from_incoming(df_incoming)
            st.write(f"🔍 WIP 데이터 로드됨: {len(wip_df)}건")
            if not wip_df.empty:
                st.write("WIP 샘플:", wip_df.head())
                # BA00021, BA00022 WIP 파싱 결과 확인
                ba00021_wip = wip_df[wip_df["resource_code"] == "BA00021"]
                ba00022_wip = wip_df[wip_df["resource_code"] == "BA00022"]
                if not ba00021_wip.empty:
                    st.caption("🔍 BA00021 WIP 파싱 확인")
                    st.dataframe(ba00021_wip)
                    st.write("WIP 샘플(BA00021):", 
                             wip_df.loc[wip_df["resource_code"]=="BA00021",
                                        ["resource_code","wip_start","wip_ready","qty_ea"]].head(10))
                if not ba00022_wip.empty:
                    st.caption("🔍 BA00022 WIP 파싱 확인")
                    st.dataframe(ba00022_wip)
            moves = merge_wip_as_moves(moves_raw, wip_df)
            st.write(f"🔍 최종 moves 데이터: {len(moves)}건")
            if not moves.empty:
                wip_moves = moves[moves["carrier_mode"] == "WIP"]
                st.write(f"🔍 WIP moves: {len(wip_moves)}건")
                if not wip_moves.empty:
                    st.write("WIP moves 샘플:", wip_moves[["resource_code", "onboard_date", "event_date", "qty_ea"]].head())
                    # BA00021 WIP moves 확인
                    ba00021_moves = wip_moves[wip_moves["resource_code"] == "BA00021"]
                    if not ba00021_moves.empty:
                        st.write("🔍 BA00021 WIP moves:", ba00021_moves[["resource_code", "onboard_date", "event_date", "qty_ea"]].head())
                        st.write("🔍 BA00021 event_date 상태:", ba00021_moves["event_date"].isna().sum(), "개가 NaT")
                        st.write("🔍 BA00021 event_date 값들:", ba00021_moves["event_date"].tolist())
                
                # 디버깅: moves 데이터 전체 상태 확인
                st.write(f"🔍 moves 전체 carrier_mode 분포:")
                carrier_counts = moves["carrier_mode"].value_counts()
                st.write(carrier_counts)
                
                # BA00021 moves 전체 확인
                ba00021_all = moves[moves["resource_code"] == "BA00021"]
                st.write(f"🔍 BA00021 moves 전체: {len(ba00021_all)}건")
                if not ba00021_all.empty:
                    st.write("BA00021 moves 전체 샘플:", ba00021_all[["resource_code", "carrier_mode", "event_date"]].head())
            st.success(f"WIP {len(wip_df)}건 반영 완료" if wip_df is not None and not wip_df.empty else "WIP 없음")
        except Exception as e:
            moves = moves_raw
            st.warning(f"WIP 불러오기 실패: {e}")

with tab2:
    cs_snap = st.file_uploader("정제 스냅샷 CSV 업로드 (snap_정제: date,center,resource_code,stock_qty)", type=["csv"], key="snapcsv")
    cs_move = st.file_uploader("SCM_통합.csv 업로드", type=["csv"], key="movecsv")
    if cs_snap is not None and cs_move is not None:
        df_ref = pd.read_csv(cs_snap)
        move_raw = pd.read_csv(cs_move)

        # 이동 데이터 정규화
        moves = normalize_moves(move_raw)

        # 정제 스냅샷 정규화 (date/center/resource_code/stock_qty로 통일)
        snap_cols = {c.strip().lower(): c for c in df_ref.columns}

        # 유연 매핑
        col_date = snap_cols.get("date") or snap_cols.get("snapshot_date") or snap_cols.get("스냅샷 일자")
        col_center = snap_cols.get("center") or snap_cols.get("창고명")
        col_sku = snap_cols.get("resource_code") or snap_cols.get("sku") or snap_cols.get("상품코드")
        col_qty = (snap_cols.get("stock_qty") or snap_cols.get("qty") or
                   snap_cols.get("quantity") or snap_cols.get("수량"))

        if not all([col_date, col_center, col_sku, col_qty]):
            st.error("정제 스냅샷 CSV에 'date, center, resource_code, stock_qty' 컬럼(또는 동의어)이 필요합니다.")
            st.stop()

        sr = df_ref.rename(columns={
            col_date: "date",
            col_center: "center",
            col_sku: "resource_code",
            col_qty: "stock_qty",
        }).copy()

        sr["date"] = pd.to_datetime(sr["date"], errors="coerce").dt.normalize()
        sr["center"] = sr["center"].astype(str)
        sr["resource_code"] = sr["resource_code"].astype(str)
        sr["stock_qty"] = pd.to_numeric(sr["stock_qty"], errors="coerce").fillna(0).astype(int)

        snap_long = sr[["date","center","resource_code","stock_qty"]].dropna()

with tab3:
    st.subheader("Google Sheets 연동 (Apps Script Web App)")
    st.info("Google Apps Script Web App을 통해 Google Sheets에서 실시간으로 데이터를 가져옵니다.")
    
    # 설정 상태 표시
    cfg = _get_proxy_cfg()
    if cfg["webapp_url"] and cfg["sheet_id"] and cfg["token"]:
        st.success("✅ WebApp 프록시 설정 완료")
        st.caption(f"시트 ID: {cfg['sheet_id']}")
    else:
        st.error("❌ WebApp 프록시 설정이 필요합니다.")
        st.caption("Streamlit Secrets에 gs_proxy 설정을 추가하거나 환경변수를 설정해주세요.")
    
    if st.button("Google Sheets에서 데이터 로드", type="primary"):
        try:
            with st.spinner("Google Sheets에서 데이터를 가져오는 중..."):
                df_move, df_ref, df_incoming = load_from_gsheet()
                moves_raw = normalize_moves(df_move)
                snap_long = normalize_refined_snapshot(df_ref)
                
                # WIP 불러오기 및 병합
                try:
                    wip_df = load_wip_from_incoming(df_incoming)
                    moves = merge_wip_as_moves(moves_raw, wip_df)
                    st.success(f"Google Sheets 로드 완료! WIP {len(wip_df)}건 반영" if wip_df is not None and not wip_df.empty else "Google Sheets 로드 완료! WIP 없음")
                except Exception as e:
                    moves = moves_raw
                    st.warning(f"WIP 불러오기 실패: {e}")
                    
        except Exception as e:
            st.error(f"Google Sheets 로드 실패: {e}")
            st.caption("WebApp URL, 시트 ID, 토큰을 확인해주세요.")

# 앱 시작 시 Google Sheets에서 자동 로드 시도
if "snap_long" not in locals():
    try:
        with st.spinner("Google Sheets에서 데이터를 자동 로드하는 중..."):
            df_move, df_ref, df_incoming = load_from_gsheet()
            moves_raw = normalize_moves(df_move)
            snap_long = normalize_refined_snapshot(df_ref)
            
            # WIP 불러오기 및 병합
            try:
                wip_df = load_wip_from_incoming(df_incoming)
                st.write(f"🔍 WIP 데이터 로드됨: {len(wip_df)}건")
                if not wip_df.empty:
                    st.write("WIP 샘플:", wip_df.head())
                    # BA00021, BA00022 WIP 파싱 결과 확인
                    ba00021_wip = wip_df[wip_df["resource_code"] == "BA00021"]
                    ba00022_wip = wip_df[wip_df["resource_code"] == "BA00022"]
                    if not ba00021_wip.empty:
                        st.caption("🔍 BA00021 WIP 파싱 확인")
                        st.dataframe(ba00021_wip)
                        st.write("WIP 샘플(BA00021):", 
                                 wip_df.loc[wip_df["resource_code"]=="BA00021",
                                            ["resource_code","wip_start","wip_ready","qty_ea"]].head(10))
                    if not ba00022_wip.empty:
                        st.caption("🔍 BA00022 WIP 파싱 확인")
                        st.dataframe(ba00022_wip)
                moves = merge_wip_as_moves(moves_raw, wip_df)
                st.write(f"🔍 최종 moves 데이터: {len(moves)}건")
                if not moves.empty:
                    wip_moves = moves[moves["carrier_mode"] == "WIP"]
                    st.write(f"🔍 WIP moves: {len(wip_moves)}건")
                    if not wip_moves.empty:
                        st.write("WIP moves 샘플:", wip_moves[["resource_code", "onboard_date", "event_date", "qty_ea"]].head())
                        # BA00021 WIP moves 확인
                        ba00021_moves = wip_moves[wip_moves["resource_code"] == "BA00021"]
                        if not ba00021_moves.empty:
                            st.write("🔍 BA00021 WIP moves:", ba00021_moves[["resource_code", "onboard_date", "event_date", "qty_ea"]].head())
                            st.write("🔍 BA00021 event_date 상태:", ba00021_moves["event_date"].isna().sum(), "개가 NaT")
                            st.write("🔍 BA00021 event_date 값들:", ba00021_moves["event_date"].tolist())
                st.success(f"✅ Google Sheets 자동 로드 완료! WIP {len(wip_df)}건 반영" if wip_df is not None and not wip_df.empty else "✅ Google Sheets 자동 로드 완료! WIP 없음")
            except Exception as e:
                moves = moves_raw
                st.warning(f"⚠️ WIP 불러오기 실패: {e}")
                
    except Exception as e:
        st.error(f"❌ Google Sheets 자동 로드 실패: {e}")
        st.info("엑셀, CSV 또는 Google Sheets에서 데이터를 로드하면 필터/차트가 나타납니다.")
        st.caption("WebApp 프록시 설정이 필요합니다. Streamlit Secrets에 gs_proxy 설정을 추가해주세요.")
        st.stop()



# -------------------- Filters --------------------
# 센터 목록: 스냅샷 + 이동 데이터에서 가져오기 (중복 제거)
centers_snap = set(snap_long["center"].dropna().astype(str).unique().tolist())
centers_moves = set(moves["from_center"].dropna().astype(str).unique().tolist() + 
                   moves["to_center"].dropna().astype(str).unique().tolist())

# 센터명 통일 (AcrossBUS = 어크로스비US) + 유효하지 않은 센터 제거
centers_moves_unified = set()
for center in centers_moves:
    if center == "AcrossBUS":
        centers_moves_unified.add("어크로스비US")
    elif center not in ["WIP", "In-Transit", "", "nan", "None"]:  # 유효하지 않은 센터 제거
        centers_moves_unified.add(center)

# 스냅샷 센터도 유효하지 않은 센터 제거
centers_snap_clean = set()
for center in centers_snap:
    if center not in ["WIP", "In-Transit", "", "nan", "None"]:  # 유효하지 않은 센터 제거
        centers_snap_clean.add(center)

centers = sorted(list(centers_snap_clean | centers_moves_unified))

skus = sorted(snap_long["resource_code"].dropna().astype(str).unique().tolist())
min_date = snap_long["date"].min()
max_date = snap_long["date"].max()

st.sidebar.header("필터")
centers_sel = st.sidebar.multiselect("센터 선택", centers, default=(["태광KR"] if "태광KR" in centers else centers[:1]))
skus_sel = st.sidebar.multiselect("SKU 선택", skus, default=([s for s in ["BA00022","BA00021"] if s in skus] or skus[:2]))


# === 기간 제어: 과거 42일, 미래 60일 ===
today = pd.Timestamp.today().normalize()
# 타임존 문제 해결: tz-naive로 통일
today = today.tz_localize(None) if today.tz else today
PAST_DAYS = 42  # 과거 42일
FUTURE_DAYS = 60  # 미래 60일

# 스냅샷 범위와 교차(스냅샷이 더 짧아도 안전)
snap_min = pd.to_datetime(snap_long["date"]).min().normalize()
snap_max = pd.to_datetime(snap_long["date"]).max().normalize()

# 타임존 문제 해결: 모든 날짜를 tz-naive로 통일
today = today.tz_localize(None) if today.tz else today
snap_min = snap_min.tz_localize(None) if snap_min.tz else snap_min
snap_max = snap_max.tz_localize(None) if snap_max.tz else snap_max

# snap_long의 date 컬럼도 tz-naive로 통일
snap_long["date"] = pd.to_datetime(snap_long["date"]).dt.tz_localize(None)

bound_min = max(today - pd.Timedelta(days=PAST_DAYS), snap_min)
bound_max = min(today + pd.Timedelta(days=FUTURE_DAYS), snap_max + pd.Timedelta(days=60))  # +60은 약간의 전망 여지

def _init_range():
    if "date_range" not in st.session_state:
        st.session_state.date_range = (max(today - pd.Timedelta(days=20), bound_min),
                                       min(today + pd.Timedelta(days=20), bound_max))
    if "horizon_days" not in st.session_state:
        st.session_state.horizon_days = 20

def _apply_horizon_to_range():
    h = int(st.session_state.horizon_days)
    h = max(0, min(h, FUTURE_DAYS))   # ← 입력 최대를 60일로 클램프
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

st.sidebar.subheader("기간 설정")
st.sidebar.number_input(
    "미래 전망 일수",
    min_value=0, max_value=FUTURE_DAYS, step=1,
    key="horizon_days", on_change=_apply_horizon_to_range
)

# 슬라이더: 범위를 ±6주 경계로 제한
date_range = st.sidebar.slider(
    "기간",
    min_value=bound_min.to_pydatetime(),
    max_value=bound_max.to_pydatetime(),
    value=tuple(d.to_pydatetime() for d in _clamp_range(st.session_state.date_range)),
    format="YYYY-MM-DD"
)

# 내부 사용 기간(슬라이더 우선)
start_dt = pd.Timestamp(date_range[0]).normalize()
end_dt   = pd.Timestamp(date_range[1]).normalize()
# 타임존 문제 해결: 모든 날짜를 tz-naive로 통일
start_dt = start_dt.tz_localize(None) if start_dt.tz else start_dt
end_dt = end_dt.tz_localize(None) if end_dt.tz else end_dt

# 전망일(빌드용): 최신 스냅샷 이후만 계산
_latest_snap = snap_long["date"].max()
# 타임존 문제 해결: 모든 날짜를 tz-naive로 통일
_latest_snap = _latest_snap.tz_localize(None) if _latest_snap.tz else _latest_snap
end_dt = end_dt.tz_localize(None) if end_dt.tz else end_dt
proj_days_for_build = max(0, int((end_dt - _latest_snap).days))



st.sidebar.header("표시 옵션")

# 표시 토글 (순서/문구 정리)
show_prod = st.sidebar.checkbox("생산중(미완료) 표시", value=True)
show_transit = st.sidebar.checkbox("이동중 표시", value=True)

# 추세 기반 재고 예측
use_cons_forecast = st.sidebar.checkbox("추세 기반 재고 예측", value=True)
lookback_days = st.sidebar.number_input("추세 계산 기간(일)", min_value=7, max_value=56, value=28, step=7)

# 프로모션 가중치 (단일)
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

# 스냅샷 기준 최신일
latest_dt = snap_long["date"].max()
latest_dt_str = pd.to_datetime(latest_dt).strftime("%Y-%m-%d")

# SKU별 현재 재고(선택 센터 기준, 최신 스냅샷)
latest_rows = snap_long[(snap_long["date"] == latest_dt) & (snap_long["center"].isin(centers_sel))]
if latest_rows.empty:
    st.warning("선택된 센터에 해당하는 최신 스냅샷 데이터가 없습니다.")
    sku_totals = {sku: 0 for sku in skus_sel}
else:
    sku_totals = {sku: int(latest_rows[latest_rows["resource_code"]==sku]["stock_qty"].sum()) for sku in skus_sel}

# In-Transit (WIP 제외, 입/출고 모두 포함)
today = pd.Timestamp.today().normalize()
# 타임존 문제 해결: tz-naive로 통일
today = today.tz_localize(None) if today.tz else today
# normalize_moves에서 이미 astype(str) 처리됨, carrier_mode만 upper() 적용
moves_typed = moves.copy()
moves_typed["carrier_mode"] = moves_typed["carrier_mode"].str.upper()

# 타임존 문제 해결: 모든 날짜 컬럼을 tz-naive로 통일
date_columns = ["onboard_date", "arrival_date", "inbound_date", "real_departure", "event_date"]
for col in date_columns:
    if col in moves_typed.columns:
        moves_typed[col] = pd.to_datetime(moves_typed[col], errors="coerce")
        moves_typed[col] = moves_typed[col].dt.tz_localize(None) if moves_typed[col].dt.tz is not None else moves_typed[col]

# moves 원본 데이터도 타임존 통일
for col in date_columns:
    if col in moves.columns:
        moves[col] = pd.to_datetime(moves[col], errors="coerce")
        moves[col] = moves[col].dt.tz_localize(None) if moves[col].dt.tz is not None else moves[col]

in_transit_mask = (
    (moves_typed["onboard_date"].notna()) &
    (moves_typed["onboard_date"] <= today) &
    (moves_typed["inbound_date"].isna()) &
    ((moves_typed["arrival_date"].isna()) | (moves_typed["arrival_date"] > today)) &
    (moves_typed["to_center"].isin(centers_sel)) &  # 선택된 센터로 이동중인 재고만
    (moves_typed["resource_code"].isin(skus_sel)) &
    (moves_typed["carrier_mode"] != "WIP")
)
in_transit_total = int(moves_typed[in_transit_mask]["qty_ea"].sum())

# WIP(오늘 기준 잔량, 선택 센터/SKU 범위)
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

# SKU별 현재 재고 카드
def _chunk(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

for group in _chunk(skus_sel, 4):
    cols = st.columns(len(group))
    for i, sku in enumerate(group):
        cols[i].metric(f"{sku} 현재 재고(스냅샷 {latest_dt_str})", f"{sku_totals.get(sku, 0):,}")

# 통합 KPI
k_it, k_wip = st.columns(2)
k_it.metric("이동 중 재고", f"{in_transit_total:,}")
k_wip.metric("생산중(미완료)", f"{wip_today:,}")

st.divider()

# -------------------- Step Chart (Plotly) --------------------
st.subheader("계단식 재고 흐름")

# 디버깅: build_timeline 호출 전 moves 데이터 상태 확인
if "BA00021" in skus_sel:
    ba00021_moves = moves[moves["resource_code"] == "BA00021"]
    st.write(f"🔍 build_timeline 호출 전 - BA00021 moves: {len(ba00021_moves)}건")
    if not ba00021_moves.empty:
        st.write("moves 샘플:", ba00021_moves[["resource_code", "carrier_mode", "event_date"]].head())
        st.write("event_date 상태:", ba00021_moves["event_date"].isna().sum(), "개가 NaT")

timeline = build_timeline(snap_long, moves, centers_sel, skus_sel,
                          start_dt, end_dt, horizon_days=proj_days_for_build)

# (A) 소진 추세 + 이벤트 가중치 적용
if use_cons_forecast and not timeline.empty:
    timeline = apply_consumption_with_events(
        timeline, snap_long, centers_sel, skus_sel,
        start_dt, end_dt,
        lookback_days=int(lookback_days),
        events=events
    )

if timeline.empty:
    st.info("선택 조건에 해당하는 타임라인 데이터가 없습니다.")
else:
    vis_df = timeline[(timeline["date"] >= start_dt) & (timeline["date"] <= end_dt)].copy()

    # (B) 센터 표시 명칭 한글화 + 토글
    # In-Transit → 이동중 (통합)
    vis_df["center"] = vis_df["center"].str.replace(r"^In-Transit.*$", "이동중", regex=True)
    # WIP → 생산중
    vis_df.loc[vis_df["center"] == "WIP", "center"] = "생산중"

    # ✅ 생산중(WIP)은 오늘 이후만 표시 (과거 구간 숨김)
    today = pd.Timestamp.today().normalize()
    vis_df = vis_df[~((vis_df["center"] == "생산중") & (vis_df["date"] < today))]

    # ✅ 태광KR 미선택 시 '생산중' 라인 숨김
    if "태광KR" not in centers_sel:
        vis_df = vis_df[vis_df["center"] != "생산중"]
    
    if not show_prod:
        vis_df = vis_df[vis_df["center"] != "생산중"]
    if not show_transit:
        vis_df = vis_df[~vis_df["center"].str.startswith("이동중")]
    
    # 0 값 데이터 필터링 (차트 표시 문제 해결)
    vis_df = vis_df[vis_df["stock_qty"] > 0]

    # 라벨
    vis_df["label"] = vis_df["resource_code"] + " @ " + vis_df["center"]

    # (C) 그리기
    fig = px.line(
        vis_df, x="date", y="stock_qty", color="label",
        line_shape="hv",
        title="선택한 SKU × 센터(및 이동중/생산중) 계단식 재고 흐름",
        render_mode="svg"
    )
    fig.update_layout(
        hovermode="x unified",
        xaxis_title="날짜",
        yaxis_title="재고량(EA)",
        legend_title_text="SKU @ Center / 이동중(점선) / 생산중(점선)",
        margin=dict(l=20, r=20, t=60, b=20)
    )

    # === 오늘 기준선 표시 ===
    today = pd.Timestamp.today().normalize()
    # 타임존 문제 해결: tz-naive로 통일
    today = today.tz_localize(None) if today.tz else today
    if start_dt <= today <= end_dt:
        fig.add_vline(
            x=today,
            line_width=1,
            line_dash="solid",  # 실선으로 변경
            line_color="rgba(255, 0, 0, 0.4)",  # 더 희미한 반투명 빨간색
        )
        fig.add_annotation(
            x=today,
            y=1.02,
            xref="x",
            yref="paper",
            text="오늘",
            showarrow=False,
            font=dict(size=12, color="#555"),
            align="center",
            yanchor="bottom",
        )


    
    # Y축 눈금 정수로
    fig.update_yaxes(tickformat=",.0f")
    # 호버도 정수 천단위
    fig.update_traces(hovertemplate="날짜: %{x|%Y-%m-%d}<br>재고: %{y:,.0f} EA<br>%{fullData.name}<extra></extra>")


    # (D) 색상/선 스타일 - 각 라인마다 다른 색상 사용
    PALETTE = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
        "#c49c94", "#f7b6d3", "#c7c7c7", "#dbdb8d", "#9edae5",
        "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
        "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac"
    ]
    
    # 각 라인마다 고유한 색상 할당
    line_colors = {}
    color_idx = 0
    for tr in fig.data:
        name = tr.name or ""
        if " @ " in name:
            if name not in line_colors:
                line_colors[name] = PALETTE[color_idx % len(PALETTE)]
                color_idx += 1

    for i, tr in enumerate(fig.data):
        name = tr.name or ""
        if " @ " not in name:
            continue
        sku, kind = name.split(" @ ", 1)
        line_color = line_colors.get(name, PALETTE[0])

        if kind == "이동중":
            # 점선 + 두께 1.2
            fig.data[i].update(line=dict(color=line_color, dash="dot", width=1.2), opacity=0.9)
            fig.data[i].legendgroup = f"{sku} (이동중)"
            fig.data[i].legendrank = 20
        elif kind == "생산중":
            # 파선 + 두께 1.0
            fig.data[i].update(line=dict(color=line_color, dash="dash", width=1.0), opacity=0.8)
            fig.data[i].legendgroup = f"{sku} (생산중)"
            fig.data[i].legendrank = 30
        else:
            # 실선 + 두께 1.5
            fig.data[i].update(line=dict(color=line_color, dash="solid", width=1.5), opacity=1.0)
            fig.data[i].legendgroup = f"{sku} (센터)"
            fig.data[i].legendrank = 10


    chart_key = (
        f"stepchart|centers={','.join(centers_sel)}|skus={','.join(skus_sel)}|"
        f"{start_dt:%Y%m%d}-{end_dt:%Y%m%d}"
        f"|h{int(st.session_state.horizon_days)}"
        f"|prod{int(show_prod)}|tran{int(show_transit)}"
    )

    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False}, key=chart_key)



# -------------------- Upcoming Arrivals --------------------
st.subheader("입고 예정 내역 (선택 센터/SKU)")

today = pd.Timestamp.today().normalize()
# 타임존 문제 해결: tz-naive로 통일
today = today.tz_localize(None) if today.tz else today
window_start = max(start_dt, today)   # ✅ 오늘 이후만
window_end   = end_dt
# 타임존 문제 해결: window_start와 window_end도 tz-naive로 통일
window_start = window_start.tz_localize(None) if window_start.tz else window_start
window_end = window_end.tz_localize(None) if window_end.tz else window_end

# (A) 운송(비 WIP) - 기존 타입 변환된 데이터 재사용
arr_transport = moves_typed[
    (moves_typed["event_date"].notna()) &
    (moves_typed["event_date"] >= window_start) & (moves_typed["event_date"] <= window_end) &
    (moves_typed["carrier_mode"] != "WIP") &
    (moves_typed["to_center"].isin([c for c in centers_sel if c != "태광KR"])) &
    (moves_typed["resource_code"].isin(skus_sel))
]

# (B) WIP - 태광KR 선택 시
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

if upcoming.empty:
    st.caption("도착 예정 없음 (오늘 이후 / 선택 기간)")
else:
    # ✅ days_to_arrival는 항상 0 이상
    upcoming["days_to_arrival"] = (upcoming["event_date"] - today).dt.days
    upcoming = upcoming.sort_values(["event_date","to_center","resource_code","qty_ea"], 
                                   ascending=[True,True,True,False])
    cols = ["event_date","days_to_arrival","to_center","resource_code","qty_ea","carrier_mode","onboard_date","lot"]
    cols = [c for c in cols if c in upcoming.columns]
    st.dataframe(upcoming[cols].head(1000), use_container_width=True, height=300)


# -------------------- Simulation --------------------
st.subheader("출고 가능 시뮬레이션")
sim_c1, sim_c2, sim_c3, sim_c4 = st.columns(4)
with sim_c1:
    sim_center = st.selectbox("센터", centers, index=max(0, centers.index("태광KR") if "태광KR" in centers else 0))
with sim_c2:
    default_skus = [s for s in ["BA00022","BA00021"] if s in skus] or skus
    sim_sku = st.selectbox("SKU", skus, index=max(0, skus.index(default_skus[0])))
with sim_c3:
    sim_days = st.number_input(f"며칠 뒤 (기준일: {today.strftime('%Y-%m-%d')})", min_value=0, max_value=60, value=20, step=1)
    st.caption(f"→ 목표일: {(today + pd.Timedelta(days=int(sim_days))).strftime('%Y-%m-%d')}")
with sim_c4:
    sim_qty = st.number_input("필요 수량", min_value=0, step=1000, value=20000)

# === 출고 가능 시뮬레이션 (소진/이벤트 적용 포함) ===
sim_target_dt = (today + pd.Timedelta(days=int(sim_days))).normalize()

# 1) 타임라인 생성 (시뮬 윈도우만)
sim_tl = build_timeline(
    snap_long, moves,
    centers_sel=[sim_center], skus_sel=[sim_sku],
    start_dt=today, end_dt=sim_target_dt,
    horizon_days=max(0, (sim_target_dt - snap_long["date"].max()).days)
)

# 2) 소진 + 이벤트 가중치 동일하게 적용 (본문과 완전 동일)
if use_cons_forecast and not sim_tl.empty:
    sim_tl = apply_consumption_with_events(
        sim_tl, snap_long,
        centers_sel=[sim_center], skus_sel=[sim_sku],
        start_dt=today, end_dt=sim_target_dt,
        lookback_days=int(lookback_days),
        events=events  # 사이드바 단일 이벤트 세팅 그대로 재사용
    )

if sim_tl.empty:
    st.info("해당 조합의 타임라인이 없습니다.")
else:
    # 3) 목표일 재고(실제 센터 라인만: 이동중/WIP 제외)
    real_mask = (
        (sim_tl["center"] == sim_center) &
        (~sim_tl["center"].isin(["WIP"])) &
        (~sim_tl["center"].str.startswith("In-Transit", na=False))
    )
    sim_stock = int(pd.to_numeric(
        sim_tl.loc[(sim_tl["date"] == sim_target_dt) & real_mask, "stock_qty"],
        errors="coerce"
    ).fillna(0).sum().round())

    # 4) 결과 표시 (본문과 일치)
    ok = sim_stock >= sim_qty
    st.metric(
        f"{int(sim_days)}일 뒤({sim_target_dt:%Y-%m-%d}) '{sim_center}'의 '{sim_sku}' 예상 재고",
        f"{sim_stock:,}",
        delta=f"필요 {sim_qty:,}"
    )
    if ok:
        st.success("출고 가능")
    else:
        st.error("출고 불가")

    # (선택) 디버그 배지: 현재 소진 추정치
    if use_cons_forecast:
        rates_dbg = estimate_daily_consumption(
            snap_long, [sim_center], [sim_sku],
            asof_dt=pd.to_datetime(snap_long["date"]).max().normalize(),
            lookback_days=int(lookback_days)
        )
        r0 = float(next(iter(rates_dbg.values()), 0.0))
        st.caption(f"소진 추정(최근 {int(lookback_days)}일): {sim_center} / {sim_sku} ≈ {int(round(r0))} EA/일"
                   + (" · 이벤트 가중치 적용" if events else ""))

# ==================== 선택 센터 현재 재고 (전체 SKU) ====================
st.subheader(f"선택 센터 현재 재고 (스냅샷 {latest_dt_str} / 전체 SKU)")

# 1) 최신 스냅샷에서 선택 센터만 추출
cur = snap_long[
    (snap_long["date"] == latest_dt) &
    (snap_long["center"].isin(centers_sel))
].copy()

# 2) SKU×센터 피벗 (빈 값은 0)
pivot = (
    cur.groupby(["resource_code", "center"], as_index=False)["stock_qty"].sum()
       .pivot(index="resource_code", columns="center", values="stock_qty")
       .fillna(0)
       .astype(int)
)

# 3) 총합 컬럼 추가
pivot["총합"] = pivot.sum(axis=1)

# 4) UX: 필터/정렬 옵션
# UI 개선: 체크박스를 상단으로 이동
col1, col2 = st.columns([2, 1])
with col1:
    q = st.text_input("SKU 필터(포함 검색)", "", key="sku_filter_text")
with col2:
    sort_by = st.selectbox("정렬 기준", ["총합"] + list(pivot.columns.drop("총합")), index=0)

# 체크박스들을 별도 행에 배치
col1, col2 = st.columns(2)
with col1:
    hide_zero = st.checkbox("총합=0 숨기기", value=True)
with col2:
    show_cost = st.checkbox("재고자산(제조원가) 표시", value=False)

# 5) 필터 적용
view = pivot.copy()
if q.strip():
    view = view[view.index.str.contains(q.strip(), case=False, regex=False)]
if hide_zero:
    view = view[view["총합"] > 0]

# 6) 정렬(내림차순)  
view = view.sort_values(by=sort_by, ascending=False)

# 7) 비용 통합 로직
if show_cost:
    snap_raw_df = load_snapshot_raw()
    cost_pivot = pivot_inventory_cost_from_raw(snap_raw_df, latest_dt, centers_sel)  # SKU × 센터비용
    # 총비용 컬럼
    if not cost_pivot.empty:
        cost_cols = [c for c in cost_pivot.columns if c.endswith("_재고자산")]
        cost_pivot["총 재고자산"] = cost_pivot[cost_cols].sum(axis=1).astype(int)
        # 수량 view(피벗)과 병합
        merged = view.reset_index().rename(columns={"resource_code":"SKU"}) \
                     .merge(cost_pivot.rename(columns={"resource_code":"SKU"}), on="SKU", how="left")
        # 결측 0
        cost_cols2 = [c for c in merged.columns if c.endswith("_재고자산")] + (["총 재고자산"] if "총 재고자산" in merged.columns else [])
        if cost_cols2:
            merged[cost_cols2] = merged[cost_cols2].fillna(0).astype(int)
            # 재고자산 컬럼 포맷팅 (천 단위 구분자 + 원)
            for col in cost_cols2:
                merged[col] = merged[col].apply(lambda x: f"{x:,}원" if pd.notna(x) else "0원")
        # 컬럼 순서: [SKU] + (센터 수량들) + [총합] + (센터재고자산들) + [총 재고자산]
        qty_center_cols = [c for c in merged.columns if c not in ["SKU","총합"] + cost_cols2]
        ordered = ["SKU"] + qty_center_cols + (["총합"] if "총합" in merged.columns else []) + cost_cols2
        merged = merged[ordered]
        show_df = merged
    else:
        show_df = view.reset_index().rename(columns={"resource_code":"SKU"})
else:
    show_df = view.reset_index().rename(columns={"resource_code":"SKU"})

# 8) 수량 컬럼 포맷팅 (천 단위 구분자)
qty_columns = [col for col in show_df.columns if col not in ["SKU"] and not col.endswith("_재고자산") and col != "총 재고자산"]
for col in qty_columns:
    if col in show_df.columns:
        show_df[col] = show_df[col].apply(lambda x: f"{x:,}" if pd.notna(x) and isinstance(x, (int, float)) else x)

# 9) 보여주기
st.dataframe(show_df, use_container_width=True, height=380)

# 10) CSV 다운로드
csv_bytes = show_df.to_csv(index=False).encode("utf-8-sig")
st.download_button(
    "현재 표 CSV 다운로드",
    data=csv_bytes,
    file_name=f"centers_{'-'.join(centers_sel)}_snapshot_{latest_dt_str}.csv",
    mime="text/csv"
)

st.caption("※ 이 표는 **선택된 센터 전체 SKU**의 최신 스냅샷 재고입니다. 상단 'SKU 선택'과 무관하게 모든 SKU가 포함됩니다.")

# === 로트 상세 자동 표시 (선택 SKU가 1개일 때) ===
# ※ show_df는 위에서 실제로 st.dataframe에 표시한 테이블
#    (cost 미표시: view.reset_index().rename(columns={"resource_code":"SKU"})
#     cost 표시:   merged → show_df) 로 구성됨.

# 1) 현재 화면에 보이는 표에서 SKU 목록 추출
if 'show_df' in locals() and "SKU" in show_df.columns:
    filtered_df = show_df
else:
    # 혹시라도 show_df가 없으면 view를 기준으로 생성
    filtered_df = view.reset_index().rename(columns={"resource_code": "SKU"})

visible_skus = (
    filtered_df["SKU"].dropna().astype(str).unique().tolist()
)

# 2) SKU가 1개일 때만 로트 상세 표시
if len(visible_skus) == 1:
    lot_sku = visible_skus[0]
    snap_raw_df = load_snapshot_raw()
    lot_tbl = build_lot_detail(snap_raw_df, latest_dt, lot_sku, centers_sel)

    st.markdown(f"### 로트 상세 (스냅샷 {latest_dt_str} / **{', '.join(centers_sel)}** / **{lot_sku}**)")
    if lot_tbl.empty:
        st.caption("해당 조건의 로트 상세가 없습니다.")
    else:
        st.dataframe(lot_tbl, use_container_width=True, height=320)
        st.download_button(
            "로트 상세 CSV 다운로드",
            data=lot_tbl.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"lot_detail_{lot_sku}_{latest_dt:%Y%m%d}.csv",
            mime="text/csv"
        )
else:
    st.caption("※ 특정 SKU 한 개만 필터링하면 로트 상세가 자동 표시됩니다.")