import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Tuple, Optional
import os, time, json, requests

# === Apps Script Web App 프록시 설정 ===
def _get_proxy_cfg():
    if "gs_proxy" in st.secrets:
        cfg = st.secrets["gs_proxy"]
        return dict(
            webapp_url = cfg.get("webapp_url", "").strip(),
            sheet_id   = cfg.get("sheet_id", "").strip(),
            token      = cfg.get("token", "").strip(),
            timeout_s  = int(cfg.get("timeout_s", 15)),
        )
    return dict(
        webapp_url = os.getenv("GS_WEBAPP_URL","").strip(),
        sheet_id   = os.getenv("GS_SHEET_ID","").strip(),
        token      = os.getenv("GS_TOKEN","").strip(),
        timeout_s  = int(os.getenv("GS_TIMEOUT","15")),
    )

def _fetch_sheet_via_webapp(tab_name: str) -> pd.DataFrame:
    cfg = _get_proxy_cfg()
    if not (cfg["webapp_url"] and cfg["sheet_id"] and cfg["token"]):
        st.error("WebApp 프록시 설정이 누락되었습니다. (webapp_url/sheet_id/token)")
        return pd.DataFrame()

    params = {
        "id": cfg["sheet_id"],
        "sheet": tab_name,
        "token": cfg["token"],
        "_": str(int(time.time()))
    }

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
    return _fetch_sheet_via_webapp(sheet_name)

# === 데이터 로딩 함수들 ===
def load_from_excel(uploaded_file) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Excel 파일에서 데이터 로드"""
    with st.spinner("Excel 파일을 읽는 중..."):
        df_move = pd.read_excel(uploaded_file, sheet_name="SCM_통합")
        df_ref = pd.read_excel(uploaded_file, sheet_name="snap_정제")
        df_incoming = pd.read_excel(uploaded_file, sheet_name="입고예정내역")
    return df_move, df_ref, df_incoming

def load_from_gsheet() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Google Sheets에서 데이터 로드"""
    with st.spinner("Google Sheets에서 데이터를 로드하는 중..."):
        df_move = gs_csv("SCM_통합")
        df_ref = gs_csv("snap_정제")
        df_incoming = gs_csv("입고예정내역")
    return df_move, df_ref, df_incoming

def normalize_moves(df_move: pd.DataFrame) -> pd.DataFrame:
    """이동 내역 데이터 정규화"""
    df = df_move.copy()
    df.columns = [str(c).strip() for c in df.columns]
    
    # 컬럼 매핑
    col_map = {
        "resource_code": ["resource_code", "상품코드", "product_code"],
        "qty_ea": ["qty_ea", "수량", "quantity"],
        "carrier_mode": ["carrier_mode", "운송수단", "carrier_name"],
        "from_center": ["from_center", "출발센터", "출발창고"],
        "to_center": ["to_center", "도착센터", "도착창고"],
        "onboard_date": ["onboard_date", "출발일", "departure_date"],
        "arrival_date": ["arrival_date", "도착일", "arrival_date"],
        "inbound_date": ["inbound_date", "입고일", "inbound_date"],
        "real_departure": ["real_departure", "실제출발일", "actual_departure"],
        "event_date": ["event_date", "이벤트일", "event_date"]
    }
    
    result = pd.DataFrame()
    for target_col, source_cols in col_map.items():
        for source_col in source_cols:
            if source_col in df.columns:
                result[target_col] = df[source_col]
                break
        if target_col not in result.columns:
            result[target_col] = pd.NaT if "date" in target_col else ""
    
    # 수량 정규화 (쉼표 제거)
    result["qty_ea"] = pd.to_numeric(result["qty_ea"].astype(str).str.replace(',', ''), errors='coerce').fillna(0).astype(int)
    
    # 날짜 정규화
    for date_col in ["onboard_date", "arrival_date", "inbound_date", "real_departure", "event_date"]:
        result[date_col] = pd.to_datetime(result[date_col], errors='coerce')
    
    return result

def normalize_refined_snapshot(df_ref: pd.DataFrame) -> pd.DataFrame:
    """정제된 스냅샷 데이터 정규화"""
    df = df_ref.copy()
    df.columns = [str(c).strip() for c in df.columns]
    
    # 컬럼 매핑
    col_map = {
        "date": ["date", "날짜", "snapshot_date"],
        "center": ["center", "센터", "창고"],
        "resource_code": ["resource_code", "상품코드", "product_code"],
        "stock_qty": ["stock_qty", "재고수량", "quantity"]
    }
    
    result = pd.DataFrame()
    for target_col, source_cols in col_map.items():
        for source_col in source_cols:
            if source_col in df.columns:
                result[target_col] = df[source_col]
                break
        if target_col not in result.columns:
            result[target_col] = pd.NaT if target_col == "date" else 0
    
    # 날짜 정규화
    result["date"] = pd.to_datetime(result["date"], errors='coerce')
    
    # 수량 정규화
    result["stock_qty"] = pd.to_numeric(result["stock_qty"], errors='coerce').fillna(0).astype(int)
    
    return result

def load_wip_from_incoming(df_incoming: pd.DataFrame) -> pd.DataFrame:
    """입고예정내역에서 WIP 데이터 로드"""
    df = df_incoming.copy()
    df.columns = [str(c).strip() for c in df.columns]
    
    # 컬럼 매핑
    sku_col = next((c for c in df.columns if "product_code" in c.lower() or "resource_code" in c.lower()), None)
    qty_col = next((c for c in df.columns if "total_quantity" in c.lower() or "quantity" in c.lower()), None)
    date_col = next((c for c in df.columns if "intended_push_date" in c.lower() or "push" in c.lower()), None)
    
    if not all([sku_col, qty_col, date_col]):
        return pd.DataFrame()
    
    result = pd.DataFrame({
        "resource_code": df[sku_col].astype(str).str.strip(),
        "to_center": "태광KR",  # 기본 도착 센터
        "wip_ready": pd.to_datetime(df[date_col], errors='coerce').dt.normalize(),
        "qty_ea": pd.to_numeric(df[qty_col].astype(str).str.replace(',', ''), errors='coerce').fillna(0).astype(int),
        "lot": df.get("lot", "")
    })
    
    # wip_start 계산 (wip_ready - 30일)
    result["wip_start"] = result["wip_ready"] - pd.to_timedelta(30, unit="D")
    
    # 유효값만 반환
    result = result.dropna(subset=["resource_code", "wip_ready", "wip_start"]).reset_index(drop=True)
    return result[["resource_code", "to_center", "wip_start", "wip_ready", "qty_ea", "lot"]]

def merge_wip_as_moves(moves_df: pd.DataFrame, wip_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """WIP 데이터를 moves에 병합"""
    if wip_df is None or wip_df.empty:
        return moves_df
    
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
    
    return pd.concat([moves_df, wip_moves], ignore_index=True)

@st.cache_data(ttl=1800)
def build_timeline(snap_long: pd.DataFrame, moves: pd.DataFrame, 
                   centers_sel: List[str], skus_sel: List[str],
                   start_dt: pd.Timestamp, end_dt: pd.Timestamp, 
                   horizon_days: int = 0) -> pd.DataFrame:
    """새로운 WIP 처리 로직 - event_date 손실 문제 해결"""
    horizon_end = end_dt + pd.Timedelta(days=horizon_days)
    full_dates = pd.date_range(start_dt, horizon_end, freq="D")

    # 1) 실제 센터 라인 (스냅샷 기반)
    base = snap_long[
        snap_long["center"].isin(centers_sel) &
        snap_long["resource_code"].isin(skus_sel)
    ].copy().rename(columns={"snapshot_date":"date"})
    
    if base.empty:
        return pd.DataFrame(columns=["date","center","resource_code","stock_qty"])
    
    base = base[(base["date"] >= start_dt) & (base["date"] <= end_dt)]
    lines = []

    # 실제 센터 라인 생성
    for (ct, sku), grp in base.groupby(["center","resource_code"]):
        grp = grp.sort_values("date")
        last_dt = grp["date"].max()

        if horizon_days > 0:
            proj_dates = pd.date_range(last_dt + pd.Timedelta(days=1), horizon_end, freq="D")
            proj_df = pd.DataFrame({"date": proj_dates, "center": ct,
                                  "resource_code": sku, "stock_qty": 0})
            grp = pd.concat([grp, proj_df], ignore_index=True)

        ts = grp[["date","stock_qty"]].copy()
        ts["stock_qty"] = ts["stock_qty"].astype(float)
        ts["resource_code"] = sku
        ts["center"] = ct

        # 해당 SKU의 이동 내역 (WIP 제외)
        mv = moves[
            (moves["resource_code"] == sku) &
            ((moves["from_center"] == ct) | (moves["to_center"] == ct)) &
            (moves["carrier_mode"] != "WIP")
        ].copy()

        # 출발: onboard_date에 - (센터에서 나감)
        eff_minus = (
            mv[mv["from_center"] == ct]
            .dropna(subset=["onboard_date"])
            .groupby("onboard_date", as_index=False)["qty_ea"].sum()
            .rename(columns={"onboard_date":"date","qty_ea":"delta"})
        )
        eff_minus["delta"] *= -1

        # 도착: event_date에 + (센터로 들어옴)
        eff_plus = (
            mv[mv["to_center"] == ct]
            .dropna(subset=["event_date"])
            .groupby("event_date", as_index=False)["qty_ea"].sum()
            .rename(columns={"event_date":"date","qty_ea":"delta"})
        )

        eff_all = pd.concat([eff_minus, eff_plus], ignore_index=True).sort_values("date")
        for d, delta in zip(eff_all["date"], eff_all["delta"]):
            ts.loc[ts["date"] >= d, "stock_qty"] = ts.loc[ts["date"] >= d, "stock_qty"] + delta

        ts["stock_qty"] = ts["stock_qty"].clip(lower=0)
        lines.append(ts)

    # 2) WIP 라인 생성 (새로운 로직)
    wip_moves = moves[moves["carrier_mode"] == "WIP"].copy()
    if not wip_moves.empty:
        for sku in skus_sel:
            sku_wip = wip_moves[wip_moves["resource_code"] == sku]
            if sku_wip.empty:
                continue
                
            # WIP 시작 (onboard_date에 +)
            wip_start = (
                sku_wip[sku_wip["onboard_date"].notna()]
                .groupby("onboard_date", as_index=False)["qty_ea"].sum()
                .rename(columns={"onboard_date":"date","qty_ea":"delta"})
            )
            
            # WIP 완료 (event_date에 -)
            wip_complete = (
                sku_wip[sku_wip["event_date"].notna()]
                .groupby("event_date", as_index=False)["qty_ea"].sum()
                .rename(columns={"event_date":"date","qty_ea":"delta"})
            )
            wip_complete["delta"] *= -1
            
            # WIP 라인 생성
            wip_deltas = pd.concat([wip_start, wip_complete], ignore_index=True)
            if not wip_deltas.empty:
                s = pd.Series(0, index=pd.to_datetime(full_dates))
                for d, v in wip_deltas.groupby("date")["delta"].sum().items():
                    d = pd.to_datetime(d)
                    if d < full_dates[0]:
                        s.iloc[:] = s.iloc[:] + v
                    else:
                        s.loc[s.index >= d] = s.loc[s.index >= d] + v

                wip_df = pd.DataFrame({
                    "date": s.index,
                    "center": "생산중",
                    "resource_code": sku,
                    "stock_qty": s.values.clip(min=0)
                })
                lines.append(wip_df)
                
                # WIP 완료 물량을 센터 라인에 반영
                wip_done = sku_wip[sku_wip["event_date"].notna()].copy()
                if not wip_done.empty:
                    add_to_center = (
                        wip_done.groupby(["to_center","event_date"], as_index=False)["qty_ea"].sum()
                        .rename(columns={"to_center":"center","event_date":"date","qty_ea":"delta"})
                    )
                    
                    # 해당 센터 라인에 WIP 완료 물량 추가
                    for idx_line, ts_line in enumerate(lines):
                        if ts_line.empty or ts_line["resource_code"].iloc[0] != sku:
                            continue
                        ct_name = ts_line["center"].iloc[0]
                        if ct_name in ("생산중", "이동중"):
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

    result = pd.concat(lines, ignore_index=True)
    result["date"] = pd.to_datetime(result["date"]).dt.normalize()
    result = result[(result["date"] >= start_dt) & (result["date"] <= horizon_end)]
    return result.sort_values(["resource_code","center","date"]).reset_index(drop=True)

# === 메인 앱 ===
st.set_page_config(page_title="SCM 재고 대시보드", layout="wide")

st.title("🚀 SCM 재고 대시보드 (새로운 버전)")

# 탭 생성
tab1, tab2, tab3 = st.tabs(["📊 Excel 업로드", "📈 CSV 업로드", "☁️ Google Sheets"])

# 앱 시작 시 Google Sheets에서 자동 로드 시도
if "snap_long" not in locals():
    try:
        with st.spinner("Google Sheets에서 데이터를 자동 로드하는 중..."):
            df_move, df_ref, df_incoming = load_from_gsheet()
            moves_raw = normalize_moves(df_move)
            snap_long = normalize_refined_snapshot(df_ref)
            
            # WIP 불러오기 및 병합
            wip_df = load_wip_from_incoming(df_incoming)
            moves = merge_wip_as_moves(moves_raw, wip_df)
            
            st.success(f"✅ Google Sheets 자동 로드 완료! WIP {len(wip_df)}건 반영")
    except Exception as e:
        st.error(f"Google Sheets 자동 로드 실패: {e}")
        st.caption("WebApp URL, 시트 ID, 토큰을 확인해주세요.")

# 센터 목록 생성 (정리된 버전)
if "snap_long" in locals():
    centers_snap = set(snap_long["center"].dropna().astype(str).unique().tolist())
    centers_moves = set(moves["from_center"].dropna().astype(str).unique().tolist() + 
                       moves["to_center"].dropna().astype(str).unique().tolist())

    # 센터명 통일 + 유효하지 않은 센터 제거
    centers_moves_unified = set()
    for center in centers_moves:
        if center == "AcrossBUS":
            centers_moves_unified.add("어크로스비US")
        elif center not in ["WIP", "In-Transit", "", "nan", "None"]:
            centers_moves_unified.add(center)

    centers_snap_clean = set()
    for center in centers_snap:
        if center not in ["WIP", "In-Transit", "", "nan", "None"]:
            centers_snap_clean.add(center)

    centers = sorted(list(centers_snap_clean | centers_moves_unified))
    skus = sorted(snap_long["resource_code"].dropna().astype(str).unique().tolist())

    # 필터 UI
    st.sidebar.header("필터")
    centers_sel = st.sidebar.multiselect("센터 선택", centers, default=(["태광KR"] if "태광KR" in centers else centers[:1]))
    skus_sel = st.sidebar.multiselect("SKU 선택", skus, default=([s for s in ["BA00022","BA00021"] if s in skus] or skus[:2]))

    # 기간 설정
    today = pd.Timestamp.today().normalize()
    start_dt = st.sidebar.date_input("시작일", value=today - pd.Timedelta(days=42))
    end_dt = st.sidebar.date_input("종료일", value=today + pd.Timedelta(days=60))
    
    start_dt = pd.Timestamp(start_dt).tz_localize(None)
    end_dt = pd.Timestamp(end_dt).tz_localize(None)
    
    # snap_long의 date 컬럼을 타임존 없는 것으로 변환
    snap_long["date"] = snap_long["date"].dt.tz_localize(None)
    
    # moves의 모든 날짜 컬럼을 타임존 없는 것으로 변환
    for date_col in ["onboard_date", "arrival_date", "inbound_date", "real_departure", "event_date"]:
        if date_col in moves.columns:
            moves[date_col] = moves[date_col].dt.tz_localize(None)

    # 타임라인 생성
    if centers_sel and skus_sel:
        timeline = build_timeline(snap_long, moves, centers_sel, skus_sel, start_dt, end_dt, horizon_days=60)
        
        if not timeline.empty:
            # 차트 생성
            fig = px.line(timeline, x="date", y="stock_qty", color="center", 
                         title="재고 추이", labels={"stock_qty": "재고량(EA)"})
            
            # 오늘 표시선 추가
            fig.add_vline(x=today, line_dash="solid", line_color="rgba(255, 0, 0, 0.4)", line_width=1)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 데이터 테이블
            st.subheader("재고 데이터")
            st.dataframe(timeline, use_container_width=True)
        else:
            st.warning("선택된 조건에 해당하는 데이터가 없습니다.")
    else:
        st.info("센터와 SKU를 선택해주세요.")
else:
    st.info("데이터를 로드해주세요.")
