
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import re
from datetime import datetime, timedelta
from io import BytesIO

st.set_page_config(page_title="글로벌 대시보드 — v4", layout="wide")

st.title("📦 SCM 재고 흐름 대시보드 — v4")

st.caption("현재 재고는 항상 **스냅샷 기준(snap_정제)**입니다. 이동중 / 생산중 라인은 예측용 가상 라인입니다.")

# -------------------- Helpers --------------------
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
    df = df_ref.copy()
    cols = {c.strip(): c for c in df.columns}
    req = ["date","center","resource_code","stock_qty"]
    miss = [c for c in req if c not in cols]
    if miss:
        st.error(f"'snap_정제' 시트에 누락된 컬럼: {miss}")
        st.stop()
    # 이름 정규화
    df = df.rename(columns={cols["date"]:"date",
                            cols["center"]:"center",
                            cols["resource_code"]:"resource_code",
                            cols["stock_qty"]:"stock_qty"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["center"] = df["center"].astype(str)
    df["resource_code"] = df["resource_code"].astype(str)
    df["stock_qty"] = pd.to_numeric(df["stock_qty"], errors="coerce").fillna(0).astype(int)
    return df.dropna(subset=["date","center","resource_code"])


def normalize_moves(df):
    df = df.copy()
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
        "qty_ea": pd.to_numeric(qty_ea, errors="coerce").fillna(0).astype(int),
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


def _parse_po_date(po_str):
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

def load_wip_from_incoming(df_incoming, default_center="태광KR"):
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
    df.columns = [str(c).strip().lower() for c in df.columns]

    # 컬럼 추론
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

    # 발주일 파싱 → wip_start
    out["wip_start"] = df[po_col].map(_parse_po_date) if po_col else pd.NaT
    # 발주일이 못 읽혔으면(예외 케이스) 최소한 wip_ready - 10일로 보정
    mask_na = out["wip_start"].isna() & out["wip_ready"].notna()
    out.loc[mask_na, "wip_start"] = out.loc[mask_na, "wip_ready"] - pd.to_timedelta(10, unit="D")

    # 유효값만
    out = out.dropna(subset=["resource_code","wip_ready","wip_start"]).reset_index(drop=True)
    return out[["resource_code","to_center","wip_start","wip_ready","qty_ea","lot"]]

def merge_wip_as_moves(moves_df, wip_df):
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

# ===== 소비(소진) 추세 + 이벤트 가중치 =====
import numpy as np
import pandas as pd

def estimate_daily_consumption(snap_long: pd.DataFrame,
                               centers_sel, skus_sel,
                               asof_dt: pd.Timestamp,
                               lookback_days: int = 28) -> dict:
    """
    최근 lookback_days 동안 스냅샷으로 SKU×센터별 일일 소진량(개/일)을 추정.
    감소 추세만 소진으로 간주(증가면 0). 반환: {(center, sku): rate}
    """
    snap = snap_long.rename(columns={"snapshot_date":"date"}).copy()
    snap["date"] = pd.to_datetime(snap["date"], errors="coerce").dt.normalize()
    start = (asof_dt - pd.Timedelta(days=lookback_days-1)).normalize()

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
        a = np.polyfit(x, y, 1)[0]       # 1차 회귀 기울기
        daily_out = max(0.0, -a)         # 감소분만 소진으로 간주
        rates[(ct, sku)] = daily_out
    return rates

def apply_consumption_with_events(
    timeline: pd.DataFrame,
    snap_long: pd.DataFrame,
    centers_sel, skus_sel,
    start_dt: pd.Timestamp, end_dt: pd.Timestamp,
    lookback_days: int = 28,
    events: list[dict] | None = None
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
            u = float(e.get("uplift", 0.0))
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




def build_timeline(snap_long, moves, centers_sel, skus_sel,
                   start_dt, end_dt, horizon_days=0):
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

    lines =()

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
    mv_sel = moves[
        moves["resource_code"].isin(skus_sel) &
        (moves["from_center"].astype(str).isin(centers_sel) | moves["to_center"].astype(str).isin(centers_sel) | (moves["carrier_mode"].astype(str).str.upper()=="WIP"))
    ].copy()

    for sku, g in mv_sel.groupby("resource_code"):
        # Non-WIP 통합 In-Transit
        g_nonwip = g[g["carrier_mode"].astype(str).str.upper() != "WIP"]
        if not g_nonwip.empty:
            add_onboard = (
                g_nonwip[g_nonwip["onboard_date"].notna()]
                .groupby("onboard_date", as_index=False)["qty_ea"].sum()
                .rename(columns={"onboard_date":"date","qty_ea":"delta"})
            )
            add_event = (
                g_nonwip[g_nonwip["event_date"].notna()]
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

        # WIP 별도 라인
        g_wip = g[g["carrier_mode"].astype(str).str.upper() == "WIP"]
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
                    "center": "WIP",
                    "resource_code": sku,
                    "stock_qty": s.values.clip(min=0)
                })
                lines.append(vdf)

    if not lines:
        return pd.DataFrame(columns=["date","center","resource_code","stock_qty"])

    out = pd.concat(lines, ignore_index=True)
    out = out[(out["date"] >= start_dt) & (out["date"] <= horizon_end)]
    return out

# -------------------- Tabs for inputs --------------------
tab1, tab2 = st.tabs(["엑셀 업로드", "CSV 수동 업로드"])

with tab1:
    xfile = st.file_uploader("엑셀 업로드 (.xlsx)", type=["xlsx"], key="excel")
    if xfile is not None:
        df_move, df_ref, df_incoming = load_from_excel(xfile)
        moves_raw = normalize_moves(df_move)
        snap_long = normalize_refined_snapshot(df_ref)


        # WIP 불러오기 및 병합
        try:
            wip_df = load_wip_from_incoming(df_incoming)
            moves = merge_wip_as_moves(moves_raw, wip_df)
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

if "snap_long" not in locals():
    st.info("엑셀 또는 CSV를 업로드하면 필터/차트가 나타납니다.")
    st.stop()

# -------------------- Filters --------------------
centers = sorted(snap_long["center"].dropna().astype(str).unique().tolist())
skus = sorted(snap_long["resource_code"].dropna().astype(str).unique().tolist())
min_date = snap_long["date"].min()
max_date = snap_long["date"].max()

st.sidebar.header("필터")
centers_sel = st.sidebar.multiselect("센터 선택", centers, default=(["태광KR"] if "태광KR" in centers else centers[:1]))
skus_sel = st.sidebar.multiselect("SKU 선택", skus, default=([s for s in ["BA00022","BA00021"] if s in skus] or skus[:2]))

# 전망 일수 (기간 자동 설정의 기준)
horizon = st.sidebar.number_input("미래 전망 일수", min_value=0, max_value=60, value=20)

# 기간 모드
today = pd.Timestamp.today().normalize()
range_mode = st.sidebar.radio("기간 설정", ["자동(오늘±전망)", "수동(직접 지정)"], index=0)

if range_mode.startswith("자동"):
    start_dt = (today - pd.Timedelta(days=int(horizon))).normalize()
    end_dt   = (today + pd.Timedelta(days=int(horizon))).normalize()
    st.sidebar.markdown(f"**기간:** {start_dt:%Y-%m-%d} → {end_dt:%Y-%m-%d}")
else:
    date_range = st.sidebar.slider(
        "기간",
        min_value=min(min_date.to_pydatetime(), (today - timedelta(days=60)).to_pydatetime()),
        max_value=max(max_date.to_pydatetime(), (today + timedelta(days=60)).to_pydatetime()),
        value=((today - timedelta(days=20)).to_pydatetime(), (today + timedelta(days=20)).to_pydatetime()),
        format="YYYY-MM-DD",
    )
    start_dt = pd.to_datetime(date_range[0]).normalize()
    end_dt   = pd.to_datetime(date_range[1]).normalize()


# 빌드용 전망일: 기간 끝이 최신 스냅샷보다 뒤일 때만 그 차이만큼 예측 필요
_latest_snap = snap_long["date"].max()
proj_days_for_build = max(0, int((end_dt - _latest_snap).days))

st.sidebar.header("표시 옵션")

# 소진 추세 적용
use_cons_forecast = st.sidebar.checkbox("4주 추세 기반 소진 예측 적용", value=True)
lookback_days = st.sidebar.number_input("추세 계산 기간(일)", min_value=7, max_value=56, value=28, step=7)

# 이벤트/프로모션: 단일 기간 + 가중치
with st.sidebar.expander("이벤트/프로모션 가중치 (+%)", expanded=False):
    enable_event = st.checkbox("이벤트 가중치 적용", value=False)
    s = st.date_input("시작일")
    t = st.date_input("종료일")
    u = st.number_input("가중치(%)", min_value=-100.0, max_value=300.0, value=30.0, step=5.0)
    events = [{"start": pd.Timestamp(s).strftime("%Y-%m-%d"),
               "end":   pd.Timestamp(t).strftime("%Y-%m-%d"),
               "uplift": (u/100.0)}] if enable_event else []

# ‘생산중(WIP)’, ‘이동중(In-Transit)’ 토글 (기본 ON)
show_prod = st.sidebar.checkbox("생산중(미완료 생산) 표시", value=True)
show_transit = st.sidebar.checkbox("이동중 표시", value=True)



# -------------------- KPIs --------------------
st.subheader("요약 KPI")

# 스냅샷 기준 최신일
latest_dt = snap_long["date"].max()
latest_dt_str = pd.to_datetime(latest_dt).strftime("%Y-%m-%d")

# SKU별 현재 재고(선택 센터 기준, 최신 스냅샷)
latest_rows = snap_long[(snap_long["date"] == latest_dt) & (snap_long["center"].isin(centers_sel))]
sku_totals = {sku: int(latest_rows[latest_rows["resource_code"]==sku]["stock_qty"].sum()) for sku in skus_sel}

# In-Transit (WIP 제외, 입/출고 모두 포함)
today = pd.Timestamp.today().normalize()
in_transit_mask = (
    (moves["onboard_date"].notna()) &
    (moves["onboard_date"] <= today) &
    (moves["inbound_date"].isna()) &
    ((moves["arrival_date"].isna()) | (moves["arrival_date"] > today)) &
    ((moves["to_center"].astype(str).isin(centers_sel)) | (moves["from_center"].astype(str).isin(centers_sel))) &
    (moves["resource_code"].astype(str).isin(skus_sel)) &
    (moves["carrier_mode"].astype(str).str.upper() != "WIP")
)
in_transit_total = int(moves[in_transit_mask]["qty_ea"].sum())

# WIP(오늘 기준 잔량, 선택 센터/SKU 범위)
wip_moves = moves[moves["carrier_mode"].astype(str).str.upper() == "WIP"].copy()
wip_moves = wip_moves[
    wip_moves["to_center"].astype(str).isin(centers_sel) &
    wip_moves["resource_code"].astype(str).isin(skus_sel)
].copy()
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
    # In-Transit → 이동중 (정규화 후 치환)
    vis_df["center"] = vis_df["center"].str.replace(r"^In-Transit.*$", "이동중", regex=True)
    # WIP → 생산중
    vis_df.loc[vis_df["center"] == "WIP", "center"] = "생산중"

    # ✅ 태광KR 미선택 시 '생산중' 라인 숨김
    if "태광KR" not in centers_sel:
        vis_df = vis_df[vis_df["center"] != "생산중"]
    
    if not show_prod:
        vis_df = vis_df[vis_df["center"] != "생산중"]
    if not show_transit:
        vis_df = vis_df[vis_df["center"] != "이동중"]

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
        xaxis_title="Date",
        yaxis_title="Inventory (qty_ea)",
        legend_title_text="SKU @ Center / 이동중(점선) / 생산중(점선)",
        margin=dict(l=20, r=20, t=60, b=20)
    )
    # Y축 눈금 정수로
    fig.update_yaxes(tickformat=",.0f")
    # 호버도 정수 천단위
    fig.update_traces(hovertemplate="날짜: %{x|%Y-%m-%d}<br>재고: %{y:,.0f} EA<br>%{fullData.name}<extra></extra>")


    # (D) 색상/선 스타일 (한국어 라벨 기준)
    PALETTE = [
        "#4E79A7","#F28E2B","#E15759","#76B7B2","#59A14F",
        "#EDC948","#B07AA1","#FF9DA7","#9C755F","#BAB0AC",
        "#1F77B4","#FF7F0E","#2CA02C","#D62728","#9467BD",
        "#8C564B","#E377C2","#7F7F7F","#BCBD22","#17BECF",
        "#8DD3C7","#FFFFB3","#BEBADA","#FB8072","#80B1D3",
        "#FDB462","#B3DE69","#FCCDE5","#D9D9D9","#BC80BD",
        "#CCEBC5","#FFED6F"
    ]
    # SKU → 고정 색
    sku_colors, ci = {}, 0
    for tr in fig.data:
        name = tr.name or ""
        if " @ " in name:
            sku = name.split(" @ ")[0]
            if sku not in sku_colors:
                sku_colors[sku] = PALETTE[ci % len(PALETTE)]
                ci += 1

    for i, tr in enumerate(fig.data):
        name = tr.name or ""
        if " @ " not in name:
            continue
        sku, kind = name.split(" @ ", 1)
        color = sku_colors.get(sku, PALETTE[0])

        if kind == "이동중":
            fig.data[i].update(line=dict(color=color, dash="dot", width=2.0), opacity=0.85)
            fig.data[i].legendgroup = f"{sku} (이동중)"
            fig.data[i].legendrank = 20
        elif kind == "생산중":
            fig.data[i].update(line=dict(color=color, dash="dot", width=2.0), opacity=0.85)
            fig.data[i].legendgroup = f"{sku} (생산중)"
            fig.data[i].legendrank = 30
        else:
            fig.data[i].update(line=dict(color=color, dash="solid", width=2.2), opacity=0.95)
            fig.data[i].legendgroup = f"{sku} (센터)"
            fig.data[i].legendrank = 10

    chart_key = (
        f"stepchart|centers={','.join(centers_sel)}|skus={','.join(skus_sel)}|"
        f"{start_dt:%Y%m%d}-{end_dt:%Y%m%d}|h{horizon}|prod{int(show_prod)}|tran{int(show_transit)}"
    )
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False}, key=chart_key)



# -------------------- Upcoming Arrivals --------------------
st.subheader("입고 예정 내역 (선택 센터/SKU)")

today = pd.Timestamp.today().normalize()
window_start = max(start_dt, today)   # ✅ 오늘 이후만
window_end   = end_dt

# (A) 운송(비 WIP)
arr_transport = moves[
    (moves["event_date"].notna()) &
    (moves["event_date"] >= window_start) & (moves["event_date"] <= window_end) &
    (moves["carrier_mode"].astype(str).str.upper() != "WIP") &
    (moves["to_center"].astype(str).isin([c for c in centers_sel if c != "태광KR"])) &
    (moves["resource_code"].astype(str).isin(skus_sel))
].copy()

# (B) WIP - 태광KR 선택 시
arr_wip = pd.DataFrame()
if "태광KR" in centers_sel:
    arr_wip = moves[
        (moves["event_date"].notna()) &
        (moves["event_date"] >= window_start) & (moves["event_date"] <= window_end) &
        (moves["carrier_mode"].astype(str).str.upper() == "WIP") &
        (moves["to_center"].astype(str) == "태광KR") &
        (moves["resource_code"].astype(str).isin(skus_sel))
    ].copy()

upcoming = pd.concat([arr_transport, arr_wip], ignore_index=True)

if upcoming.empty:
    st.caption("도착 예정 없음 (오늘 이후 / 선택 기간)")
else:
    # ✅ days_to_arrival는 항상 0 이상
    upcoming["days_to_arrival"] = (upcoming["event_date"] - today).dt.days
    upcoming = upcoming.sort_values(["event_date","to_center","resource_code"])
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

sim_tl = build_timeline(snap_long, moves, [sim_center], [sim_sku],
                        start_dt=min_date, end_dt=max_date, horizon_days=sim_days)
if sim_tl.empty:
    st.info("해당 조합의 타임라인이 없습니다.")
else:
    sim_target_date = sim_tl["date"].max()
    real_mask = (sim_tl["center"]==sim_center) & ~sim_tl["center"].isin(["WIP"]) & ~sim_tl["center"].str.startswith("In-Transit", na=False)
    sim_stock = int(sim_tl[(sim_tl["date"]==sim_target_date) & real_mask]["stock_qty"].sum())
    ok = sim_stock >= sim_qty
    st.metric(f"{sim_days}일 뒤({(today + pd.Timedelta(days=int(sim_days))).strftime('%Y-%m-%d')}) '{sim_center}'의 '{sim_sku}' 예상 재고",
              f"{sim_stock:,}", delta=f"필요 {sim_qty:,}")
    if ok:
        st.success("출고 가능")
    else:
        st.error("출고 불가")

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
c1, c2, c3 = st.columns([1.2, 1, 1])
with c1:
    q = st.text_input("SKU 필터(포함 검색)", "")
with c2:
    hide_zero = st.checkbox("총합=0 숨기기", value=True)
with c3:
    sort_by = st.selectbox("정렬 기준", ["총합"] + list(pivot.columns.drop("총합")), index=0)

# 5) 필터 적용
view = pivot.copy()
if q.strip():
    view = view[view.index.str.contains(q.strip(), case=False, regex=False)]
if hide_zero:
    view = view[view["총합"] > 0]

# 6) 정렬(내림차순)  
view = view.sort_values(by=sort_by, ascending=False)

# 7) 보여주기
st.dataframe(
    view.reset_index().rename(columns={"resource_code": "SKU"}),
    use_container_width=True, height=380
)

# 8) CSV 다운로드
csv_bytes = view.reset_index().rename(columns={"resource_code": "SKU"}).to_csv(index=False).encode("utf-8-sig")
st.download_button(
    "현재 표 CSV 다운로드",
    data=csv_bytes,
    file_name=f"centers_{'-'.join(centers_sel)}_snapshot_{latest_dt_str}.csv",
    mime="text/csv"
)

st.caption("※ 이 표는 **선택된 센터 전체 SKU**의 최신 스냅샷 재고입니다. 상단 ‘SKU 선택’과 무관하게 모든 SKU가 포함됩니다.")
