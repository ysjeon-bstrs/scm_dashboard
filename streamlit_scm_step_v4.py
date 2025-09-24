
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import re
from datetime import datetime, timedelta
from io import BytesIO

st.set_page_config(page_title="글로벌 대시보드 — v4", layout="wide")

st.title("📦 센터×SKU 재고 흐름 (계단식) 대시보드 — v4")

st.caption("현재 재고는 항상 **스냅샷 기준**입니다. In-Transit / WIP 라인은 예측용 가상 라인으로, 스냅샷 수치에는 반영되지 않습니다.")

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
    # Streamlit UploadedFile 지원
    data = file.read() if hasattr(file, "read") else file
    bio = BytesIO(data) if isinstance(data, (bytes, bytearray)) else file

    xl = pd.ExcelFile(bio, engine="openpyxl")
    need = {"SCM_통합": None, "sample_snapshot": None}
    for s in xl.sheet_names:
        if s in need:
            need[s] = s
    if need["SCM_통합"] is None or need["sample_snapshot"] is None:
        st.error("엑셀에 'SCM_통합'과 'sample_snapshot' 시트가 필요합니다.")
        st.stop()

    df_move = pd.read_excel(bio, sheet_name=need["SCM_통합"], engine="openpyxl")
    bio.seek(0)
    df_snap = pd.read_excel(bio, sheet_name=need["sample_snapshot"], engine="openpyxl")
    bio.seek(0)
    # WIP 시트는 있을 수도/없을 수도
    wip_df = None
    if "입고예정내역" in xl.sheet_names:
        wip_df = pd.read_excel(bio, sheet_name="입고예정내역", engine="openpyxl")
    return df_move, df_snap, wip_df

def normalize_snapshot(df_snap):
    df = df_snap.copy()
    df.columns = [str(c).strip() for c in df.columns]
    rename_map = {}
    for c in df.columns:
        cs = str(c)
        if ("스냅샷" in cs and "일자" in cs) or cs.lower() in ["snapshot_date","date"]:
            rename_map[c] = "snapshot_date"
        elif cs.strip() in ["창고명","center"]:
            rename_map[c] = "center"
    df = df.rename(columns=rename_map)
    if "snapshot_date" not in df.columns or "center" not in df.columns:
        st.error("sample_snapshot 시트에 '스냅샷 일자'와 '창고명'이 필요합니다.")
        st.stop()
    id_cols = ["snapshot_date", "center"]
    sku_cols = [c for c in df.columns if c not in id_cols and "스냅샷" not in str(c)]
    longy = df.melt(id_vars=id_cols, value_vars=sku_cols, var_name="resource_code", value_name="stock_qty")
    longy["snapshot_date"] = pd.to_datetime(longy["snapshot_date"], errors="coerce")
    longy["stock_qty"] = pd.to_numeric(longy["stock_qty"], errors="coerce").fillna(0).astype(int)
    return longy.dropna(subset=["snapshot_date","center","resource_code"])

def normalize_moves(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    resource_code = _coalesce_columns(df, [["resource_code","상품코드","RESOURCE_CODE"]])
    qty_ea       = _coalesce_columns(df, [["qty_ea","QTY_EA","수량(EA)"]])
    carrier_mode = _coalesce_columns(df, [["carrier_mode","운송방법","carrier mode"]])
    from_center  = _coalesce_columns(df, [["from_center","출발창고"]])
    to_center    = _coalesce_columns(df, [["to_center","도착창고"]])
    onboard_date = _coalesce_columns(df, [["onboard_date","배정일","출발일","H"]], parse_date=True)  # H열 의미
    arrival_date = _coalesce_columns(df, [["arrival_date","도착일","eta_date","ETA"]], parse_date=True)
    inbound_date = _coalesce_columns(df, [["inbound_date","입고일"]], parse_date=True)
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
        df_move, df_snap, df_incoming = load_from_excel(xfile)
        moves_raw = normalize_moves(df_move)
        snap_long = normalize_snapshot(df_snap)

        # WIP 불러오기 및 병합
        try:
            wip_df = load_wip_from_incoming(df_incoming)
            moves = merge_wip_as_moves(moves_raw, wip_df)
            st.success(f"WIP {len(wip_df)}건 반영 완료" if wip_df is not None and not wip_df.empty else "WIP 없음")
        except Exception as e:
            moves = moves_raw
            st.warning(f"WIP 불러오기 실패: {e}")

with tab2:
    cs_snap = st.file_uploader("sample_snapshot.csv 업로드", type=["csv"], key="snapcsv")
    cs_move = st.file_uploader("SCM_통합.csv 업로드", type=["csv"], key="movecsv")
    if cs_snap is not None and cs_move is not None:
        snap_raw = pd.read_csv(cs_snap)
        move_raw = pd.read_csv(cs_move)
        moves = normalize_moves(move_raw)
        if "resource_code" in [c.lower() for c in snap_raw.columns]:
            sr = snap_raw.copy()
            cn = {}
            for c in sr.columns:
                cl = c.lower()
                if cl in ["date","snapshot_date","스냅샷 일자"]:
                    cn[c] = "snapshot_date"
                elif cl in ["center","창고명"]:
                    cn[c] = "center"
                elif cl == "resource_code":
                    cn[c] = "resource_code"
                elif cl in ["stock_qty","qty","quantity"]:
                    cn[c] = "stock_qty"
            sr = sr.rename(columns=cn)
            sr["snapshot_date"] = pd.to_datetime(sr["snapshot_date"], errors="coerce")
            sr["stock_qty"] = pd.to_numeric(sr["stock_qty"], errors="coerce").fillna(0).astype(int)
            snap_long = sr[["snapshot_date","center","resource_code","stock_qty"]].dropna()
        else:
            snap_long = normalize_snapshot(snap_raw)

if "snap_long" not in locals():
    st.info("엑셀 또는 CSV를 업로드하면 필터/차트가 나타납니다.")
    st.stop()

# -------------------- Filters --------------------
centers = sorted(snap_long["center"].dropna().astype(str).unique().tolist())
skus = sorted(snap_long["resource_code"].dropna().astype(str).unique().tolist())
min_date = snap_long["snapshot_date"].min()
max_date = snap_long["snapshot_date"].max()

st.sidebar.header("필터")
centers_sel = st.sidebar.multiselect("센터 선택", centers, default=(["태광KR"] if "태광KR" in centers else centers[:1]))
skus_sel = st.sidebar.multiselect("SKU 선택", skus, default=([s for s in ["BA00022","BA00021"] if s in skus] or skus[:2]))
date_range = st.sidebar.slider("기간",
    min_value=min(min_date.to_pydatetime(), (pd.Timestamp.today().normalize() - timedelta(days=60)).to_pydatetime()),
    max_value=max(max_date.to_pydatetime(), (pd.Timestamp.today().normalize() + timedelta(days=60)).to_pydatetime()),
    value=((pd.Timestamp.today().normalize() - timedelta(days=20)).to_pydatetime(), (pd.Timestamp.today().normalize() + timedelta(days=20)).to_pydatetime()),
    format="YYYY-MM-DD")
horizon = st.sidebar.number_input("미래 전망 일수", min_value=0, max_value=60, value=20)
show_wip   = st.sidebar.checkbox("WIP 표시", value=True)

start_dt = pd.to_datetime(date_range[0]).normalize()
end_dt = pd.to_datetime(date_range[1]).normalize()

# -------------------- KPIs --------------------
st.subheader("요약 KPI")

# 스냅샷 기준 최신일
latest_dt = snap_long["snapshot_date"].max()
latest_dt_str = pd.to_datetime(latest_dt).strftime("%Y-%m-%d")

# SKU별 현재 재고(선택 센터 기준, 최신 스냅샷)
latest_rows = snap_long[(snap_long["snapshot_date"] == latest_dt) & (snap_long["center"].isin(centers_sel))]
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
k_it.metric("이동 중 재고 (In-Transit)", f"{in_transit_total:,}")
k_wip.metric("현재 WIP(미완료 생산)", f"{wip_today:,}")

st.divider()

# -------------------- Step Chart (Plotly) --------------------
st.subheader("계단식 재고 흐름")
timeline = build_timeline(snap_long, moves, centers_sel, skus_sel,
                          start_dt, end_dt, horizon_days=horizon)
if timeline.empty:
    st.info("선택 조건에 해당하는 타임라인 데이터가 없습니다.")
else:
    # 기간 + 전망 범위 슬라이스
    vis_df = timeline[
        (timeline["date"] >= start_dt) &
        (timeline["date"] <= (end_dt + pd.Timedelta(days=horizon)))
    ].copy()

    # In-Transit은 항상 통합
    vis_df["center"] = vis_df["center"].str.replace(r"^In-Transit.*", "In-Transit", regex=True)

    # WIP 표시 옵션
    if not show_wip:
        vis_df = vis_df[vis_df["center"] != "WIP"]

    # 라벨 재생성 (반드시 마지막에)
    vis_df["label"] = vis_df["resource_code"] + " @ " + vis_df["center"]

    # 그리기
    fig = px.line(
        vis_df,
        x="date",
        y="stock_qty",
        color="label",
        line_shape="hv",
        title="선택한 SKU × 센터(및 In-Transit/WIP) 계단식 재고 흐름",
        render_mode="svg"
    )
    fig.update_traces(
        mode="lines",
        hovertemplate="날짜: %{x|%Y-%m-%d}<br>재고: %{y:,} EA<br>%{fullData.name}<extra></extra>"
    )
    fig.update_layout(
        hovermode="x unified",
        xaxis_title="Date",
        yaxis_title="Inventory (qty_ea)",
        legend_title_text="SKU @ Center / In-Transit(파랑 점선) / WIP(빨강 점선)",
        margin=dict(l=20, r=20, t=60, b=20)
    )

    # ---- 스타일: 실제 센터 얇게(2px, 0.9), IT/WIP 점선 2px 0.8 ----
COLOR_WIP = "#e11d48"   # 빨강 (WIP)
COLOR_IT  = "#2563eb"   # 파랑 (In-Transit)

# 파랑/빨강과 겹치지 않는 '센터' 전용 팔레트
CENTER_PALETTE = [
    "#111827",  # almost black (neutral)
    "#475569",  # slate
    "#0f766e",  # teal
    "#d97706",  # amber
    "#7c3aed",  # violet
    "#16a34a",  # green
    "#a855f7",  # purple
    "#f59e0b",  # orange/amber
]

# 센터 라인: 얇고(2px) 약간 투명(0.9)
center_colors = {}
ci = 0
for i, tr in enumerate(fig.data):
    name = (tr.name or "")
    if ("In-Transit" in name) or (" WIP" in name):
        continue
    fig.data[i].update(line=dict(width=2.0), opacity=0.9)
    center = name.split(" @ ")[-1] if " @ " in name else ""
    if center not in center_colors:
        center_colors[center] = CENTER_PALETTE[ci % len(CENTER_PALETTE)]
        ci += 1
    fig.data[i].update(line=dict(color=center_colors[center]))
    fig.data[i].legendgroup = "Center"
    fig.data[i].legendrank = 10  # 상단

# In-Transit: 파랑 점선 2px, 0.8
for i, tr in enumerate(fig.data):
    name = (tr.name or "")
    if "In-Transit" in name:
        fig.data[i].update(line=dict(color=COLOR_IT, dash="dot", width=2.0), opacity=0.8)
        fig.data[i].legendgroup = "In-Transit"
        fig.data[i].legendrank = 20

# WIP: 빨강 점선 2px, 0.8
for i, tr in enumerate(fig.data):
    name = (tr.name or "")
    if name.endswith(" @ WIP") or " WIP" in name:
        fig.data[i].update(line=dict(color=COLOR_WIP, dash="dot", width=2.0), opacity=0.8)
        fig.data[i].legendgroup = "WIP"
        fig.data[i].legendrank = 30

    # 차트 키 (필터 조합으로 유니크하게)
chart_key = f"stepchart|centers={','.join(centers_sel)}|skus={','.join(skus_sel)}|{start_dt:%Y%m%d}-{end_dt:%Y%m%d}|h{horizon}|w{int(show_wip)}"

st.plotly_chart(
    fig,
    use_container_width=True,
    config={"displaylogo": False},
    key=chart_key,  # ← 고유 키
)


# -------------------- Upcoming Arrivals --------------------
st.subheader("입고 예정 내역 (선택 센터/SKU)")

today = pd.Timestamp.today().normalize()
window_end = end_dt + pd.Timedelta(days=horizon)
start_cut = today  # 과거 ETA 제외 (D-0 포함)

# (A) 운송(비 WIP) - 태광 제외 센터
arr_transport = moves[
    (moves["event_date"].notna()) &
    (moves["event_date"] >= start_cut) & (moves["event_date"] <= window_end) &
    (moves["carrier_mode"].astype(str).str.upper() != "WIP") &
    (moves["to_center"].astype(str).isin([c for c in centers_sel if c != "태광KR"])) &
    (moves["resource_code"].astype(str).isin(skus_sel))
].copy()
arr_transport["lot"] = ""  # 일반 이동건은 lot 비움

# (B) WIP - 태광KR 선택 시
arr_wip = pd.DataFrame()
if "태광KR" in centers_sel:
    arr_wip = moves[
        (moves["event_date"].notna()) &
        (moves["event_date"] >= start_cut) & (moves["event_date"] <= window_end) &
        (moves["carrier_mode"].astype(str).str.upper() == "WIP") &
        (moves["to_center"].astype(str) == "태광KR") &
        (moves["resource_code"].astype(str).isin(skus_sel))
    ].copy()

upcoming = pd.concat([arr_transport, arr_wip], ignore_index=True)
if upcoming.empty:
    st.caption("도착 예정 없음 (오늘 이후)")
else:
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
