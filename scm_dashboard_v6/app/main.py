"""
SCM Dashboard v6 — Streamlit 엔트리

- v6 구조 검증: 컨트롤 수집 → 타임라인 섹션 → (옵션) 아마존/인벤토리 섹션 호출
- 데이터 로딩은 일단 v5 로더를 위임 사용하고, 후속 단계에서 v6 data로 전환
"""

from __future__ import annotations

# 패키지 임포트 경로 부트스트랩: 파일 실행 경로에서 상위 프로젝트 루트를 sys.path에 추가
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import streamlit as st

from scm_dashboard_v6.ui.controls import collect_sidebar_controls
from scm_dashboard_v6.features.timeline import render_timeline_section
from scm_dashboard_v6.features.amazon import render_amazon_panel
from scm_dashboard_v6.features.inventory_view import (
    render_inventory_pivot,
    render_upcoming_inbound,
    render_wip_progress,
)
from scm_dashboard_v6.data.loaders import load_gsheet, load_snapshot_raw, load_excel
from scm_dashboard_v5.ui import render_sku_summary_cards
from scm_dashboard_v4.processing import (
    load_wip_from_incoming,
    merge_wip_as_moves,
    normalize_moves,
    normalize_refined_snapshot,
)


def main() -> None:
    st.set_page_config(page_title="SCM Dashboard v6", layout="wide")
    st.title("SCM Dashboard v6")
    st.caption("v6 구조 도입 — v5 동작을 유지하면서 경계/모듈 분리")

    st.markdown("### 데이터 소스")
    st.caption("초기 단계에서는 v5 로더 위임 — 후속 단계에서 v6 data로 전환")
    # 기본값: 비어 있는 프레임으로 초기화(업로드 전용 화면 유지)
    df_move = pd.DataFrame()
    df_ref = pd.DataFrame()
    df_incoming = pd.DataFrame()
    snapshot_raw_df = None
    gsheet_loaded = False
    try:
        with st.spinner("Google Sheets 데이터 불러오는 중..."):
            g_mv, g_ref, g_in = load_gsheet()
            # v5와 동일하게 정규화 (단, 이미 정규화된 경우는 중복 적용하지 않음)
            g_mv = normalize_moves(g_mv)
            required_snap = {"date", "center", "resource_code", "stock_qty"}
            if not g_ref.empty and not required_snap.issubset(set(map(str, g_ref.columns))):
                g_ref = normalize_refined_snapshot(g_ref)
            # WIP 병합
            try:
                wip_df = load_wip_from_incoming(g_in)
                if wip_df is not None and not wip_df.empty:
                    g_mv = merge_wip_as_moves(g_mv, wip_df)
            except Exception:
                pass
            df_move, df_ref, df_incoming = g_mv, g_ref, g_in
            # 최소 유효성 통과 시에만 성공 메시지 표시
            gsheet_loaded = (not g_ref.empty) and required_snap.issubset(set(map(str, g_ref.columns)))
            # GSheet 경로에서는 snapshot_raw를 별도로 로드 (세션 캐시 활용)
            try:
                snapshot_raw_df = load_snapshot_raw()
            except Exception:
                snapshot_raw_df = None
    except Exception as exc:
        # 시크릿 부재 등으로 실패해도 업로드 경로를 노출해야 하므로 경고만 표시
        st.warning("Google Sheets API 인증 실패: secrets에 [google_sheets] 섹션이 없습니다. 아래에서 엑셀 업로드를 이용하세요.")
    if gsheet_loaded:
        st.success("Google Sheets 데이터가 로드되었습니다.")

    # 스냅샷 정규화: date 컬럼 통일
    snapshot_df = df_ref.copy()
    if not snapshot_df.empty:
        if "date" in snapshot_df.columns:
            snapshot_df["date"] = pd.to_datetime(snapshot_df["date"], errors="coerce").dt.normalize()
        elif "snapshot_date" in snapshot_df.columns:
            snapshot_df["date"] = pd.to_datetime(snapshot_df["snapshot_date"], errors="coerce").dt.normalize()
        else:
            snapshot_df["date"] = pd.NaT

    # (선택) 엑셀 업로드로 데이터 교체
    with st.expander("엑셀 파일 업로드 (선택)", expanded=False):
        st.caption("필요할 때 업로드하면 현재 데이터 소스를 엑셀로 교체합니다.")
        up = st.file_uploader("엑셀 업로드 (.xlsx)", type=["xlsx"], key="v6_excel")
        if up is not None:
            try:
                df_move_x, df_ref_x, _df_incoming_x, snapshot_raw_x = load_excel(up)
                # 정규화 (이미 정규화된 경우는 생략)
                df_move_x = normalize_moves(df_move_x)
                required_snap = {"date", "center", "resource_code", "stock_qty"}
                if not df_ref_x.empty and not required_snap.issubset(set(map(str, df_ref_x.columns))):
                    df_ref_x = normalize_refined_snapshot(df_ref_x)
                # 업로드에서도 WIP 병합
                try:
                    wip_x = load_wip_from_incoming(_df_incoming_x)
                    if wip_x is not None and not wip_x.empty:
                        df_move_x = merge_wip_as_moves(df_move_x, wip_x)
                except Exception:
                    pass
                df_move = df_move_x.copy()
                snapshot_df = df_ref_x.copy()
                snapshot_raw_df = snapshot_raw_x if snapshot_raw_x is not None else snapshot_raw_df
                if "date" in snapshot_df.columns:
                    snapshot_df["date"] = pd.to_datetime(snapshot_df["date"], errors="coerce").dt.normalize()
                elif "snapshot_date" in snapshot_df.columns:
                    snapshot_df["date"] = pd.to_datetime(snapshot_df["snapshot_date"], errors="coerce").dt.normalize()
                else:
                    snapshot_df["date"] = pd.NaT
                st.success("엑셀 데이터가 로드되었습니다.")
            except Exception as exc:
                st.error(f"엑셀 데이터 로딩 실패: {exc}")

    # 업로드/GS가 모두 실패한 경우에는 여기서 종료
    required_snap = {"date", "center", "resource_code", "stock_qty"}
    if snapshot_df.empty or not required_snap.issubset(set(map(str, snapshot_df.columns))):
        st.info("스냅샷 데이터가 없습니다. 엑셀을 업로드해 주세요.")
        # 업로드 영역은 이미 상단에서 노출되어 있음. 초기에 차트/표 렌더는 건너뜀
        return

    # 선택 옵션 후보 계산 (간단화)
    centers = sorted(snapshot_df.get("center", pd.Series([], dtype=str)).dropna().astype(str).unique().tolist())
    skus = sorted(snapshot_df.get("resource_code", pd.Series([], dtype=str)).dropna().astype(str).unique().tolist())

    if not centers or not skus:
        st.warning("센터 또는 SKU 정보를 찾을 수 없습니다.")
        return

    today = pd.Timestamp.today().normalize()
    latest_dt = snapshot_df["date"].dropna().max() if not snapshot_df.empty else pd.NaT
    # 스냅샷 과거는 42일, 미래 예측은 60일까지 보이도록 범위 설정
    bound_min = today - pd.Timedelta(days=42)
    bound_max = today + pd.Timedelta(days=60)

    ui = collect_sidebar_controls(
        centers=centers,
        skus=skus,
        bound_min=pd.Timestamp(bound_min).normalize(),
        bound_max=pd.Timestamp(bound_max).normalize(),
    )

    # 요약 KPI (v5와 동일 호출)
    st.subheader("요약 KPI")
    render_sku_summary_cards(
        snapshot_df,
        df_move,
        centers=ui.centers,
        skus=ui.skus,
        today=today,
        latest_snapshot=latest_dt,
        lag_days=int(getattr(ui, "inbound_lead_days", 7) or 7),
        start=ui.start,
        end=ui.end,
        lookback_days=ui.lookback_days,
        horizon_pad_days=60,
        events=[],
    )

    st.subheader("타임라인")
    timeline = render_timeline_section(
        snapshot=snapshot_df,
        moves=df_move,
        centers=ui.centers,
        skus=ui.skus,
        start=ui.start,
        end=ui.end,
        today=today,
        lookback_days=ui.lookback_days,
        lag_days=int(getattr(ui, "inbound_lead_days", 7) or 7),
        show_production=ui.show_production,
        show_in_transit=ui.show_in_transit,
    )

    st.divider()
    st.subheader("Amazon US 일별 판매 vs. 재고")
    if snapshot_raw_df is None:
        try:
            snapshot_raw_df = load_snapshot_raw()
        except Exception:
            snapshot_raw_df = None
    # v5 차트가 내부에서 moves_df 가공 시 event_date를 기대하는 부분을 회피하기 위해
    # 스텝 차트에서 얻은 피벗 기반 tidy(실측/예측)를 전달한다.
    # 또한 아마존 차트에는 아마존 센터만 표시되도록 필터링한다.
    amazon_centers = [c for c in ui.centers if c.upper().startswith("AMZ") or "AMAZON" in c.upper()]
    inv_actual_tidy = (
        timeline[(timeline["date"] <= today) & (timeline["center"].isin(amazon_centers))]
        if isinstance(timeline, pd.DataFrame)
        else None
    )
    inv_forecast_tidy = (
        timeline[(timeline["date"] > today) & (timeline["center"].isin(amazon_centers))]
        if isinstance(timeline, pd.DataFrame)
        else None
    )

    # 프로모션 설정에서 이벤트와 소비기반 예측 사용 여부 결정
    promo_enabled = bool(getattr(ui, "promotion_enabled", False))
    promo_events = (
        [
            {
                "start": pd.to_datetime(getattr(ui, "promotion_start", today)).normalize(),
                "end": pd.to_datetime(getattr(ui, "promotion_end", today)).normalize(),
                # v5 컨텍스트는 'uplift' 비율을 기대 (예: 0.3 => +30%)
                "uplift": float(getattr(ui, "promotion_percent", 0.0)) / 100.0,
            }
        ]
        if promo_enabled
        else None
    )

    render_amazon_panel(
        snapshot_long=snapshot_df,
        moves=df_move,
        snapshot_raw=snapshot_raw_df,
        centers=amazon_centers,
        skus=ui.skus,
        start=ui.start,
        end=ui.end,
        today=today,
        lookback_days=ui.lookback_days,
        promotion_events=promo_events,
        # 프로모션 적용 시 소비기반 예측을 사용해 uplift가 반영되도록 함
        use_consumption_forecast=not promo_enabled,
        inv_actual=inv_actual_tidy,
        inv_forecast=inv_forecast_tidy,
    )

    # --- 아래부터 표/테이블 섹션 ---
    # 간단 품명 매핑 (있으면)
    name_map: dict[str, str] = {}
    if "resource_name" in snapshot_df.columns:
        rows = snapshot_df.loc[snapshot_df["resource_name"].notna(), ["resource_code", "resource_name"]].copy()
        if not rows.empty:
            name_map = rows.drop_duplicates("resource_code").set_index("resource_code")["resource_name"].to_dict()

    # 확정 입고 / WIP 진행 현황 표 표시
    window_start = ui.start
    window_end = ui.end
    lag_days = int(getattr(ui, "inbound_lead_days", 7) or 7)
    st.divider()
    render_upcoming_inbound(
        moves=df_move,
        centers=ui.centers,
        skus=ui.skus,
        window_start=window_start,
        window_end=window_end,
        today=today,
        lag_days=lag_days,
        resource_name_map=name_map,
    )
    render_wip_progress(
        moves=df_move,
        centers=ui.centers,
        skus=ui.skus,
        window_start=window_start,
        window_end=window_end,
        today=today,
        resource_name_map=name_map,
    )

    st.divider()
    _ = render_inventory_pivot(
        snapshot=snapshot_df,
        centers=ui.centers,
        latest_snapshot=pd.to_datetime(latest_dt).normalize() if pd.notna(latest_dt) else today,
        resource_name_map=name_map,
        load_snapshot_raw_fn=load_snapshot_raw,
        snapshot_raw=snapshot_raw_df,
    )


if __name__ == "__main__":
    main()
