"""
SCM Dashboard v7 — Streamlit 엔트리 포인트

설계 요약(한글):
- v7은 v5 구현을 래핑(wrapping)하여 동일한 동작을 보장합니다.
- 본 파일은 UI(App) 계층으로서 사용자 입력 수집, 메시지/렌더링만 담당합니다.
- 도메인/피처/예측 로직은 각 레이어에서 DataFrame 입출력으로만 처리합니다.

주의:
- v7은 v6와 혼선을 피하기 위해 별도 엔트리/모듈 경로를 사용합니다.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from scm_dashboard_v7.ui.controls import collect_sidebar_controls
from scm_dashboard_v7.features.timeline import render_timeline_section
from scm_dashboard_v7.features.amazon import render_amazon_panel
from scm_dashboard_v7.features.inventory_view import (
    render_inventory_pivot,
    render_upcoming_inbound,
    render_wip_progress,
)
from scm_dashboard_v7.data.loaders import load_gsheet, load_snapshot_raw, load_excel
from scm_dashboard_v5.ui import render_sku_summary_cards
from scm_dashboard_v4.processing import (
    load_wip_from_incoming,
    merge_wip_as_moves,
    normalize_moves,
    normalize_refined_snapshot,
)
from center_alias import normalize_center_value


def main() -> None:
    """
    v7 대시보드 진입점

    - 데이터 로딩: GS 또는 업로드 엑셀 → 정규화
    - 선택값 수집: 사이드바 컨트롤
    - 섹션 렌더: KPI → 타임라인 → 아마존 → 입고/WIP → 인벤토리 피벗
    """

    st.set_page_config(page_title="SCM Dashboard v7", layout="wide")
    st.title("SCM Dashboard v7")
    st.caption("v7 구조 — v5 동작을 완전 재현(골든 테스트 기반)")

    st.markdown("### 데이터 소스")
    st.caption("GS 로드 실패 시 엑셀 업로드로 대체 가능합니다.")

    # 1) 기본: Google Sheets 시도
    df_move = pd.DataFrame()
    df_ref = pd.DataFrame()
    df_incoming = pd.DataFrame()
    snapshot_raw_df = None
    try:
        with st.spinner("Google Sheets 데이터 불러오는 중..."):
            g_mv, g_ref, g_in = load_gsheet()
            g_mv = normalize_moves(g_mv)
            required_snap = {"date", "center", "resource_code", "stock_qty"}
            if not g_ref.empty and not required_snap.issubset(set(map(str, g_ref.columns))):
                g_ref = normalize_refined_snapshot(g_ref)
            try:
                wip_df = load_wip_from_incoming(g_in)
                if wip_df is not None and not wip_df.empty:
                    g_mv = merge_wip_as_moves(g_mv, wip_df)
            except Exception:
                pass
            df_move, df_ref, df_incoming = g_mv, g_ref, g_in
            try:
                snapshot_raw_df = load_snapshot_raw()
            except Exception:
                snapshot_raw_df = None
        if not df_ref.empty:
            st.success("Google Sheets 데이터가 로드되었습니다.")
    except Exception:
        st.warning("Google Sheets 로드 실패. 아래 엑셀 업로드를 사용하세요.")

    # 2) (선택) 엑셀 업로드로 대체
    with st.expander("엑셀 파일 업로드 (선택)", expanded=False):
        up = st.file_uploader("엑셀 업로드 (.xlsx)", type=["xlsx"], key="v7_excel")
        if up is not None:
            try:
                df_move_x, df_ref_x, _df_incoming_x, snapshot_raw_x = load_excel(up)
                df_move_x = normalize_moves(df_move_x)
                required_snap = {"date", "center", "resource_code", "stock_qty"}
                if not df_ref_x.empty and not required_snap.issubset(set(map(str, df_ref_x.columns))):
                    df_ref_x = normalize_refined_snapshot(df_ref_x)
                try:
                    wip_x = load_wip_from_incoming(_df_incoming_x)
                    if wip_x is not None and not wip_x.empty:
                        df_move_x = merge_wip_as_moves(df_move_x, wip_x)
                except Exception:
                    pass
                df_move, df_ref = df_move_x.copy(), df_ref_x.copy()
                snapshot_raw_df = snapshot_raw_x if snapshot_raw_x is not None else snapshot_raw_df
                st.success("엑셀 데이터가 로드되었습니다.")
            except Exception as exc:
                st.error(f"엑셀 데이터 로딩 실패: {exc}")

    # 스냅샷 date 통일
    snapshot_df = df_ref.copy()
    if not snapshot_df.empty:
        if "date" in snapshot_df.columns:
            snapshot_df["date"] = pd.to_datetime(snapshot_df["date"], errors="coerce").dt.normalize()
        elif "snapshot_date" in snapshot_df.columns:
            snapshot_df["date"] = pd.to_datetime(snapshot_df["snapshot_date"], errors="coerce").dt.normalize()
        else:
            snapshot_df["date"] = pd.NaT

    # 필수 스냅샷 체크
    required_snap = {"date", "center", "resource_code", "stock_qty"}
    if snapshot_df.empty or not required_snap.issubset(set(map(str, snapshot_df.columns))):
        st.info("스냅샷 데이터가 없습니다. 엑셀을 업로드해 주세요.")
        return

    # 선택값 후보
    centers = sorted(snapshot_df.get("center", pd.Series([], dtype=str)).dropna().astype(str).unique().tolist())
    skus = sorted(snapshot_df.get("resource_code", pd.Series([], dtype=str)).dropna().astype(str).unique().tolist())
    if not centers or not skus:
        st.warning("센터 또는 SKU 정보를 찾을 수 없습니다.")
        return

    today = pd.Timestamp.today().normalize()
    latest_dt = snapshot_df["date"].dropna().max() if not snapshot_df.empty else pd.NaT
    bound_min = today - pd.Timedelta(days=42)
    bound_max = today + pd.Timedelta(days=60)

    # 컨트롤 수집
    ui = collect_sidebar_controls(
        centers=centers,
        skus=skus,
        bound_min=pd.Timestamp(bound_min).normalize(),
        bound_max=pd.Timestamp(bound_max).normalize(),
    )

    # KPI
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
        events=(
            [
                {
                    "start": pd.to_datetime(getattr(ui, "promotion_start", today)).normalize(),
                    "end": pd.to_datetime(getattr(ui, "promotion_end", today)).normalize(),
                    "uplift": float(getattr(ui, "promotion_percent", 0.0)) / 100.0,
                }
            ]
            if bool(getattr(ui, "promotion_enabled", False))
            else []
        ),
    )

    # 타임라인
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
        promotion_events=(
            [
                {
                    "start": pd.to_datetime(getattr(ui, "promotion_start", today)).normalize(),
                    "end": pd.to_datetime(getattr(ui, "promotion_end", today)).normalize(),
                    "uplift": float(getattr(ui, "promotion_percent", 0.0)) / 100.0,
                }
            ]
            if bool(getattr(ui, "promotion_enabled", False))
            else []
        ),
        show_production=ui.show_production,
        show_in_transit=ui.show_in_transit,
    )

    st.divider()
    st.subheader("Amazon US 일별 판매 vs. 재고")
    amazon_centers = [c for c in ui.centers if c.upper().startswith("AMZ") or "AMAZON" in c.upper()]
    amazon_centers = [normalize_center_value(c) for c in amazon_centers]
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
    try:
        if inv_actual_tidy is not None and not inv_actual_tidy.empty and "center" in inv_actual_tidy.columns:
            inv_actual_tidy = inv_actual_tidy.copy()
            inv_actual_tidy["center"] = inv_actual_tidy["center"].apply(normalize_center_value)
        if inv_forecast_tidy is not None and not inv_forecast_tidy.empty and "center" in inv_forecast_tidy.columns:
            inv_forecast_tidy = inv_forecast_tidy.copy()
            inv_forecast_tidy["center"] = inv_forecast_tidy["center"].apply(normalize_center_value)
    except Exception:
        pass

    snapshot_for_amazon = snapshot_df.copy()
    try:
        if "sales_qty" in snapshot_for_amazon.columns:
            snapshot_for_amazon["sales_qty"] = (
                pd.to_numeric(snapshot_for_amazon["sales_qty"], errors="coerce").fillna(0.0).clip(lower=0.0)
            )
            snapshot_for_amazon["date"] = pd.to_datetime(snapshot_for_amazon.get("date"), errors="coerce").dt.normalize()
            snapshot_for_amazon["center"] = snapshot_for_amazon.get("center", "").astype(str)
            snapshot_for_amazon["resource_code"] = snapshot_for_amazon.get("resource_code", "").astype(str)
    except Exception:
        pass

    render_amazon_panel(
        snapshot_long=snapshot_for_amazon,
        moves=df_move,
        snapshot_raw=snapshot_raw_df,
        centers=amazon_centers,
        skus=ui.skus,
        start=ui.start,
        end=ui.end,
        today=today,
        lookback_days=ui.lookback_days,
        promotion_events=(
            [
                {
                    "start": pd.to_datetime(getattr(ui, "promotion_start", today)).normalize(),
                    "end": pd.to_datetime(getattr(ui, "promotion_end", today)).normalize(),
                    "uplift": float(getattr(ui, "promotion_percent", 0.0)) / 100.0,
                }
            ]
            if bool(getattr(ui, "promotion_enabled", False))
            else []
        ),
        lag_days=int(getattr(ui, "inbound_lead_days", 7) or 7),
        inv_actual=inv_actual_tidy,
        inv_forecast=inv_forecast_tidy,
        use_inventory_for_sales=True,
        sales_forecast_from_inventory=None,
    )

    # 확정 입고 / WIP / 인벤토리 피벗
    st.divider()
    render_upcoming_inbound(
        moves=df_move,
        centers=ui.centers,
        skus=ui.skus,
        window_start=ui.start,
        window_end=ui.end,
        today=today,
        lag_days=int(getattr(ui, "inbound_lead_days", 7) or 7),
        resource_name_map={},
    )
    render_wip_progress(
        moves=df_move,
        centers=ui.centers,
        skus=ui.skus,
        window_start=ui.start,
        window_end=ui.end,
        today=today,
        resource_name_map={},
    )

    st.divider()
    _ = render_inventory_pivot(
        snapshot=snapshot_df,
        centers=ui.centers,
        latest_snapshot=pd.to_datetime(latest_dt).normalize() if pd.notna(latest_dt) else today,
        resource_name_map={},
        load_snapshot_raw_fn=load_snapshot_raw,
        snapshot_raw=snapshot_raw_df,
    )


if __name__ == "__main__":
    main()


