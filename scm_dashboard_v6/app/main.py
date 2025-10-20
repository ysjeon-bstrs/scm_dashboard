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
from scm_dashboard_v5.forecast.consumption import forecast_sales_and_inventory
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
    # 디버깅 배너 제거

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
    # 프로모션 설정 (전역)
    promo_enabled = bool(getattr(ui, "promotion_enabled", False))
    promo_events = (
        [
            {
                "start": pd.to_datetime(getattr(ui, "promotion_start", today)).normalize(),
                "end": pd.to_datetime(getattr(ui, "promotion_end", today)).normalize(),
                "uplift": float(getattr(ui, "promotion_percent", 0.0)) / 100.0,
            }
        ]
        if promo_enabled
        else None
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
        events=(promo_events or []),
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
        promotion_events=promo_events or [],
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
    # v6 전용 가드: v5 렌더러의 '첫 품절일 이후 전체 0' 클램프를 회피하기 위해
    # 미래 재고가 0으로 떨어지는 날에도 극소 양수로 유지하여(시각적으로는 0)
    # 입고 이후 판매가 재개될 수 있도록 한다.
    try:
        if inv_forecast_tidy is not None and not inv_forecast_tidy.empty:
            inv_forecast_tidy = inv_forecast_tidy.copy()
            inv_forecast_tidy["stock_qty"] = pd.to_numeric(inv_forecast_tidy["stock_qty"], errors="coerce").fillna(0.0)
            inv_forecast_tidy.loc[:, "stock_qty"] = inv_forecast_tidy["stock_qty"].clip(lower=1e-6)
    except Exception:
        pass

    # 안전장치: 타임라인에 입고 반영이 누락된 경우, moves 기반 입고량을 미래 재고에 가산
    try:
        if inv_forecast_tidy is not None and not inv_forecast_tidy.empty and not df_move.empty:
            mv = df_move.copy()
            mv["to_center"] = mv.get("to_center", "").astype(str)
            mv["resource_code"] = mv.get("resource_code", "").astype(str)
            mv["qty_ea"] = pd.to_numeric(mv.get("qty_ea"), errors="coerce").fillna(0)
            # 입고일 coalesce: event_date → pred_inbound_date → inbound_date → arrival_date(+lag if past)
            cols = {str(c).strip().lower(): c for c in mv.columns}
            col_event = cols.get("event_date")
            col_pred = cols.get("pred_inbound_date")
            col_inb = cols.get("inbound_date")
            col_arr = cols.get("arrival_date")
            eta = pd.Series(pd.NaT, index=mv.index, dtype="datetime64[ns]")
            if col_event and col_event in mv.columns:
                eta = eta.fillna(pd.to_datetime(mv[col_event], errors="coerce"))
            if col_pred and col_pred in mv.columns:
                eta = eta.fillna(pd.to_datetime(mv[col_pred], errors="coerce"))
            if col_inb and col_inb in mv.columns:
                eta = eta.fillna(pd.to_datetime(mv[col_inb], errors="coerce"))
            if col_arr and col_arr in mv.columns:
                arr = pd.to_datetime(mv[col_arr], errors="coerce")
                lag = int(getattr(ui, "inbound_lead_days", 7) or 7)
                past_mask = arr.notna() & (arr <= today)
                fut_mask = arr.notna() & (arr > today)
                eta = eta.where(eta.notna(), arr)
                if past_mask.any():
                    eta.loc[past_mask] = (arr.loc[past_mask] + pd.Timedelta(days=lag)).dt.normalize()
                if fut_mask.any():
                    eta.loc[fut_mask] = arr.loc[fut_mask].dt.normalize()
            mv["_eta"] = pd.to_datetime(eta, errors="coerce").dt.normalize()

            inbound = mv[
                (mv["to_center"].isin(amazon_centers))
                & (mv["_eta"].notna())
                & (mv["_eta"] > today)
                & (mv["resource_code"].isin(ui.skus))
            ][["_eta", "resource_code", "qty_ea"]].rename(columns={"_eta": "date"})
            if not inbound.empty:
                inbound = inbound.groupby(["date", "resource_code"], as_index=False)["qty_ea"].sum()
                base = inv_forecast_tidy.copy()
                base["date"] = pd.to_datetime(base["date"], errors="coerce").dt.normalize()
                base["resource_code"] = base.get("resource_code", "").astype(str)
                base["stock_qty"] = pd.to_numeric(base.get("stock_qty"), errors="coerce").fillna(0.0)
                merged = base.merge(inbound, on=["date", "resource_code"], how="left")
                merged["qty_ea"] = pd.to_numeric(merged.get("qty_ea"), errors="coerce").fillna(0.0)
                merged["stock_qty"] = (merged["stock_qty"] + merged["qty_ea"]).astype(float)
                inv_forecast_tidy = merged.drop(columns=["qty_ea"], errors="ignore")
    except Exception:
        pass

    # 프로모션 여부와 무관하게 인벤토리 추세는 스텝 타임라인 값을 사용해 일관성 유지
    inv_actual_param = inv_actual_tidy
    inv_forecast_param = inv_forecast_tidy

    # v6: 판매 이력 우선순위 — snap_정제.sales_qty 우선, 없을 때만 snapshot_raw의 FBA 출고로 보강
    snapshot_for_amazon = snapshot_df.copy()
    try:
        # 1) snap_정제에 sales_qty가 있으면 그 값을 우선 사용 (형변환만 보정)
        if "sales_qty" in snapshot_for_amazon.columns:
            snapshot_for_amazon["sales_qty"] = (
                pd.to_numeric(snapshot_for_amazon["sales_qty"], errors="coerce").fillna(0.0).clip(lower=0.0)
            )
            snapshot_for_amazon["date"] = pd.to_datetime(snapshot_for_amazon.get("date"), errors="coerce").dt.normalize()
            snapshot_for_amazon["center"] = snapshot_for_amazon.get("center", "").astype(str)
            snapshot_for_amazon["resource_code"] = snapshot_for_amazon.get("resource_code", "").astype(str)
        else:
            # 2) 없으면 snapshot_raw의 fba_output_stock를 일자×SKU로 집계하여 주입
            if snapshot_raw_df is not None and not snapshot_raw_df.empty:
                raw = snapshot_raw_df.copy()
                cols = {str(c).strip().lower(): c for c in raw.columns}
                col_date = cols.get("snapshot_date") or cols.get("date")
                col_sku = cols.get("resource_code") or cols.get("sku") or cols.get("상품코드")
                # fba_output_stock (아마존 출고)를 판매로 사용
                col_out = None
                for name in raw.columns:
                    lname = str(name).strip().lower()
                    if "fba_output_stock" in lname or lname in {"fba출고","출고","출고수량","출고 ea","fba_output"}:
                        col_out = name
                        break
                if col_date and col_sku and col_out:
                    sales = raw[[col_date, col_sku, col_out]].copy()
                    sales[col_date] = pd.to_datetime(sales[col_date], errors="coerce").dt.normalize()
                    sales[col_sku] = sales[col_sku].astype(str)
                    sales[col_out] = pd.to_numeric(sales[col_out], errors="coerce").fillna(0.0).clip(lower=0.0)
                    # 최신 스냅샷 구간만 집계 (불필요한 과거는 제외)
                    if pd.notna(latest_dt):
                        sales = sales[sales[col_date] <= pd.to_datetime(latest_dt).normalize()]
                    sales_tidy = (
                        sales.groupby([col_date, col_sku], as_index=False)[col_out].sum()
                        .rename(columns={col_date: "date", col_sku: "resource_code", col_out: "sales_qty"})
                    )
                    # 타겟 아마존 센터에 주입 (센터 별도로 구분이 없다면 첫 AMZ 센터에 귀속)
                    target_center = (amazon_centers[0] if amazon_centers else "AMZUS")
                    sales_tidy["center"] = target_center
                    base = snapshot_for_amazon.copy()
                    base["date"] = pd.to_datetime(base.get("date"), errors="coerce").dt.normalize()
                    base["center"] = base.get("center", "").astype(str)
                    base["resource_code"] = base.get("resource_code", "").astype(str)
                    snapshot_for_amazon = base.merge(
                        sales_tidy, on=["date","center","resource_code"], how="left"
                    )
    except Exception:
        pass

    # v5 소비예측 함수로 판매 예측 생성: lookback_days와 프로모션 반영 → 차트에 강제 주입
    sales_from_inv: pd.DataFrame | None = None
    try:
        # 아마존 센터만 모은 타임라인 (과거+미래)
        timeline_amz = (
            timeline[timeline["center"].isin(amazon_centers)]
            if isinstance(timeline, pd.DataFrame) and not timeline.empty
            else pd.DataFrame(columns=["date","center","resource_code","stock_qty"])
        )
        # 일별 판매 이력: snap_정제.sales_qty 우선 → snapshot_raw FBA 출고 대체
        daily_sales = pd.DataFrame(columns=["date","center","resource_code","sales_ea"])
        if not snapshot_df.empty and "sales_qty" in snapshot_df.columns:
            ds = snapshot_df[["date","center","resource_code","sales_qty"]].copy()
            ds["date"] = pd.to_datetime(ds["date"], errors="coerce").dt.normalize()
            ds["center"] = ds.get("center", "").astype(str)
            ds["resource_code"] = ds.get("resource_code", "").astype(str)
            ds["sales_ea"] = pd.to_numeric(ds.get("sales_qty"), errors="coerce").fillna(0.0)
            daily_sales = ds[["date","center","resource_code","sales_ea"]]
        elif snapshot_raw_df is not None and not snapshot_raw_df.empty:
            raw = snapshot_raw_df.copy()
            cols = {str(c).strip().lower(): c for c in raw.columns}
            col_date = cols.get("snapshot_date") or cols.get("date")
            col_sku = cols.get("resource_code") or cols.get("sku") or cols.get("상품코드")
            col_out = None
            for name in raw.columns:
                lname = str(name).strip().lower()
                if "fba_output_stock" in lname or lname in {"fba출고","출고","출고수량","출고 ea","fba_output"}:
                    col_out = name
                    break
            if col_date and col_sku and col_out:
                sales = raw[[col_date, col_sku, col_out]].copy()
                sales[col_date] = pd.to_datetime(sales[col_date], errors="coerce").dt.normalize()
                sales[col_sku] = sales[col_sku].astype(str)
                sales[col_out] = pd.to_numeric(sales[col_out], errors="coerce").fillna(0.0).clip(lower=0.0)
                if pd.notna(latest_dt):
                    sales = sales[sales[col_date] <= pd.to_datetime(latest_dt).normalize()]
                grouped = (
                    sales.groupby([col_date, col_sku], as_index=False)[col_out].sum()
                    .rename(columns={col_date: "date", col_sku: "resource_code", col_out: "sales_ea"})
                )
                target_center = (amazon_centers[0] if amazon_centers else "AMZUS")
                grouped["center"] = target_center
                grouped["date"] = pd.to_datetime(grouped["date"], errors="coerce").dt.normalize()
                daily_sales = grouped[["date","center","resource_code","sales_ea"]]

        if not timeline_amz.empty:
            sales_fc, _inv_proj = forecast_sales_and_inventory(
                daily_sales=daily_sales,
                timeline_center=timeline_amz,
                start=ui.start,
                end=ui.end,
                lookback_days=int(ui.lookback_days or 28),
                uplift_events=(promo_events or []),
            )
            if sales_fc is not None and not sales_fc.empty:
                forced = sales_fc.rename(columns={"sales_ea":"sales_qty"})
                forced = forced[(forced["date"] > today) & (forced["center"].isin(amazon_centers))]
                # v5 차트 필터를 통과하려면 center 컬럼이 필요
                sales_from_inv = forced[["date","center","resource_code","sales_qty"]].copy()
                sales_from_inv["date"] = pd.to_datetime(sales_from_inv["date"], errors="coerce").dt.normalize()
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
        promotion_events=promo_events,
        # 판매 예측은 v5 소비예측(추세+프로모션)을 사용
        use_consumption_forecast=True,
        inv_actual=inv_actual_param,
        inv_forecast=inv_forecast_param,
        # 인벤토리 기반 판매 유도는 끈다(추세/프로모션 반영 우선)
        use_inventory_for_sales=False,
        sales_forecast_from_inventory=sales_from_inv,
    )

    # 디버그 마커 제거됨

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
