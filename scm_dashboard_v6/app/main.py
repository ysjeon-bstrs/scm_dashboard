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
from scm_dashboard_v6.features.inventory_view import render_inventory_pivot
from scm_dashboard_v6.data.loaders import load_gsheet, load_snapshot_raw, load_excel


def main() -> None:
    st.set_page_config(page_title="SCM Dashboard v6", layout="wide")
    st.title("SCM Dashboard v6")
    st.caption("v6 구조 도입 — v5 동작을 유지하면서 경계/모듈 분리")

    st.markdown("### 데이터 소스")
    st.caption("초기 단계에서는 v5 로더 위임 — 후속 단계에서 v6 data로 전환")
    try:
        with st.spinner("Google Sheets 데이터 불러오는 중..."):
            df_move, df_ref, df_incoming = load_gsheet()
    except Exception as exc:
        st.error(f"데이터 로딩 실패: {exc}")
        return

    # 스냅샷 정규화: date 컬럼 통일
    snapshot_df = df_ref.copy()
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
                df_move_x, df_ref_x, _df_incoming_x, _ = load_excel(up)
                df_move = df_move_x.copy()
                snapshot_df = df_ref_x.copy()
                if "date" in snapshot_df.columns:
                    snapshot_df["date"] = pd.to_datetime(snapshot_df["date"], errors="coerce").dt.normalize()
                elif "snapshot_date" in snapshot_df.columns:
                    snapshot_df["date"] = pd.to_datetime(snapshot_df["snapshot_date"], errors="coerce").dt.normalize()
                else:
                    snapshot_df["date"] = pd.NaT
                st.success("엑셀 데이터가 로드되었습니다.")
            except Exception as exc:
                st.error(f"엑셀 데이터 로딩 실패: {exc}")

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
        show_production=ui.show_production,
        show_in_transit=ui.show_in_transit,
    )

    st.divider()
    st.subheader("Amazon US 일별 판매 vs. 재고")
    snapshot_raw_df = load_snapshot_raw()
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
        promotion_events=None,
        use_consumption_forecast=True,
        inv_actual=inv_actual_tidy,
        inv_forecast=inv_forecast_tidy,
    )

    st.divider()
    st.subheader("선택 센터 현재 재고 (최신 스냅샷)")
    # 간단 품명 매핑 (있으면)
    name_map: dict[str, str] = {}
    if "resource_name" in snapshot_df.columns:
        rows = snapshot_df.loc[snapshot_df["resource_name"].notna(), ["resource_code", "resource_name"]].copy()
        if not rows.empty:
            name_map = rows.drop_duplicates("resource_code").set_index("resource_code")["resource_name"].to_dict()

    _ = render_inventory_pivot(
        snapshot=snapshot_df,
        centers=ui.centers,
        latest_snapshot=pd.to_datetime(latest_dt).normalize() if pd.notna(latest_dt) else today,
        resource_name_map=name_map,
    )


if __name__ == "__main__":
    main()
