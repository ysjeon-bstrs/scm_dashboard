"""태광KR 가상창고 배분 대시보드 렌더러."""

from __future__ import annotations

from typing import Mapping, Sequence

import pandas as pd
import streamlit as st

from .amazon_snapshot import _format_int, _inject_card_styles


def render_taekwang_stock_dashboard(
    tk_stock_df: pd.DataFrame | None,
    *,
    selected_skus: Sequence[str],
    resource_name_map: Mapping[str, str] | None = None,
    sku_colors: Mapping[str, str] | None = None,
) -> None:
    """태광KR 가상창고 배분 현황을 Amazon 카드 UI로 렌더링합니다."""

    # 가상창고 배분 데이터가 없으면 사용자에게 안내 메시지를 출력
    if tk_stock_df is None or tk_stock_df.empty:
        st.info("태광KR 가상창고 배분 데이터를 찾을 수 없습니다.")
        return

    # 선택된 SKU가 없으면 계산을 진행할 수 없으므로 종료
    if not selected_skus:
        st.info("표시할 SKU를 선택해주세요.")
        return

    # SKU 필터링을 위해 제품코드를 문자열로 정리하고 선택된 SKU만 남김
    working_df = tk_stock_df.copy()
    working_df["product_code"] = working_df["product_code"].astype(str).str.strip()
    working_df = working_df[working_df["product_code"].isin(selected_skus)]

    # 필터 결과가 없으면 안내 메시지를 출력
    if working_df.empty:
        st.info("선택된 SKU에 대한 태광KR 가상창고 배분 데이터가 없습니다.")
        return

    # 가상창고 구조는 운영창고/키핑창고로 나뉘며, 각 수량 컬럼을 합산해야 하므로 안전하게 숫자로 변환
    quantity_cols = [
        "global_b2b_running",
        "global_b2b_keeping",
        "global_b2c_running",
        "global_b2c_keeping",
    ]
    for col in quantity_cols:
        working_df[col] = pd.to_numeric(working_df[col], errors="coerce").fillna(0)

    # 최신 스냅샷 시각을 표시하기 위해 Timestamp로 변환
    working_df["snap_time"] = pd.to_datetime(working_df["snap_time"], errors="coerce")
    latest_snap = working_df["snap_time"].dropna().max()

    # 로트 단위 데이터를 SKU 단위로 묶기 위해 합계를 계산
    aggregated = (
        working_df.groupby("product_code", dropna=False)[quantity_cols]
        .sum()
        .reset_index()
    )

    # 운영창고와 키핑창고 재고를 모두 합쳐 총 재고를 계산
    aggregated["total_qty"] = aggregated[quantity_cols].sum(axis=1)

    # Amazon KPI 카드 스타일을 재사용하여 UI 일관성 확보
    _inject_card_styles()

    # SKU-색상 매핑이 없으면 빈 dict를 사용
    color_map = dict(sku_colors or {})

    # 카드 HTML을 순차적으로 생성
    cards_html: list[str] = []
    for row in aggregated.itertuples(index=False):
        sku = str(row.product_code)
        color = color_map.get(sku, "#4E79A7")

        # 품명을 표시하기 위해 사전 매핑을 조회
        resource_name = ""
        if resource_name_map is not None:
            resource_name = str(resource_name_map.get(sku, "")).strip()

        # 헤더는 품명과 SKU를 함께 노출
        if resource_name:
            header_html = (
                f"<h4><span class='color-dot' style='background-color:{color}'></span>"
                f"{resource_name} <span style='color: #666; font-size: 0.9em;'>[{sku}]</span></h4>"
            )
        else:
            header_html = (
                f"<h4><span class='color-dot' style='background-color:{color}'></span>"
                f"{sku}</h4>"
            )

        # 가상창고 내 운영/키핑 창고별 수량을 KPI 카드로 구성
        metrics = [
            (
                "총 재고",
                _format_int(row.total_qty),
                "태광KR 가상창고에 배정된 전체 수량 (운영+키핑 포함)",
            ),
            (
                "글로벌B2B 운영창고",
                _format_int(row.global_b2b_running),
                "글로벌 B2B 부서가 운영창고에 보유 중인 배정 재고",
            ),
            (
                "글로벌B2B 키핑창고",
                _format_int(row.global_b2b_keeping),
                "글로벌 B2B 주문/출고 예약분(키핑창고) 수량",
            ),
            (
                "글로벌B2C 운영창고",
                _format_int(row.global_b2c_running),
                "글로벌 B2C 부서 운영창고 배정 재고",
            ),
            (
                "글로벌B2C 키핑창고",
                _format_int(row.global_b2c_keeping),
                "글로벌 B2C 주문·발송 예약분(키핑창고) 수량",
            ),
        ]

        metric_html_parts: list[str] = []
        for label, value, tooltip in metrics:
            title_attr = f" title='{tooltip}'" if tooltip else ""
            metric_html_parts.append(
                f"<div class='amz-kpi-metric'{title_attr}><span class='label'>{label}</span>"
                f"<span class='value'>{value}</span></div>"
            )

        card_html = (
            "<div class='amz-kpi-card'>"
            + header_html
            + "<div class='amz-kpi-grid'>"
            + "".join(metric_html_parts)
            + "</div></div>"
        )
        cards_html.append(card_html)

    st.subheader("태광KR 대시보드")
    st.markdown(
        "<div class='amz-kpi-container'>" + "".join(cards_html) + "</div>",
        unsafe_allow_html=True,
    )

    if pd.notna(latest_snap):
        st.caption(f"{latest_snap:%Y-%m-%d %H:%M} 기준")
    else:
        st.caption("스냅샷 시각 정보를 확인할 수 없습니다.")
