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
    inbound_moves: pd.DataFrame | None = None,
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

    # 입고예정 정보를 표시하기 위해 WIP(입고예정내역) 데이터를 미리 가공
    inbound_summary: dict[str, dict[str, tuple[int, pd.Timestamp]]] = {}
    if inbound_moves is not None and not inbound_moves.empty:
        # 입고예정 정보는 생산 진행(WIP) 물량으로 들어오므로, WIP 레코드만 필터링
        inbound_working = inbound_moves.copy()
        resource_series = inbound_working.get("resource_code")
        if resource_series is None:
            inbound_working["resource_code"] = ""
        else:
            inbound_working["resource_code"] = resource_series.astype(str).str.strip()
        inbound_working = inbound_working[
            inbound_working["resource_code"].isin(selected_skus)
        ]
        inbound_working = inbound_working[inbound_working.get("carrier_mode") == "WIP"]

        if not inbound_working.empty:
            # 입고 예정일은 event_date(=wip_ready)를 최우선으로 사용하고, 없으면 arrival_date를 사용
            event_dates = pd.to_datetime(
                inbound_working.get("event_date"), errors="coerce"
            )
            arrival_dates = pd.to_datetime(
                inbound_working.get("arrival_date"), errors="coerce"
            )
            inbound_working["_inbound_date"] = event_dates.fillna(
                arrival_dates
            ).dt.normalize()

            # 최근 날짜 기준 비교를 위해 오늘 날짜(현지 시간 기준)를 확보
            today_norm = pd.Timestamp.now(tz=None).normalize()

            # 채널별(글로벌B2B/글로벌B2C)로 가장 가까운 입고예정 수량과 일자를 추출
            for sku, sku_group in inbound_working.groupby("resource_code"):
                channel_info: dict[str, tuple[int, pd.Timestamp]] = {}

                for channel_col in ["global_b2b", "global_b2c"]:
                    if channel_col not in sku_group.columns:
                        continue

                    # 채널별 배정 수량을 숫자로 변환하여 양수(실제 입고 예정 물량)만 남김
                    channel_qty = pd.to_numeric(
                        sku_group[channel_col], errors="coerce"
                    ).fillna(0)
                    positive_mask = channel_qty > 0
                    if not positive_mask.any():
                        continue

                    channel_df = sku_group.loc[positive_mask].copy()
                    channel_df["_channel_qty"] = channel_qty[positive_mask].astype(int)
                    channel_df["_inbound_date"] = pd.to_datetime(
                        channel_df["_inbound_date"], errors="coerce"
                    )
                    channel_df = channel_df.dropna(subset=["_inbound_date"])
                    if channel_df.empty:
                        continue

                    # 오늘 이후 일정이 있으면 그중 가장 빠른 날짜, 없으면 전체 중 가장 빠른 날짜
                    future_df = channel_df[channel_df["_inbound_date"] >= today_norm]
                    target_df = future_df if not future_df.empty else channel_df
                    target_row = target_df.sort_values("_inbound_date").iloc[0]

                    channel_info[channel_col] = (
                        int(target_row["_channel_qty"]),
                        target_row["_inbound_date"],
                    )

                if channel_info:
                    inbound_summary[str(sku)] = channel_info

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

        # 가상창고 내 운영/입고예정 창고 구분을 명확히 설명하며 KPI 카드 구성
        b2b_inbound = inbound_summary.get(sku, {}).get("global_b2b")
        b2c_inbound = inbound_summary.get(sku, {}).get("global_b2c")

        def _format_inbound(value: tuple[int, pd.Timestamp] | None) -> str:
            """입고예정 수량과 일자를 KPI 표기 형식으로 변환합니다."""

            # 입고예정 물량이 없으면 미정으로 표시
            if value is None:
                return "+0 (미정)"

            qty, due_date = value
            if pd.isna(due_date):
                return f"+{_format_int(qty)} (미정)"
            return f"+{_format_int(qty)} ({due_date:%m/%d})"

        metrics = [
            (
                "글로벌B2B 운영창고",
                _format_int(row.global_b2b_running),
                "글로벌 B2B 부서가 운영창고에 보유 중인 배정 재고",
            ),
            (
                "글로벌B2C 운영창고",
                _format_int(row.global_b2c_running),
                "글로벌 B2C 부서 운영창고 배정 재고",
            ),
            (
                "글로벌B2B 입고예정",
                _format_inbound(b2b_inbound),
                "입고예정내역에서 글로벌 B2B용으로 배정된 가상 입고(운영창고 유입 예정)",
            ),
            (
                "글로벌B2C 입고예정",
                _format_inbound(b2c_inbound),
                "입고예정내역에서 글로벌 B2C용으로 배정된 가상 입고(운영창고 유입 예정)",
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

    st.caption("한 시간 단위로 업데이트됩니다")
