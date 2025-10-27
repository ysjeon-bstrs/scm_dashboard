"""KPI 카드 렌더링 모듈.

SKU 요약 카드, 센터별 메트릭 카드 등을 렌더링합니다.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence

import pandas as pd
import streamlit as st

from ...analytics.kpi import kpi_breakdown_per_sku
from ...core.config import CONFIG
from .cards_helpers import (
    aggregate_metrics,
    calculate_wip_pipeline,
    prepare_moves_data,
    validate_and_prepare_snapshot,
)
from .formatters import (
    calculate_coverage_days,
    calculate_sellout_date,
    escape,
    format_date,
    format_days,
    format_number,
    should_show_in_transit,
    value_font_size,
)
from .metrics import (
    compute_depletion_metrics,
    extract_daily_demand,
    movement_breakdown_per_center,
)
from .styles import inject_responsive_styles


def build_metric_card(label: str, value: str, *, compact: bool = False) -> str:
    classes = ["kpi-metric-card"]
    if compact:
        classes.append("kpi-metric-card--compact")
    value_text = "-" if value is None else str(value)
    base_font = 1.55 if not compact else 1.3
    min_font = 1.05 if not compact else 0.85
    font_size = value_font_size(value_text, base_size=base_font, min_size=min_font)
    return (
        f'<div class="{" ".join(classes)}">'
        f'<div class="kpi-metric-label">{escape(label)}</div>'
        f'<div class="kpi-metric-value" style="font-size:{font_size}; white-space:nowrap;">{escape(value_text)}</div>'
        "</div>"
    )


def build_grid(
    items: Sequence[str],
    *,
    min_width: int | None = None,
    extra_class: str = "",
    columns: int | None = None,
    data_attrs: Mapping[str, object] | None = None,
) -> str:
    if not items:
        return ""
    classes = ["kpi-card-grid"]
    if extra_class:
        classes.append(extra_class)
    style_parts: list[str] = []
    if min_width is not None:
        style_parts.append(f"--min-card-width: {int(min_width)}px;")
    if columns is not None and columns > 0:
        style_parts.append(
            "grid-template-columns: repeat("
            f"{int(columns)}, minmax(var(--min-card-width, 280px), 1fr));"
        )
    style_value = " ".join(style_parts)
    style = f' style="{style_value}"' if style_value else ""

    attr_parts: list[str] = []
    if data_attrs:
        for key, value in data_attrs.items():
            if value is None:
                continue
            attr_parts.append(f'{escape(key)}="{escape(value)}"')
    attrs = (" " + " ".join(attr_parts)) if attr_parts else ""

    return (
        f'<div class="{" ".join(classes)}"{attrs}{style}>' + "".join(items) + "</div>"
    )


def center_grid_layout(count: int) -> tuple[int | None, int, str]:
    """Return (columns, min_width, modifier_class) for the center KPI grid."""

    if count <= 2:
        return None, 320, "kpi-grid--centers-narrow"
    if count <= 4:
        return None, 280, "kpi-grid--centers-medium"
    if count <= 6:
        return None, 250, "kpi-grid--centers-wide"
    return None, 220, "kpi-grid--centers-dense"


def build_center_card(center_info: Mapping[str, object]) -> str:
    metric_cards = [
        build_metric_card("재고", format_number(center_info["current"]), compact=True),
        build_metric_card(
            "이동중",
            (
                format_number(center_info["in_transit"])
                if center_info["show_in_transit"]
                else "-"
            ),
            compact=True,
        ),
        build_metric_card(
            "생산중(30일 내 완료)", format_number(center_info["wip"]), compact=True
        ),
        build_metric_card(
            "예상 소진일수", format_days(center_info["coverage"]), compact=True
        ),
        build_metric_card(
            "소진 예상일", format_date(center_info["sellout_date"]), compact=True
        ),
    ]
    metrics_html = build_grid(
        metric_cards,
        extra_class="kpi-grid--compact kpi-grid--center-metrics",
        min_width=140,
    )
    return (
        '<div class="kpi-center-card">'
        f'<div class="kpi-center-title">{escape(center_info["center"])}</div>'
        f"{metrics_html}"
        "</div>"
    )


def render_sku_summary_cards(
    snapshot: pd.DataFrame,
    moves: pd.DataFrame,
    *,
    centers: Sequence[str],
    skus: Sequence[str],
    today: pd.Timestamp,
    date_column: str = "date",
    latest_snapshot: pd.Timestamp | None = None,
    lag_days: int = 7,
    chunk_size: int = 2,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    lookback_days: int | None = None,
    horizon_pad_days: int = 60,
    events: Optional[Sequence[Dict[str, object]]] | None = None,
) -> pd.DataFrame:
    """Render SKU summary KPI cards and return the underlying DataFrame."""

    if snapshot is None or snapshot.empty:
        st.caption("스냅샷 데이터가 없어 KPI를 계산할 수 없습니다.")
        return pd.DataFrame()

    (
        snapshot_view,
        filtered_snapshot,
        centers_list,
        sku_list,
        centers_all,
        latest_snapshot_dt,
        global_latest_snapshot_dt,
        name_map,
    ) = validate_and_prepare_snapshot(
        snapshot, centers, skus, date_column, latest_snapshot
    )

    if snapshot_view.empty:
        st.caption("스냅샷 데이터 검증 실패: 유효한 데이터가 없습니다.")
        return pd.DataFrame()
    if filtered_snapshot.empty:
        st.caption("선택한 센터/SKU 조합에 해당하는 KPI 데이터가 없습니다.")
        return pd.DataFrame()
    if not centers_list or not sku_list:
        st.caption("센터 또는 SKU 선택이 비어 있어 KPI를 계산할 수 없습니다.")
        return pd.DataFrame()
    if pd.isna(latest_snapshot_dt) or pd.isna(global_latest_snapshot_dt):
        st.caption("최신 스냅샷 일자를 확인할 수 없어 KPI를 계산할 수 없습니다.")
        return pd.DataFrame()

    latest_snapshot = latest_snapshot_dt

    moves_view, moves_global = prepare_moves_data(moves, centers_list, sku_list)

    kpi_df = kpi_breakdown_per_sku(
        filtered_snapshot,
        moves_view,
        centers_list,
        sku_list,
        pd.to_datetime(today).normalize(),
        "date",
        pd.to_datetime(latest_snapshot).normalize(),
        int(lag_days),
    )

    if kpi_df.empty:
        st.caption("※ KPI 계산 결과가 없습니다.")
        return kpi_df

    kpi_df.index = kpi_df.index.astype(str)

    latest_snapshot_dt = pd.to_datetime(latest_snapshot).normalize()
    today_dt = pd.to_datetime(today).normalize()

    wip_pipeline_totals, wip_30d_by_center = calculate_wip_pipeline(
        moves, sku_list, centers_list, today_dt
    )

    start_dt = (
        pd.to_datetime(start).normalize()
        if start is not None
        else pd.to_datetime(filtered_snapshot["date"].min()).normalize()
    )
    end_dt = (
        pd.to_datetime(end).normalize()
        if end is not None
        else pd.to_datetime(filtered_snapshot["date"].max()).normalize()
    )
    lookback_val = int(lookback_days) if lookback_days is not None else int(lag_days)

    depletion_df = compute_depletion_metrics(
        snap_long=filtered_snapshot,
        moves=moves_view,
        centers=centers_list,
        skus=sku_list,
        today=today_dt,
        start=start_dt,
        end=end_dt,
        lookback_days=lookback_val,
        horizon_pad_days=int(horizon_pad_days),
        events=list(events) if events else None,
    )

    depletion_df["center"] = depletion_df.get("center").astype(str)
    center_depletion_map: Dict[tuple[str, str], Dict[str, object]] = {}
    if not depletion_df.empty:
        for row in depletion_df.to_dict("records"):
            center_name = str(row.get("center"))
            sku_code = str(row.get("resource_code"))
            if center_name != "__TOTAL__":
                center_depletion_map[(sku_code, center_name)] = row

    metrics = aggregate_metrics(
        filtered_snapshot,
        snapshot_view,
        latest_snapshot_dt,
        global_latest_snapshot_dt,
        moves_view,
        moves_global,
        centers_list,
        centers_all,
        sku_list,
        today_dt,
        int(lag_days),
    )

    current_by_center = metrics.current_by_center
    current_totals = metrics.current_totals
    global_current_totals = metrics.global_current_totals
    daily_demand_series = metrics.daily_demand_series
    in_transit_series = metrics.in_transit_series
    global_in_transit_totals = metrics.global_in_transit_totals

    inject_responsive_styles()

    sku_cards_html: list[str] = []
    # chunk_size 기본값을 CONFIG에서 가져옴 (None만 대체, 0은 유지)
    if chunk_size is None:
        chunk_size = CONFIG.ui.kpi_card_chunk_size
    sku_min_width = max(280, int(1024 / max(chunk_size, 1))) if chunk_size else 320

    for sku in sku_list:
        display_name = name_map.get(sku, "") if isinstance(name_map, Mapping) else ""

        base_current = kpi_df.at[sku, "current"] if sku in kpi_df.index else 0
        current_val = current_totals.get(sku, base_current)
        total_current = int(current_val) if pd.notna(current_val) else int(base_current)
        if not global_current_totals.empty:
            current_all_val = global_current_totals.get(sku, float("nan"))
            total_current_all = (
                int(round(current_all_val))
                if pd.notna(current_all_val)
                else total_current
            )
        else:
            total_current_all = total_current
        total_transit = int(kpi_df.at[sku, "in_transit"]) if sku in kpi_df.index else 0
        if not global_in_transit_totals.empty:
            transit_all_val = global_in_transit_totals.get(sku, float("nan"))
            total_transit_all = (
                int(round(transit_all_val))
                if pd.notna(transit_all_val)
                else total_transit
            )
        else:
            total_transit_all = total_transit

        wip_pipeline_value = int(wip_pipeline_totals.get(sku, 0))

        summary_cards = [
            build_metric_card("전체 센터 재고 합계", format_number(total_current_all)),
            build_metric_card("선택 센터 재고 합계", format_number(total_current)),
            build_metric_card(
                "전체 이동중 재고 합계", format_number(total_transit_all)
            ),
            build_metric_card(
                "전체 생산 예정 재고 합계",
                format_number(wip_pipeline_value),
            ),
        ]
        summary_html = build_grid(
            summary_cards,
            extra_class="kpi-grid--summary",
            min_width=220,
        )

        center_cards: list[str] = []
        for center in centers_list:
            center_current = (
                int(current_by_center.get((sku, center), 0))
                if not current_by_center.empty
                else 0
            )
            center_transit = (
                int(in_transit_series.get((sku, center), 0))
                if not in_transit_series.empty
                else 0
            )
            center_wip_30d = int(wip_30d_by_center.get((sku, center), 0))
            center_demand = float(daily_demand_series.get((sku, center), float("nan")))
            center_depletion = center_depletion_map.get((sku, center), {})
            center_coverage = center_depletion.get("days_to_depletion")
            center_sellout_date = center_depletion.get("depletion_date")

            if center_coverage is None:
                center_coverage = calculate_coverage_days(center_current, center_demand)
                center_sellout_date = calculate_sellout_date(today_dt, center_coverage)
            center_cards.append(
                build_center_card(
                    {
                        "center": center,
                        "current": center_current,
                        "in_transit": center_transit,
                        "wip": center_wip_30d,
                        "coverage": center_coverage,
                        "sellout_date": center_sellout_date,
                        "show_in_transit": should_show_in_transit(
                            center, center_transit
                        ),
                    }
                )
            )

        center_cols, center_min_width, center_modifier = center_grid_layout(
            len(center_cards)
        )
        centers_html = build_grid(
            center_cards,
            extra_class=f"kpi-grid--centers {center_modifier}".strip(),
            min_width=center_min_width,
            columns=center_cols,
            data_attrs={"data-center-count": len(center_cards)},
        )

        if display_name:
            title_html = (
                '<div class="kpi-sku-title">'
                f"{escape(display_name)} "
                f'<span class="kpi-sku-code">{escape(sku)}</span>'
                "</div>"
            )
        else:
            title_html = (
                '<div class="kpi-sku-title">'
                f'<span class="kpi-sku-code">{escape(sku)}</span>'
                "</div>"
            )

        centers_section = (
            '<div class="kpi-section-title">센터별 상세</div>' + centers_html
            if centers_html
            else ""
        )

        sku_cards_html.append(
            '<div class="kpi-sku-card">'
            + title_html
            + summary_html
            + centers_section
            + "</div>"
        )

    cards_html = build_grid(
        sku_cards_html, min_width=sku_min_width, extra_class="kpi-grid--sku"
    )
    st.markdown(cards_html, unsafe_allow_html=True)
    st.caption(
        f"※ {pd.to_datetime(latest_snapshot).normalize():%Y-%m-%d} 스냅샷 기준 KPI이며, 현재 대표 시나리오 필터(센터/기간/SKU)가 반영되었습니다.\n"
        "※ 전체 생산 예정 재고 합계는 오늘 이후 완료 예정인 모든 생산분 반영\n"
        "※ 센터별 생산중(30일 내 완료)은 해당 센터 기준, 오늘부터 30일 이내 완료 예정 생산량 합계"
    )
    return kpi_df
