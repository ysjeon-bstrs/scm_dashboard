"""SHOPEE 스냅샷 KPI 빌드 및 렌더링 유틸리티."""

from __future__ import annotations

import logging
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import streamlit as st

from .amazon_snapshot import _format_int, _inject_card_styles

logger = logging.getLogger(__name__)

# SHOPEE 센터 목록
SHOPEE_CENTERS = ["SBSMY", "SBSSG", "SBSTH", "SBSPH"]

# SHOPEE 센터 표시명
SHOPEE_CENTER_NAMES = {
    "SBSMY": "말레이시아",
    "SBSSG": "싱가포르",
    "SBSTH": "태국",
    "SBSPH": "필리핀",
}

_NUMERIC_COLUMNS = {
    "stock_available",
    "stock_readytoship",
    "sales_qty",
    "selling_speed",
    "coverage_days",
}


def _coerce_shopee_snapshot(
    snap_df: pd.DataFrame,
    centers: Sequence[str],
    skus: Sequence[str],
) -> pd.DataFrame:
    """
    SHOPEE 스냅샷 데이터를 정규화하고 필터링합니다.

    Args:
        snap_df: 스냅샷 데이터프레임 (snap_정제)
        centers: SHOPEE 센터 목록
        skus: 선택된 SKU 목록

    Returns:
        정규화된 SHOPEE 스냅샷 데이터프레임
    """
    if snap_df is None or snap_df.empty:
        return pd.DataFrame(
            columns=[
                "snap_time",
                "center",
                "resource_code",
                "stock_available",
                "stock_readytoship",
                "selling_speed",
                "coverage_days",
            ]
        )

    df = snap_df.copy()
    cols_lower = {str(c).strip().lower(): c for c in df.columns}

    # 컬럼 매핑
    snap_time_col = (
        cols_lower.get("snap_time")
        or cols_lower.get("snapshot_time")
        or cols_lower.get("snapshot_date")
    )
    center_col = cols_lower.get("center")
    sku_col = cols_lower.get("resource_code") or cols_lower.get("sku")
    available_col = cols_lower.get("stock_available") or cols_lower.get("available_qty")
    readytoship_col = cols_lower.get("stock_readytoship") or cols_lower.get(
        "readytoship_qty"
    )
    selling_speed_col = cols_lower.get("selling_speed") or cols_lower.get("sales_speed")
    coverage_col = cols_lower.get("coverage_days") or cols_lower.get("cover_days")

    rename_map: dict[str, str] = {}
    if snap_time_col:
        rename_map[snap_time_col] = "snap_time"
    if center_col:
        rename_map[center_col] = "center"
    if sku_col:
        rename_map[sku_col] = "resource_code"
    if available_col:
        rename_map[available_col] = "stock_available"
    if readytoship_col:
        rename_map[readytoship_col] = "stock_readytoship"
    if selling_speed_col:
        rename_map[selling_speed_col] = "selling_speed"
    if coverage_col:
        rename_map[coverage_col] = "coverage_days"

    df = df.rename(columns=rename_map)

    # snap_time 컬럼이 없지만 date 컬럼이 있는 경우, date를 snap_time으로 사용
    if "snap_time" not in df.columns:
        date_col = cols_lower.get("date")
        if date_col and date_col in df.columns:
            df["snap_time"] = df[date_col]

    required_cols = {"snap_time", "center", "resource_code"}
    if not required_cols.issubset(df.columns):
        logger.warning(
            f"SHOPEE 스냅샷에 필수 컬럼 누락: {required_cols - set(df.columns)}"
        )
        return pd.DataFrame(columns=list(required_cols | _NUMERIC_COLUMNS))

    df["snap_time"] = pd.to_datetime(df.get("snap_time"), errors="coerce")

    # snap_time이 NaT인 경우 date로 보완
    date_col = cols_lower.get("date")
    if date_col and date_col in df.columns:
        date_parsed = pd.to_datetime(df.get(date_col), errors="coerce")
        df["snap_time"] = df["snap_time"].fillna(date_parsed)

    df = df.dropna(subset=["snap_time"]).copy()

    df["center"] = df.get("center", "").astype(str).str.strip()
    df["resource_code"] = df.get("resource_code", "").astype(str).str.strip()

    # 숫자 컬럼 정규화
    for column in _NUMERIC_COLUMNS:
        if column not in df.columns:
            df[column] = 0.0
            continue

        src = df[column]
        # 천단위 구분자(,) 제거 후 숫자 변환
        if isinstance(src, pd.Series):
            cleaned = src.astype(str).str.replace(",", "", regex=False)
        else:
            cleaned = src
        values = pd.to_numeric(cleaned, errors="coerce")
        if not isinstance(values, pd.Series):
            values = pd.Series(values, index=df.index, dtype=float)
        values = values.reindex(df.index)
        df[column] = values.fillna(0).astype(float)
        df[column] = df[column].clip(lower=0)

    # SHOPEE 센터만 필터링
    centers_norm = {str(c).strip() for c in centers if str(c).strip()}
    if centers_norm:
        df = df[df["center"].isin(centers_norm)]

    # SKU 필터링
    skus_norm = {str(sku).strip() for sku in skus if str(sku).strip()}
    if skus_norm:
        df = df[df["resource_code"].isin(skus_norm)]

    return df


def build_shopee_snapshot_kpis(
    snap_df: pd.DataFrame,
    skus: Sequence[str],
    centers: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    SHOPEE 스냅샷 데이터에서 KPI 데이터프레임을 생성합니다.

    Args:
        snap_df: 스냅샷 데이터프레임 (snap_정제)
        skus: 선택된 SKU 목록
        centers: SHOPEE 센터 목록 (기본값: 모든 SHOPEE 센터)

    Returns:
        KPI 데이터프레임 (컬럼: center, resource_code, snap_time, stock_available,
                              stock_readytoship, selling_speed, coverage_days)
    """
    if centers is None:
        centers = SHOPEE_CENTERS

    df = _coerce_shopee_snapshot(snap_df, centers, skus)

    if df.empty:
        logger.info("SHOPEE 스냅샷 데이터가 비어있습니다.")
        return pd.DataFrame()

    # 최신 스냅샷만 사용 (센터별로 최신 날짜 개별 처리)
    latest_snaps = []
    for center in df["center"].unique():
        center_df = df[df["center"] == center]
        if center_df.empty:
            continue
        latest_ts = center_df["snap_time"].max()
        latest_data = center_df[center_df["snap_time"] == latest_ts]
        latest_snaps.append(latest_data)

    if not latest_snaps:
        logger.info("SHOPEE 최신 스냅샷 데이터가 없습니다.")
        return pd.DataFrame()

    result = pd.concat(latest_snaps, ignore_index=True)

    # 센터별로 정렬 (SHOPEE 센터 순서 유지)
    center_order = {c: i for i, c in enumerate(SHOPEE_CENTERS)}
    result["_center_order"] = result["center"].map(center_order).fillna(999)
    result = result.sort_values(["_center_order", "resource_code"]).drop(
        columns=["_center_order"]
    )

    return result


def _format_selling_speed(value: float | None) -> str:
    """
    판매속도를 포맷팅합니다.

    Args:
        value: 판매속도 값

    Returns:
        포맷팅된 문자열 (예: "10.5개/일")
    """
    if value is None:
        return "-"
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return "-"
    # 0 또는 매우 작은 값은 "-"로 표시
    if float(value) <= 0:
        return "-"
    try:
        return f"{float(value):.1f}개/일"
    except (TypeError, ValueError):
        return "-"


def _format_coverage_days(value: float | None) -> str:
    """
    커버일수를 포맷팅합니다.

    Args:
        value: 커버일수 값

    Returns:
        포맷팅된 문자열 (예: "30.5일")
    """
    if value is None:
        return "-"
    if np.isinf(value):
        return "∞"
    if isinstance(value, float) and np.isnan(value):
        return "-"
    # 0 또는 매우 작은 값은 "-"로 표시
    if float(value) <= 0:
        return "-"
    try:
        return f"{float(value):.1f}일"
    except (TypeError, ValueError):
        return "-"


def render_shopee_snapshot_kpis(
    kpi_df: pd.DataFrame | None,
    *,
    selected_skus: Sequence[str],
    sku_colors: Mapping[str, str] | None = None,
    resource_name_map: Mapping[str, str] | None = None,
    show_delta: bool = False,
    previous_df: pd.DataFrame | None = None,
    max_cols: int = 4,
) -> None:
    """
    SHOPEE KPI 카드를 렌더링합니다.

    Args:
        kpi_df: build_shopee_snapshot_kpis()에서 생성한 KPI 데이터프레임
        selected_skus: 선택된 SKU 목록 (수량 0이어도 표시)
        sku_colors: SKU별 색상 매핑
        resource_name_map: SKU → 품명 매핑
        show_delta: 전 스냅샷 대비 델타 표시 여부
        previous_df: 이전 스냅샷 KPI 데이터프레임
        max_cols: 그리드 최대 열 수 (기본값: 4)
    """
    # Amazon 카드 스타일 재사용
    _inject_card_styles()

    color_map = dict(sku_colors or {})

    # 국가별 이모지 매핑
    country_flags = {
        "SBSMY": "🇲🇾",
        "SBSSG": "🇸🇬",
        "SBSTH": "🇹🇭",
        "SBSPH": "🇵🇭",
    }

    # KPI 데이터를 딕셔너리로 변환 (빠른 조회용)
    kpi_dict: dict[tuple[str, str], dict] = {}
    if kpi_df is not None and not kpi_df.empty:
        for row in kpi_df.itertuples(index=False):
            key = (str(row.center), str(row.resource_code))
            kpi_dict[key] = {
                "stock_available": row.stock_available,
                "stock_readytoship": row.stock_readytoship,
                "selling_speed": row.selling_speed,
                "coverage_days": row.coverage_days,
            }

    # 이전 스냅샷 데이터를 딕셔너리로 변환 (센터, SKU별 조회)
    prev_dict: dict[tuple[str, str], dict] = {}
    if show_delta and previous_df is not None and not previous_df.empty:
        for row in previous_df.itertuples(index=False):
            key = (str(row.center), str(row.resource_code))
            prev_dict[key] = {
                "stock_available": (
                    float(row.stock_available) if pd.notna(row.stock_available) else 0
                ),
                "stock_readytoship": (
                    float(row.stock_readytoship)
                    if pd.notna(row.stock_readytoship)
                    else 0
                ),
                "selling_speed": (
                    float(row.selling_speed) if pd.notna(row.selling_speed) else 0
                ),
                "coverage_days": (
                    float(row.coverage_days) if pd.notna(row.coverage_days) else 0
                ),
            }

    # Delta 포맷팅 헬퍼 함수
    def _fmt_with_delta(value: int | float, delta: int | float | None) -> str:
        """정수 값을 delta와 함께 포맷팅합니다."""
        formatted = _format_int(value)
        if delta is not None and delta != 0:
            if delta > 0:
                delta_str = f"{int(delta):+,}"
                return f"{formatted} <span class='delta-up'>(↑{delta_str})</span>"
            else:
                delta_str = f"{abs(int(delta)):,}"
                return f"{formatted} <span class='delta-down'>(↓{delta_str})</span>"
        return formatted

    def _fmt_speed_with_delta(value: float | None, delta: float | None) -> str:
        """판매속도를 delta와 함께 포맷팅합니다."""
        formatted = _format_selling_speed(value)
        if delta is not None and abs(delta) >= 0.1:
            if delta > 0:
                delta_str = f"+{delta:.1f}"
                return f"{formatted} <span class='delta-up'>(↑{delta_str})</span>"
            else:
                delta_str = f"{abs(delta):.1f}"
                return f"{formatted} <span class='delta-down'>(↓{delta_str})</span>"
        return formatted

    def _fmt_cover_with_delta(value: float | None, delta: float | None) -> str:
        """커버일수를 delta와 함께 포맷팅합니다."""
        formatted = _format_coverage_days(value)
        if delta is not None and abs(delta) >= 0.1:
            if delta > 0:
                delta_str = f"+{delta:.1f}"
                return f"{formatted} <span class='delta-up'>(↑{delta_str})</span>"
            else:
                delta_str = f"{abs(delta):.1f}"
                return f"{formatted} <span class='delta-down'>(↓{delta_str})</span>"
        return formatted

    # 센터(국가)별로 구분하여 표시
    for idx, center in enumerate(SHOPEE_CENTERS):
        center_name = SHOPEE_CENTER_NAMES.get(center, center)
        flag = country_flags.get(center, "🏪")

        # 해당 국가의 최신 스냅샷 시각 추출
        center_snap_time = None
        if kpi_df is not None and not kpi_df.empty:
            center_data = kpi_df[kpi_df["center"] == center]
            if not center_data.empty:
                center_snap_time = center_data["snap_time"].max()

        # 국가 헤더 표시
        st.markdown(f"#### {flag} {center_name}")

        # 해당 국가의 카드들 생성 (선택된 모든 SKU에 대해)
        cards_html: list[str] = []

        for sku in selected_skus:
            sku_str = str(sku).strip()
            color = color_map.get(sku_str, "#4E79A7")

            # KPI 데이터 조회 (없으면 기본값 0)
            key = (center, sku_str)
            kpi_data = kpi_dict.get(
                key,
                {
                    "stock_available": 0,
                    "stock_readytoship": 0,
                    "selling_speed": 0,
                    "coverage_days": 0,
                },
            )

            # 이전 스냅샷 데이터 조회 및 delta 계산
            prev_data = prev_dict.get(key, {})
            delta_available = None
            delta_readytoship = None
            delta_speed = None
            delta_cover = None

            if show_delta and prev_data:
                delta_available = kpi_data["stock_available"] - int(
                    prev_data.get("stock_available", 0)
                )
                delta_readytoship = kpi_data["stock_readytoship"] - int(
                    prev_data.get("stock_readytoship", 0)
                )
                # selling_speed와 coverage_days는 0이 아닌 경우에만 delta 계산
                if (
                    kpi_data["selling_speed"] > 0
                    or prev_data.get("selling_speed", 0) > 0
                ):
                    delta_speed = kpi_data["selling_speed"] - prev_data.get(
                        "selling_speed", 0
                    )
                if (
                    kpi_data["coverage_days"] > 0
                    or prev_data.get("coverage_days", 0) > 0
                ):
                    delta_cover = kpi_data["coverage_days"] - prev_data.get(
                        "coverage_days", 0
                    )

            # 품명 조회
            resource_name = ""
            if resource_name_map is not None:
                resource_name = str(resource_name_map.get(sku_str, "")).strip()

            # 헤더: 품명 + SKU (국가명은 이미 위에 표시했으므로 제거)
            if resource_name:
                header_html = (
                    f"<h4><span class='color-dot' style='background-color:{color}'></span>"
                    f"{resource_name} "
                    f"<span style='color: #666; font-size: 0.9em;'>[{sku_str}]</span></h4>"
                )
            else:
                header_html = (
                    f"<h4><span class='color-dot' style='background-color:{color}'></span>"
                    f"{sku_str}</h4>"
                )

            # 메트릭 구성 (delta 포함)
            metrics = [
                (
                    "판매가능",
                    _fmt_with_delta(kpi_data["stock_available"], delta_available),
                    "현재 판매 가능한 재고",
                ),
                (
                    "입고등록",
                    _fmt_with_delta(kpi_data["stock_readytoship"], delta_readytoship),
                    "입고 등록된 재고 (아직 판매 불가)",
                ),
                (
                    "판매속도",
                    _fmt_speed_with_delta(kpi_data["selling_speed"], delta_speed),
                    "일평균 판매 속도",
                ),
                (
                    "커버일수",
                    _fmt_cover_with_delta(kpi_data["coverage_days"], delta_cover),
                    "현재 재고로 판매 가능한 일수",
                ),
            ]

            metric_html_parts: list[str] = []
            for label, value, tooltip in metrics:
                title_attr = f" title='{tooltip}'" if tooltip else ""
                metric_html_parts.append(
                    f"<div class='amz-kpi-metric'{title_attr}>"
                    f"<span class='label'>{label}</span>"
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

        # 해당 국가의 카드들 표시
        st.markdown(
            "<div class='amz-kpi-container'>" + "".join(cards_html) + "</div>",
            unsafe_allow_html=True,
        )

        # 해당 국가의 스냅샷 기준 시간 표시
        if pd.notna(center_snap_time):
            st.caption(f"{center_snap_time:%Y-%m-%d %H:%M} 기준")
        else:
            st.caption("스냅샷 시각 정보를 확인할 수 없습니다.")

        # 마지막 국가가 아니면 구분선 추가
        if idx < len(SHOPEE_CENTERS) - 1:
            st.markdown("<br>", unsafe_allow_html=True)
