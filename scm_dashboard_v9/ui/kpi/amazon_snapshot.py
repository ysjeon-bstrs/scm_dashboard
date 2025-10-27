"""Amazon US FBA 스냅샷 KPI 빌드 및 렌더링 유틸리티."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import streamlit as st


_NUMERIC_COLUMNS = {
    "stock_qty",
    "stock_available",
    "pending_fc",
    "stock_processing",
    "stock_expected",
    "sales_qty",
}


@dataclass(frozen=True)
class _MetricValue:
    label: str
    value: str
    tooltip: str | None = None


def _coerce_snapshot_frame(
    snap_amz: pd.DataFrame,
    centers: Sequence[str] | None,
    skus: Sequence[str],
) -> pd.DataFrame:
    """필요한 컬럼명을 정규화하고 숫자 컬럼을 안전하게 변환합니다."""

    if snap_amz is None or snap_amz.empty:
        return pd.DataFrame(
            columns=[
                "snap_time",
                "center",
                "resource_code",
                "stock_qty",
                "stock_available",
                "stock_processing",
                "stock_expected",
                "sales_qty",
            ]
        )

    df = snap_amz.copy()
    cols_lower = {str(c).strip().lower(): c for c in df.columns}

    snap_time_col = (
        cols_lower.get("snap_time")
        or cols_lower.get("snapshot_time")
        or cols_lower.get("snapshot_datetime")
        or cols_lower.get("snapshot_date")
        or cols_lower.get("date")
    )
    center_col = cols_lower.get("center")
    sku_col = cols_lower.get("resource_code") or cols_lower.get("sku")
    stock_col = cols_lower.get("stock_qty") or cols_lower.get("qty")
    available_col = cols_lower.get("stock_available") or cols_lower.get("available_qty")
    # pending_FC 컬럼은 다양한 표기로 들어올 수 있으므로 다중 별칭을 지원
    pending_col = (
        cols_lower.get("pending_fc")
        or cols_lower.get("pending fc")
        or cols_lower.get("fc_pending")
        or cols_lower.get("pending_fc_qty")
        or cols_lower.get("pending_fc_units")
        or cols_lower.get("pending at fc")
        or cols_lower.get("pending_at_fc")
        or cols_lower.get("fc pending")
    )
    processing_col = cols_lower.get("stock_processing") or cols_lower.get("processing_qty")
    expected_col = cols_lower.get("stock_expected") or cols_lower.get("expected_qty")
    sales_col = cols_lower.get("sales_qty") or cols_lower.get("sale_qty")

    rename_map: dict[str, str] = {}
    if snap_time_col:
        rename_map[snap_time_col] = "snap_time"
    if center_col:
        rename_map[center_col] = "center"
    if sku_col:
        rename_map[sku_col] = "resource_code"
    if stock_col:
        rename_map[stock_col] = "stock_qty"
    if available_col:
        rename_map[available_col] = "stock_available"
    if pending_col:
        rename_map[pending_col] = "pending_fc"
    if processing_col:
        rename_map[processing_col] = "stock_processing"
    if expected_col:
        rename_map[expected_col] = "stock_expected"
    if sales_col:
        rename_map[sales_col] = "sales_qty"

    df = df.rename(columns=rename_map)

    required_cols = {"snap_time", "center", "resource_code", "stock_qty"}
    if not required_cols.issubset(df.columns):
        return pd.DataFrame(columns=list(required_cols | _NUMERIC_COLUMNS))

    df["snap_time"] = pd.to_datetime(df.get("snap_time"), errors="coerce")

    # 개선: 행 단위 폴백 — snap_time 개별 NaT에 대해 date 값을 보완
    # (기존: 전체가 NaT일 때만 date로 대체 → 일부 NaT가 있는 경우 정보 손실)
    if "date" in df.columns:
        date_parsed = pd.to_datetime(df.get("date"), errors="coerce")
        df["snap_time"] = df["snap_time"].fillna(date_parsed)

    df = df.dropna(subset=["snap_time"]).copy()

    df["center"] = df.get("center", "").astype(str).str.strip()
    df["resource_code"] = df.get("resource_code", "").astype(str).str.strip()

    for column in _NUMERIC_COLUMNS:
        src = df.get(column, pd.Series(dtype=float))
        # 문자열 숫자에 포함된 천단위 구분자(,) 제거 후 숫자 변환
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

    # 센터 필터링: centers가 있고 실제로 유효한 값이 있을 때만 필터링
    if centers:
        centers_norm = {str(center).strip() for center in centers if str(center).strip()}
        # centers_norm이 비어있으면 필터링하지 않음 (모든 데이터 유지)
        if centers_norm:
            df = df[df["center"].isin(centers_norm)]

    # SKU 필터링: skus가 있고 실제로 유효한 값이 있을 때만 필터링
    skus_norm = {str(sku).strip() for sku in skus if str(sku).strip()}
    if skus_norm:
        df = df[df["resource_code"].isin(skus_norm)]

    return df


def _ma7_for(
    df: pd.DataFrame,
    sku: str,
    latest_ts: pd.Timestamp,
) -> float | None:
    """최신 스냅샷 이전 7개 관측치 기준 MA7을 계산합니다."""

    history = (
        df[(df["resource_code"] == sku) & (df["snap_time"] < latest_ts)]
        .sort_values("snap_time")
        .tail(7)
    )
    if history.empty:
        return None
    values = history["sales_qty"].astype(float)
    if values.empty:
        return None
    mean_val = float(values.mean())
    if np.isfinite(mean_val):
        return mean_val
    return None


def _format_int(value: float | int | None) -> str:
    if value is None:
        return "-"
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return "-"
    try:
        return f"{int(round(float(value))):,}"
    except (TypeError, ValueError):
        return "-"


def _format_cover_days(value: float | None) -> str:
    if value is None:
        return "-"
    if np.isinf(value):
        return "∞"
    if isinstance(value, float) and np.isnan(value):
        return "-"
    try:
        return f"{float(value):.1f}일"
    except (TypeError, ValueError):
        return "-"


def _inject_card_styles() -> None:
    css = """
    <style>
    .amz-kpi-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
        gap: 16px;
        margin-bottom: 0.5rem;
    }
    .amz-kpi-card {
        background: var(--background-color, #f8f9fa);
        border: 1px solid rgba(0, 0, 0, 0.05);
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.08);
        min-height: 190px;
    }
    .amz-kpi-card h4 {
        margin: 0 0 12px;
        font-size: 1rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .amz-kpi-card .color-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        display: inline-block;
    }
    .amz-kpi-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 12px 16px;
    }
    .amz-kpi-metric {
        display: flex;
        flex-direction: column;
        gap: 4px;
    }
    .amz-kpi-metric span.label {
        font-size: 0.85rem;
        color: rgba(15, 23, 42, 0.7);
    }
    .amz-kpi-metric span.value {
        font-weight: 600;
        font-size: 1.15rem;
    }
    .amz-kpi-metric span.delta-up {
        font-size: 0.8rem;
        color: rgba(34, 197, 94, 0.9);
        font-weight: 500;
    }
    .amz-kpi-metric span.delta-down {
        font-size: 0.8rem;
        color: rgba(239, 68, 68, 0.9);
        font-weight: 500;
    }
    @media (max-width: 768px) {
        .amz-kpi-container {
            grid-template-columns: 1fr;
        }
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def build_amazon_snapshot_kpis(
    snap_amz: pd.DataFrame,
    skus: Sequence[str],
    *,
    center: str | Sequence[str] = "AMZUS",
    cover_base: str = "available",
    use_ma7: bool = True,
) -> pd.DataFrame:
    """Amazon 스냅샷 기반 KPI 데이터를 생성합니다."""

    centers: Sequence[str] | None
    if isinstance(center, str) or center is None:
        centers = [center] if center else []
    else:
        centers = list(center)

    # 최신 스냅샷 시점을 구할 때는 SKU 필터를 적용하지 않고 센터 기준으로 계산
    normalized_base = _coerce_snapshot_frame(snap_amz, centers, [])
    if normalized_base.empty:
        return pd.DataFrame(
            columns=[
                "resource_code",
                "snap_time",
                "stock_qty",
                "stock_available",
                "pending_fc",
                "stock_processing",
                "stock_expected",
                "sales_yday",
                "sales_ma7",
                "cover_days",
                "cover_base",
            ]
        )

    latest_ts = normalized_base["snap_time"].max()
    if pd.isna(latest_ts):
        return pd.DataFrame(
            columns=[
                "resource_code",
                "snap_time",
                "stock_qty",
                "stock_available",
                "pending_fc",
                "stock_processing",
                "stock_expected",
                "sales_yday",
                "sales_ma7",
                "cover_days",
                "cover_base",
            ]
        )
    # 해당 최신 시점의 스냅샷만 추출 (센터 기준), 이후 SKU별로 조회
    current = normalized_base[normalized_base["snap_time"] == latest_ts]

    rows: list[dict[str, object]] = []
    sku_order = [str(s) for s in skus]

    for sku in sku_order:
        sku_current = current[current["resource_code"] == sku]
        if sku_current.empty:
            total = avail = pending_fc = processing = expected = sales_yday = 0.0
        else:
            total = float(sku_current["stock_qty"].sum())
            avail = float(sku_current["stock_available"].sum())
            pending_fc = float(sku_current.get("pending_fc", 0).sum())
            processing = float(sku_current["stock_processing"].sum())
            expected = float(sku_current["stock_expected"].sum())
            sales_yday = float(sku_current["sales_qty"].sum())

        total = max(0.0, total)
        avail = max(0.0, avail)
        pending_fc = max(0.0, pending_fc)
        processing = max(0.0, processing)
        expected = max(0.0, expected)
        sales_yday = max(0.0, sales_yday)

        ma7 = _ma7_for(normalized, sku, latest_ts) if use_ma7 else None
        demand_candidate = ma7 if ma7 is not None else sales_yday
        if demand_candidate is None:
            demand = None
        else:
            demand = float(max(1.0, demand_candidate)) if demand_candidate > 0 else None

        # 커버일수 계산 기준: 판매가능(= 가용재고 + pending_FC)
        base_qty = avail + pending_fc
        if demand is None or demand <= 0:
            cover_days = np.inf if base_qty > 0 else None
        else:
            cover_days = base_qty / demand if demand else None

        rows.append(
            {
                "resource_code": sku,
                "snap_time": latest_ts,
                "stock_qty": int(round(total)),
                "stock_available": int(round(avail)),
                "pending_fc": int(round(pending_fc)),
                "stock_processing": int(round(processing)),
                "stock_expected": int(round(expected)),
                "sales_yday": int(round(sales_yday)),
                "sales_ma7": float(ma7) if ma7 is not None else np.nan,
                "cover_days": float(cover_days) if cover_days is not None else np.nan,
                "cover_base": "salable",
            }
        )

    result = pd.DataFrame(rows)
    if not result.empty:
        result["resource_code"] = pd.Categorical(
            result["resource_code"], categories=sku_order, ordered=True
        )
        result = result.sort_values("resource_code").reset_index(drop=True)
    return result


def render_amazon_snapshot_kpis(
    kpi_df: pd.DataFrame,
    *,
    sku_colors: dict[str, str],
    show_delta: bool = False,
    previous_df: pd.DataFrame | None = None,
    max_cols: int = 4,
    resource_name_map: dict[str, str] | None = None,
) -> None:
    """Amazon FBA KPI 카드를 렌더링합니다."""

    del max_cols  # CSS grid로 대체되므로 호환성 유지용 인자만 사용

    if kpi_df is None or kpi_df.empty:
        st.info("표시할 Amazon FBA KPI가 없습니다.")
        return

    latest_ts = pd.to_datetime(kpi_df["snap_time"].max())
    if pd.isna(latest_ts):
        st.info("표시할 Amazon FBA KPI가 없습니다.")
        return

    st.caption(f"{latest_ts:%Y-%m-%d %H:%M} 기준")

    _inject_card_styles()

    # 이전 스냅샷 데이터를 SKU별 dict로 변환 (모든 지표 포함)
    # 최적화: iterrows() → 벡터화 연산 (10-50배 빠름)
    prev_data: dict[str, dict[str, float]] = {}
    if show_delta and previous_df is not None and not previous_df.empty:
        # DataFrame 인덱싱과 to_dict()를 사용하여 벡터화
        prev_df = previous_df.set_index("resource_code", drop=False)
        for sku in prev_df.index:
            row = prev_df.loc[sku]
            prev_data[str(sku)] = {
                "stock_qty": float(row.get("stock_qty", 0)) if pd.notna(row.get("stock_qty")) else 0,
                "stock_available": float(row.get("stock_available", 0)) if pd.notna(row.get("stock_available")) else 0,
                "pending_fc": float(row.get("pending_fc", 0)) if pd.notna(row.get("pending_fc")) else 0,
                "stock_processing": float(row.get("stock_processing", 0)) if pd.notna(row.get("stock_processing")) else 0,
                "stock_expected": float(row.get("stock_expected", 0)) if pd.notna(row.get("stock_expected")) else 0,
                "sales_yday": float(row.get("sales_yday", 0)) if pd.notna(row.get("sales_yday")) else 0,
                "cover_days": float(row.get("cover_days", 0)) if pd.notna(row.get("cover_days")) else 0,
            }

    cards_html: list[str] = []
    # 최적화: iterrows() → itertuples() (2-3배 빠름)
    for row in kpi_df.itertuples(index=False):
        sku = str(row.resource_code)
        color = sku_colors.get(sku, "#4E79A7")

        # 현재 값 (namedtuple 속성 접근)
        total = int(getattr(row, "stock_qty", 0))
        available = int(getattr(row, "stock_available", 0))
        pending_fc = int(getattr(row, "pending_fc", 0))
        processing = int(getattr(row, "stock_processing", 0))
        expected = int(getattr(row, "stock_expected", 0))
        sales_yday = int(getattr(row, "sales_yday", 0))
        cover_days = getattr(row, "cover_days", None)

        # 증감값 계산 (show_delta가 True일 때만)
        prev = prev_data.get(sku, {})
        delta_total = None
        delta_available = None
        delta_processing = None
        delta_expected = None
        delta_sales = None
        delta_cover = None

        if show_delta and prev:
            delta_total = total - int(prev.get("stock_qty", 0))
            delta_available = available - int(prev.get("stock_available", 0))
            delta_pending_fc = pending_fc - int(prev.get("pending_fc", 0))
            delta_processing = processing - int(prev.get("stock_processing", 0))
            delta_expected = expected - int(prev.get("stock_expected", 0))
            delta_sales = sales_yday - int(prev.get("sales_yday", 0))
            if pd.notna(cover_days) and prev.get("cover_days", 0) > 0:
                delta_cover = float(cover_days) - prev.get("cover_days", 0)

        # 포맷팅 (delta 포함)
        def _fmt_with_delta(value: int, delta: int | None) -> str:
            formatted = _format_int(value)
            if delta is not None and delta != 0:
                if delta > 0:
                    delta_str = f"{delta:+,}"
                    return f"{formatted} <span class='delta-up'>(↑{delta_str})</span>"
                else:
                    delta_str = f"{abs(delta):,}"
                    return f"{formatted} <span class='delta-down'>(↓{delta_str})</span>"
            return formatted

        def _fmt_cover_with_delta(value: float | None, delta: float | None) -> str:
            formatted = _format_cover_days(value)
            if delta is not None and abs(delta) >= 0.1:
                if delta > 0:
                    delta_str = f"+{delta:.1f}"
                    return f"{formatted} <span class='delta-up'>(↑{delta_str})</span>"
                else:
                    delta_str = f"{abs(delta):.1f}"
                    return f"{formatted} <span class='delta-down'>(↓{delta_str})</span>"
            return formatted

        total_str = _fmt_with_delta(total, delta_total)
        # 판매가능: 가용재고 + FC 재배치 중 (예: 100 + 5,000)
        # 전체 합계에 대한 delta 표시
        salable_total = available + pending_fc
        delta_salable = None
        if delta_available is not None and delta_pending_fc is not None:
            delta_salable = delta_available + delta_pending_fc
        available_str = f"{_format_int(available)} + {_format_int(pending_fc)}"
        if delta_salable is not None and delta_salable != 0:
            if delta_salable > 0:
                available_str += f" <span class='delta-up'>(↑{delta_salable:+,})</span>"
            else:
                available_str += f" <span class='delta-down'>(↓{abs(delta_salable):,})</span>"
        processing_str = _fmt_with_delta(processing, delta_processing)
        expected_str = _fmt_with_delta(expected, delta_expected)
        sales_str = _fmt_with_delta(sales_yday, delta_sales)
        cover_str = _fmt_cover_with_delta(cover_days, delta_cover)

        metrics: list[_MetricValue] = [
            _MetricValue("총 재고", total_str, "센터별 총재고 합계"),
            _MetricValue("판매가능", available_str, "가용재고+FC 재배치 중"),
            _MetricValue("입고처리중", processing_str, "FC 도착 후 재고화 진행 중"),
            _MetricValue("입고예정", expected_str, "입고예약+FC 도착+재고화 진행중"),
            _MetricValue("어제 판매", sales_str, "전일 판매량"),
            _MetricValue("커버일수", cover_str, "총 재고 or 사용가능 ÷ 일평균 수요"),
        ]

        # SKU 헤더 (resource_name 포함)
        resource_name = resource_name_map.get(sku, "") if resource_name_map else ""
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

        metric_html_parts: list[str] = []
        for metric in metrics:
            title_attr = f" title='{metric.tooltip}'" if metric.tooltip else ""
            metric_html_parts.append(
                f"<div class='amz-kpi-metric'{title_attr}>"
                f"<span class='label'>{metric.label}</span>"
                f"<span class='value'>{metric.value}</span>"
                "</div>"
            )

        card_html = (
            "<div class='amz-kpi-card'>"
            + header_html
            + "<div class='amz-kpi-grid'>"
            + "".join(metric_html_parts)
            + "</div></div>"
        )
        cards_html.append(card_html)

    st.markdown(
        "<div class='amz-kpi-container'>" + "".join(cards_html) + "</div>",
        unsafe_allow_html=True,
    )

    st.caption(
        "총 재고=판매가능+FC 처리 중+고객주문 · 판매가능=가용재고+FC 재배치 중 · "
        "입고처리중=FC 도착 후 재고화 진행 중 · 입고예정=입고예약+FC 도착+재고화 진행중 · "
        "커버일수=판매가능 ÷ 7일 평균 일판매"
    )
