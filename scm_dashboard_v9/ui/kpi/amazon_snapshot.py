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

    # snap_time이 모두 null이면 date 컬럼을 폴백으로 사용
    if df["snap_time"].isna().all():
        if "date" in df.columns:
            df["snap_time"] = pd.to_datetime(df.get("date"), errors="coerce")

    df = df.dropna(subset=["snap_time"]).copy()

    df["center"] = df.get("center", "").astype(str).str.strip()
    df["resource_code"] = df.get("resource_code", "").astype(str).str.strip()

    for column in _NUMERIC_COLUMNS:
        values = pd.to_numeric(
            df.get(column, pd.Series(dtype=float)), errors="coerce"
        )
        if not isinstance(values, pd.Series):
            values = pd.Series(values, index=df.index, dtype=float)
        values = values.reindex(df.index)
        df[column] = values.fillna(0).astype(float)
        df[column] = df[column].clip(lower=0)

    if centers:
        centers_norm = {str(center).strip() for center in centers if str(center).strip()}
        df = df[df["center"].isin(centers_norm)]
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
    .amz-kpi-metric span.delta {
        font-size: 0.8rem;
        color: rgba(15, 118, 110, 0.9);
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

    normalized = _coerce_snapshot_frame(snap_amz, centers, skus)
    if normalized.empty:
        return pd.DataFrame(
            columns=[
                "resource_code",
                "snap_time",
                "stock_qty",
                "stock_available",
                "stock_processing",
                "stock_expected",
                "sales_yday",
                "sales_ma7",
                "cover_days",
                "cover_base",
            ]
        )

    latest_ts = normalized["snap_time"].max()
    if pd.isna(latest_ts):
        return pd.DataFrame(
            columns=[
                "resource_code",
                "snap_time",
                "stock_qty",
                "stock_available",
                "stock_processing",
                "stock_expected",
                "sales_yday",
                "sales_ma7",
                "cover_days",
                "cover_base",
            ]
        )

    current = normalized[normalized["snap_time"] == latest_ts]

    rows: list[dict[str, object]] = []
    sku_order = [str(s) for s in skus]

    for sku in sku_order:
        sku_current = current[current["resource_code"] == sku]
        if sku_current.empty:
            total = avail = processing = expected = sales_yday = 0.0
        else:
            total = float(sku_current["stock_qty"].sum())
            avail = float(sku_current["stock_available"].sum())
            processing = float(sku_current["stock_processing"].sum())
            expected = float(sku_current["stock_expected"].sum())
            sales_yday = float(sku_current["sales_qty"].sum())

        total = max(0.0, total)
        avail = max(0.0, avail)
        processing = max(0.0, processing)
        expected = max(0.0, expected)
        sales_yday = max(0.0, sales_yday)

        ma7 = _ma7_for(normalized, sku, latest_ts) if use_ma7 else None
        demand_candidate = ma7 if ma7 is not None else sales_yday
        if demand_candidate is None:
            demand = None
        else:
            demand = float(max(1.0, demand_candidate)) if demand_candidate > 0 else None

        base_qty = avail if cover_base == "available" else total
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
                "stock_processing": int(round(processing)),
                "stock_expected": int(round(expected)),
                "sales_yday": int(round(sales_yday)),
                "sales_ma7": float(ma7) if ma7 is not None else np.nan,
                "cover_days": float(cover_days) if cover_days is not None else np.nan,
                "cover_base": cover_base,
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

    prev_map: dict[str, int] = {}
    if show_delta and previous_df is not None and not previous_df.empty:
        prev_map = {
            str(row["resource_code"]): int(row["stock_qty"])
            for _, row in previous_df.iterrows()
            if not pd.isna(row.get("stock_qty"))
        }

    cards_html: list[str] = []
    for _, row in kpi_df.iterrows():
        sku = str(row["resource_code"])
        color = sku_colors.get(sku, "#4E79A7")
        total = _format_int(row.get("stock_qty"))
        available = _format_int(row.get("stock_available"))
        processing = _format_int(row.get("stock_processing"))
        expected = _format_int(row.get("stock_expected"))
        sales_yday = _format_int(row.get("sales_yday"))
        cover = _format_cover_days(row.get("cover_days"))

        metrics: list[_MetricValue] = [
            _MetricValue("총 재고", total, "센터별 총재고 합계"),
            _MetricValue("사용가능", available, "Available 재고"),
            _MetricValue("입고처리중", processing, "FC 접수→가능화 전"),
            _MetricValue("입고예정", expected, "ASN/재배치 도착 예정"),
            _MetricValue("어제 판매", sales_yday, "전일 판매량"),
            _MetricValue("커버일수", cover, "사용가능 ÷ 일평균 수요"),
        ]

        delta_text: str | None = None
        if show_delta:
            prev_value = prev_map.get(sku)
            if prev_value is not None:
                delta_val = int(row.get("stock_qty", 0)) - prev_value
                delta_text = f"{delta_val:+,}" if delta_val else "±0"

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

        if delta_text:
            metric_html_parts[0] = metric_html_parts[0].replace(
                "</div>",
                f"<span class='delta'>Δ {delta_text}</span></div>",
                1,
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
        "입고처리중=FC 접수→가능화 전 · 입고예정=ASN/재배치로 도착 예정 · "
        "커버일수=사용가능 ÷ 7일 평균 일판매"
    )
