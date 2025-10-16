"""KPI rendering helpers for the Streamlit dashboard.

This module now renders KPI cards with a responsive grid layout so that
metrics automatically wrap when the available width becomes narrow.  The
``render_sku_summary_cards`` helper can be called directly in a Streamlit app:

.. code-block:: python

    import pandas as pd
    import streamlit as st

    from scm_dashboard_v5.ui.kpi import render_sku_summary_cards

    st.set_page_config(layout="wide")

    snapshot_df = pd.read_csv("snapshot.csv")
    moves_df = pd.read_csv("moves.csv")
    today = pd.Timestamp.today()

    render_sku_summary_cards(
        snapshot_df,
        moves_df,
        centers=["Amazon US", "태광KR"],
        skus=["BA00021"],
        today=today,
        lag_days=5,
        chunk_size=3,
    )

The responsive layout removes the need to manually adjust column counts for
different viewport sizes.
"""

from __future__ import annotations

import html
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import streamlit as st

from scm_dashboard_v5.analytics import kpi_breakdown_per_sku
from scm_dashboard_v5.core import build_timeline as build_core_timeline
from scm_dashboard_v5.forecast import apply_consumption_with_events


def compute_depletion_from_timeline(
    base_timeline: pd.DataFrame,
    snap_long: pd.DataFrame,
    centers: Sequence[str],
    skus: Sequence[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    today: pd.Timestamp,
    *,
    lookback_days: int,
    events: Optional[Sequence[Dict[str, object]]] = None,
) -> Dict[tuple[str, str], Dict[str, Optional[object]]]:
    """Simulate depletion using the shared consumption engine."""

    if base_timeline is None or base_timeline.empty:
        return {}

    today_norm = pd.to_datetime(today).normalize()
    start_norm = pd.to_datetime(start).normalize()
    end_norm = pd.to_datetime(end).normalize()

    timeline_with_consumption = apply_consumption_with_events(
        base_timeline,
        snap_long,
        centers=list(centers),
        skus=list(skus),
        start=start_norm,
        end=end_norm,
        lookback_days=int(lookback_days),
        events=list(events) if events else None,
    )

    if timeline_with_consumption is None or timeline_with_consumption.empty:
        return {}

    timeline_copy = timeline_with_consumption.copy()
    timeline_copy["date"] = pd.to_datetime(
        timeline_copy["date"], errors="coerce"
    ).dt.normalize()
    timeline_copy = timeline_copy.dropna(subset=["date"])
    if timeline_copy.empty:
        return {}

    filtered = timeline_copy[
        ~timeline_copy["center"].isin(["In-Transit", "WIP"])
    ].copy()
    if filtered.empty:
        return {}

    out: Dict[tuple[str, str], Dict[str, Optional[object]]] = {}

    for (center, sku), group in filtered.groupby(["center", "resource_code"]):
        segment = group.sort_values("date")
        future_mask = segment["date"] >= today_norm
        future_segment = segment.loc[future_mask]
        if future_segment.empty:
            out[(str(center), str(sku))] = {"days": None, "date": None}
            continue
        zero_idx = np.where(future_segment["stock_qty"].values <= 0)[0]
        if zero_idx.size == 0:
            out[(str(center), str(sku))] = {"days": None, "date": None}
            continue
        zero_date = pd.to_datetime(
            future_segment.iloc[int(zero_idx[0])]["date"]
        ).normalize()
        days = max(int((zero_date - today_norm).days), 0)
        out[(str(center), str(sku))] = {"days": days, "date": zero_date}

    for sku, group in filtered.groupby("resource_code"):
        segment = group[group["date"] >= today_norm].sort_values("date")
        if segment.empty:
            out[("__TOTAL__", str(sku))] = {"days": None, "date": None}
            continue
        agg = (
            segment.groupby("date", as_index=False)["stock_qty"].sum().sort_values("date")
        )
        zero_idx = np.where(agg["stock_qty"].values <= 0)[0]
        if zero_idx.size == 0:
            out[("__TOTAL__", str(sku))] = {"days": None, "date": None}
            continue
        zero_date = pd.to_datetime(agg.iloc[int(zero_idx[0])]["date"]).normalize()
        days = max(int((zero_date - today_norm).days), 0)
        out[("__TOTAL__", str(sku))] = {"days": days, "date": zero_date}

    return out


def compute_depletion_metrics(
    snap_long: pd.DataFrame,
    moves: pd.DataFrame,
    centers: Sequence[str],
    skus: Sequence[str],
    today: pd.Timestamp,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    lookback_days: int = 28,
    horizon_pad_days: int = 60,
    events: Optional[Sequence[Dict[str, object]]] = None,
    include_total: bool = True,
) -> pd.DataFrame:
    """Return deterministic depletion metrics per center and SKU.

    The calculation mirrors the timeline simulation used in the charts: the
    latest snapshot is rolled forward by ``apply_consumption_with_events``
    starting from the day after the snapshot while respecting future inbound
    and outbound moves.  The first day where the simulated stock quantity is
    zero or negative becomes the depletion date.
    """

    result_columns = ["center", "resource_code", "days_to_depletion", "depletion_date"]

    if snap_long is None or snap_long.empty:
        return pd.DataFrame(columns=result_columns)

    date_series = None
    for candidate in ("snapshot_date", "date"):
        if candidate in snap_long.columns:
            date_series = pd.to_datetime(snap_long[candidate], errors="coerce")
            break
    if date_series is None:
        return pd.DataFrame(columns=result_columns)

    latest_snap = date_series.max()
    if pd.isna(latest_snap):
        return pd.DataFrame(columns=result_columns)

    latest_snap = latest_snap.normalize()
    today = pd.to_datetime(today).normalize()
    start = pd.to_datetime(start).normalize()
    end = pd.to_datetime(end).normalize()

    horizon_days = max(0, int((end - latest_snap).days)) + int(horizon_pad_days)
    sim_end = end + pd.Timedelta(days=int(horizon_pad_days))

    timeline = build_core_timeline(
        snap_long,
        moves,
        centers=list(centers),
        skus=list(skus),
        start=start,
        end=end,
        today=today,
        horizon_days=horizon_days,
    )

    if timeline is None or timeline.empty:
        return pd.DataFrame(columns=result_columns)

    depletion_map = compute_depletion_from_timeline(
        timeline,
        snap_long,
        centers=list(centers),
        skus=list(skus),
        start=start,
        end=sim_end,
        today=today,
        lookback_days=int(lookback_days),
        events=list(events) if events else None,
    )

    if not depletion_map:
        return pd.DataFrame(columns=result_columns)

    rows: List[Dict[str, object]] = []
    for (center, sku), values in depletion_map.items():
        rows.append(
            {
                "center": center,
                "resource_code": sku,
                "days_to_depletion": values.get("days"),
                "depletion_date": values.get("date"),
            }
        )

    result = pd.DataFrame(rows, columns=result_columns)

    if not include_total:
        result = result[result["center"] != "__TOTAL__"].copy()

    return result


def _escape(value: object) -> str:
    return html.escape("" if value is None else str(value))


def _format_number(value: float | int | None) -> str:
    if value is None:
        return "-"
    if isinstance(value, float) and pd.isna(value):
        return "-"
    try:
        return f"{int(round(float(value))):,}"
    except (TypeError, ValueError):
        return "-"


def _value_font_size(value: str, *, base_size: float = 1.25, min_size: float = 0.9) -> str:
    """Return a font-size in ``rem`` units that keeps long numbers visible."""

    if not value:
        return f"{max(min_size, base_size):.2f}rem"

    digit_count = sum(ch.isdigit() for ch in value)
    if digit_count <= 4:
        scale = 1.0
    elif digit_count <= 6:
        scale = 0.9
    elif digit_count <= 8:
        scale = 0.8
    elif digit_count <= 10:
        scale = 0.72
    else:
        scale = 0.65

    size = max(min_size, base_size * scale)
    return f"{size:.2f}rem"


def _format_days(value: float | None) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "-"
    if value < 0:
        value = 0.0
    if value >= 100:
        return f"{int(round(value))}일"
    return f"{value:.1f}일"


def _format_date(value: pd.Timestamp | None) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "-"
    if pd.isna(value):
        return "-"
    return f"{pd.to_datetime(value):%Y-%m-%d}"


def _calculate_coverage_days(current_qty: float | int | None, daily_demand: float | int | None) -> float | None:
    if current_qty is None:
        return None
    if isinstance(current_qty, float) and pd.isna(current_qty):
        return None
    if daily_demand is None or (isinstance(daily_demand, float) and pd.isna(daily_demand)):
        return None
    try:
        current_val = float(current_qty)
        demand_val = float(daily_demand)
    except (TypeError, ValueError):
        return None
    if demand_val <= 0:
        return None
    if current_val <= 0:
        return 0.0
    return current_val / demand_val


def _calculate_sellout_date(today: pd.Timestamp, coverage_days: float | None) -> pd.Timestamp | None:
    if coverage_days is None or (isinstance(coverage_days, float) and pd.isna(coverage_days)):
        return None
    coverage = max(float(coverage_days), 0.0)
    return pd.to_datetime(today) + pd.to_timedelta(coverage, unit="D")


def _should_show_in_transit(center: str, in_transit_value: int) -> bool:
    center_name = str(center).replace(" ", "").lower()
    if any(keyword in center_name for keyword in ["태광", "taekwang", "tae-kwang"]):
        return in_transit_value > 0
    return True


def _inject_responsive_styles() -> None:
    """Inject shared CSS styles for KPI cards (re-inject on each run)."""

    st.markdown(
        """
        <style>
        :root {
            --kpi-card-border: rgba(49, 51, 63, 0.2);
            --kpi-card-radius: 0.75rem;
            --kpi-card-padding: 0.85rem 1rem;
        }

        .kpi-card-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(var(--min-card-width, 280px), 1fr));
            gap: 0.75rem;
            align-items: stretch;
        }

        .kpi-sku-card {
            border: 1px solid var(--kpi-card-border);
            border-radius: var(--kpi-card-radius);
            padding: var(--kpi-card-padding);
            background-color: rgba(255, 255, 255, 0.8);
            box-shadow: 0 8px 16px rgba(49, 51, 63, 0.08);
            display: flex;
            flex-direction: column;
            gap: 0.9rem;
        }

        .kpi-sku-title {
            display: flex;
            flex-wrap: wrap;
            gap: 0.35rem 0.75rem;
            align-items: baseline;
            font-size: 1.1rem;
            font-weight: 600;
        }

        .kpi-sku-code {
            font-family: "SFMono-Regular", ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas,
                "Liberation Mono", "Courier New", monospace;
            font-size: 0.85rem;
            padding: 0.1rem 0.45rem;
            border-radius: 0.4rem;
            background-color: rgba(49, 51, 63, 0.08);
        }

        .kpi-section-title {
            font-weight: 600;
            margin-top: 0.25rem;
        }

        .kpi-metric-card {
            border: 1px solid var(--kpi-card-border);
            border-radius: 0.65rem;
            padding: 0.75rem 0.85rem;
            background: rgba(250, 250, 251, 0.9);
            display: flex;
            flex-direction: column;
            gap: 0.3rem;
            min-height: 100%;
            overflow: visible;
        }

        .kpi-metric-card--compact {
            padding: 0.6rem 0.75rem;
        }

        .kpi-metric-label {
            font-size: 0.85rem;
            color: rgba(49, 51, 63, 0.75);
            white-space: normal;
        }

        .kpi-metric-value {
            font-size: 1.25rem;
            font-weight: 700;
            white-space: nowrap;
            word-break: keep-all;
            overflow: visible;
        }

        .kpi-center-card {
            border: 1px solid var(--kpi-card-border);
            border-radius: 0.65rem;
            padding: 0.8rem 0.9rem;
            background-color: rgba(255, 255, 255, 0.85);
            display: flex;
            flex-direction: column;
            gap: 0.55rem;
        }

        .kpi-center-title {
            font-weight: 600;
            font-size: 1rem;
        }

        .kpi-grid--summary {
            --min-card-width: clamp(200px, 24vw, 260px);
        }

        .kpi-grid--centers {
            --min-card-width: clamp(220px, 24vw, 320px);
        }

        .kpi-grid--sku {
            --min-card-width: 320px;
        }

        .kpi-grid--compact {
            --min-card-width: 150px;
        }

        .kpi-grid--center-metrics {
            --min-card-width: clamp(140px, 28vw, 200px);
            align-items: stretch;
        }

        .kpi-grid--centers.kpi-grid--centers-narrow {
            --min-card-width: clamp(260px, 28vw, 340px);
        }

        .kpi-grid--centers.kpi-grid--centers-medium {
            --min-card-width: clamp(240px, 26vw, 320px);
        }

        .kpi-grid--centers.kpi-grid--centers-wide {
            --min-card-width: clamp(220px, 24vw, 300px);
        }

        .kpi-grid--centers.kpi-grid--centers-dense {
            --min-card-width: clamp(200px, 22vw, 280px);
        }

        @media (max-width: 1200px) {
            .kpi-card-grid {
                gap: 0.65rem;
            }

            .kpi-metric-value {
                font-size: 1.2rem;
            }
        }

        @media (max-width: 900px) {
            .kpi-grid--summary,
            .kpi-grid--centers,
            .kpi-grid--sku {
                --min-card-width: clamp(220px, 48vw, 320px);
            }

            .kpi-grid--center-metrics {
                --min-card-width: clamp(150px, 42vw, 200px);
            }

            .kpi-sku-card {
                padding: 0.75rem 0.85rem;
            }
        }

        @media (max-width: 700px) {
            .kpi-grid--summary,
            .kpi-grid--centers,
            .kpi-grid--sku {
                grid-template-columns: repeat(auto-fit, minmax(100%, 1fr));
            }

            .kpi-grid--center-metrics {
                grid-template-columns: repeat(auto-fit, minmax(48%, 1fr));
            }

            .kpi-metric-label {
                font-size: 0.8rem;
            }

            .kpi-metric-value {
                font-size: 1.05rem;
            }
        }

        @media (max-width: 520px) {
            .kpi-grid--center-metrics {
                grid-template-columns: repeat(auto-fit, minmax(100%, 1fr));
            }

            .kpi-sku-title {
                font-size: 1.0rem;
            }

            .kpi-sku-code {
                font-size: 0.8rem;
            }
        }

        @media (prefers-color-scheme: dark) {
            .kpi-sku-card,
            .kpi-center-card,
            .kpi-metric-card {
                background-color: rgba(13, 17, 23, 0.55);
                border-color: rgba(250, 250, 251, 0.15);
            }

            .kpi-metric-label {
                color: rgba(250, 250, 251, 0.75);
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _build_metric_card(label: str, value: str, *, compact: bool = False) -> str:
    classes = ["kpi-metric-card"]
    if compact:
        classes.append("kpi-metric-card--compact")
    value_text = "-" if value is None else str(value)
    base_font = 1.55 if not compact else 1.3
    min_font = 1.05 if not compact else 0.85
    font_size = _value_font_size(value_text, base_size=base_font, min_size=min_font)
    return (
        f'<div class="{" ".join(classes)}">'
        f'<div class="kpi-metric-label">{_escape(label)}</div>'
        f'<div class="kpi-metric-value" style="font-size:{font_size}; white-space:nowrap;">{_escape(value_text)}</div>'
        "</div>"
    )


def _build_grid(
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
            attr_parts.append(f'{_escape(key)}="{_escape(value)}"')
    attrs = (" " + " ".join(attr_parts)) if attr_parts else ""

    return (
        f'<div class="{" ".join(classes)}"{attrs}{style}>'
        + "".join(items)
        + "</div>"
    )


def _center_grid_layout(count: int) -> tuple[int | None, int, str]:
    """Return (columns, min_width, modifier_class) for the center KPI grid."""

    if count <= 2:
        return None, 320, "kpi-grid--centers-narrow"
    if count <= 4:
        return None, 280, "kpi-grid--centers-medium"
    if count <= 6:
        return None, 250, "kpi-grid--centers-wide"
    return None, 220, "kpi-grid--centers-dense"


def _build_center_card(center_info: Mapping[str, object]) -> str:
    metric_cards = [
        _build_metric_card("재고", _format_number(center_info["current"]), compact=True),
        _build_metric_card(
            "이동중",
            _format_number(center_info["in_transit"]) if center_info["show_in_transit"] else "-",
            compact=True,
        ),
        _build_metric_card("생산중", _format_number(center_info["wip"]), compact=True),
        _build_metric_card("예상 소진일수", _format_days(center_info["coverage"]), compact=True),
        _build_metric_card("소진 예상일", _format_date(center_info["sellout_date"]), compact=True),
    ]
    metrics_html = _build_grid(
        metric_cards,
        extra_class="kpi-grid--compact kpi-grid--center-metrics",
        min_width=140,
    )
    return (
        '<div class="kpi-center-card">'
        f'<div class="kpi-center-title">{_escape(center_info["center"])}</div>'
        f"{metrics_html}"
        "</div>"
    )


def _extract_daily_demand(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    if frame.empty:
        empty = pd.Series(dtype=float)
        return empty, empty

    candidates = [
        "forecast_daily_qty",
        "forecast_daily_sales",
        "expected_daily_sales",
        "daily_sales",
        "daily_demand",
        "avg_daily_sales",
        "average_daily_sales",
        "sales_avg_daily",
    ]

    for column in candidates:
        if column not in frame.columns:
            continue
        demand_values = pd.to_numeric(frame[column], errors="coerce")
        if demand_values.notna().any():
            demand_frame = frame.assign(_demand=demand_values)
            demand_series = (
                demand_frame.dropna(subset=["_demand"])
                .groupby(["resource_code", "center"])["_demand"]
                .mean()
            )
            total_series = demand_series.groupby(level=0).sum()
            return demand_series, total_series

    empty = pd.Series(dtype=float)
    return empty, empty


def _movement_breakdown_per_center(
    moves: pd.DataFrame,
    centers: Sequence[str],
    skus: Sequence[str],
    today: pd.Timestamp,
    lag_days: int,
) -> tuple[pd.Series, pd.Series]:
    if moves is None or moves.empty:
        empty = pd.Series(dtype=float)
        return empty, empty

    required_columns = {"resource_code", "to_center", "qty_ea"}
    if not required_columns.issubset(moves.columns):
        empty = pd.Series(dtype=float)
        return empty, empty

    mv = moves.copy()
    mv["qty_ea"] = pd.to_numeric(mv["qty_ea"], errors="coerce").fillna(0)
    mv = mv[mv["qty_ea"] != 0]

    if mv.empty:
        empty = pd.Series(dtype=float)
        return empty, empty

    centers_set = {str(center) for center in centers}
    skus_set = {str(sku) for sku in skus}
    mv = mv[mv["to_center"].isin(centers_set) & mv["resource_code"].isin(skus_set)]

    if mv.empty:
        empty = pd.Series(dtype=float)
        return empty, empty

    pred_end = pd.Series(pd.NaT, index=mv.index, dtype="datetime64[ns]")
    if "inbound_date" in mv.columns:
        inbound_mask = mv["inbound_date"].notna()
        pred_end.loc[inbound_mask] = mv.loc[inbound_mask, "inbound_date"]
    else:
        inbound_mask = pd.Series(False, index=mv.index)

    if "arrival_date" in mv.columns:
        arrival_mask = (~inbound_mask) & mv["arrival_date"].notna()
        if arrival_mask.any():
            past_arrival = arrival_mask & (mv["arrival_date"] <= today)
            pred_end.loc[past_arrival] = mv.loc[past_arrival, "arrival_date"] + pd.Timedelta(days=lag_days)

            future_arrival = arrival_mask & (mv["arrival_date"] > today)
            pred_end.loc[future_arrival] = mv.loc[future_arrival, "arrival_date"]

    pred_end = pred_end.fillna(today + pd.Timedelta(days=1))
    mv["pred_end_date"] = pred_end

    carrier_mode = mv["carrier_mode"].str.upper() if "carrier_mode" in mv.columns else ""

    in_transit_series = pd.Series(dtype=float)
    if "onboard_date" in mv.columns:
        in_transit_mask = mv["onboard_date"].notna() & (mv["onboard_date"] <= today) & (today < mv["pred_end_date"])
        if "carrier_mode" in mv.columns:
            in_transit_mask &= carrier_mode != "WIP"
        if in_transit_mask.any():
            in_transit_series = (
                mv[in_transit_mask]
                .groupby(["resource_code", "to_center"], as_index=True)["qty_ea"]
                .sum()
            )

    wip_series = pd.Series(dtype=float)
    if "carrier_mode" in mv.columns and (carrier_mode == "WIP").any():
        wip_frame = mv[carrier_mode == "WIP"].copy()
        if not wip_frame.empty and "onboard_date" in wip_frame.columns:
            add = (
                wip_frame.dropna(subset=["onboard_date"])
                .set_index(["resource_code", "to_center", "onboard_date"])["qty_ea"]
            )
            rem = pd.Series(dtype=float)
            if "event_date" in wip_frame.columns:
                rem = (
                    wip_frame.dropna(subset=["event_date"])
                    .set_index(["resource_code", "to_center", "event_date"])["qty_ea"]
                    * -1
                )
            flow = pd.concat([add, rem]) if not rem.empty else add
            flow = flow.groupby(level=[0, 1, 2]).sum()
            flow = flow[flow.index.get_level_values(2) <= today]
            if not flow.empty:
                wip_series = (
                    flow.groupby(level=[0, 1])
                    .cumsum()
                    .groupby(level=[0, 1])
                    .last()
                    .clip(lower=0)
                )

    if not in_transit_series.empty:
        in_transit_series = in_transit_series.clip(lower=0).round().astype(int)
    if not wip_series.empty:
        wip_series = wip_series.clip(lower=0).round().astype(int)

    return in_transit_series, wip_series


def _format_number(value: float | int | None) -> str:
    if value is None:
        return "-"
    if isinstance(value, float) and pd.isna(value):
        return "-"
    try:
        return f"{int(round(float(value))):,}"
    except (TypeError, ValueError):
        return "-"


def _format_days(value: float | None) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "-"
    if value < 0:
        value = 0.0
    if value >= 100:
        return f"{int(round(value))}일"
    return f"{value:.1f}일"


def _format_date(value: pd.Timestamp | None) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "-"
    if pd.isna(value):
        return "-"
    return f"{pd.to_datetime(value):%Y-%m-%d}"


def _calculate_coverage_days(current_qty: float | int | None, daily_demand: float | int | None) -> float | None:
    if current_qty is None:
        return None
    if isinstance(current_qty, float) and pd.isna(current_qty):
        return None
    if daily_demand is None or (isinstance(daily_demand, float) and pd.isna(daily_demand)):
        return None
    try:
        current_val = float(current_qty)
        demand_val = float(daily_demand)
    except (TypeError, ValueError):
        return None
    if demand_val <= 0:
        return None
    if current_val <= 0:
        return 0.0
    return current_val / demand_val


def _calculate_sellout_date(today: pd.Timestamp, coverage_days: float | None) -> pd.Timestamp | None:
    if coverage_days is None or (isinstance(coverage_days, float) and pd.isna(coverage_days)):
        return None
    coverage = max(float(coverage_days), 0.0)
    return pd.to_datetime(today) + pd.to_timedelta(coverage, unit="D")


def _should_show_in_transit(center: str, in_transit_value: int) -> bool:
    center_name = str(center).replace(" ", "").lower()
    if any(keyword in center_name for keyword in ["태광", "taekwang", "tae-kwang"]):
        return in_transit_value > 0
    return True


def _extract_daily_demand(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    if frame.empty:
        empty = pd.Series(dtype=float)
        return empty, empty

    candidates = [
        "forecast_daily_qty",
        "forecast_daily_sales",
        "expected_daily_sales",
        "daily_sales",
        "daily_demand",
        "avg_daily_sales",
        "average_daily_sales",
        "sales_avg_daily",
    ]

    for column in candidates:
        if column not in frame.columns:
            continue
        demand_values = pd.to_numeric(frame[column], errors="coerce")
        if demand_values.notna().any():
            demand_frame = frame.assign(_demand=demand_values)
            demand_series = (
                demand_frame.dropna(subset=["_demand"])
                .groupby(["resource_code", "center"])["_demand"]
                .mean()
            )
            total_series = demand_series.groupby(level=0).sum()
            return demand_series, total_series

    empty = pd.Series(dtype=float)
    return empty, empty


def _movement_breakdown_per_center(
    moves: pd.DataFrame,
    centers: Sequence[str],
    skus: Sequence[str],
    today: pd.Timestamp,
    lag_days: int,
) -> tuple[pd.Series, pd.Series]:
    if moves is None or moves.empty:
        empty = pd.Series(dtype=float)
        return empty, empty

    required_columns = {"resource_code", "to_center", "qty_ea"}
    if not required_columns.issubset(moves.columns):
        empty = pd.Series(dtype=float)
        return empty, empty

    mv = moves.copy()
    mv["qty_ea"] = pd.to_numeric(mv["qty_ea"], errors="coerce").fillna(0)
    mv = mv[mv["qty_ea"] != 0]

    if mv.empty:
        empty = pd.Series(dtype=float)
        return empty, empty

    centers_set = {str(center) for center in centers}
    skus_set = {str(sku) for sku in skus}
    mv = mv[mv["to_center"].isin(centers_set) & mv["resource_code"].isin(skus_set)]

    if mv.empty:
        empty = pd.Series(dtype=float)
        return empty, empty

    pred_end = pd.Series(pd.NaT, index=mv.index, dtype="datetime64[ns]")
    if "inbound_date" in mv.columns:
        inbound_mask = mv["inbound_date"].notna()
        pred_end.loc[inbound_mask] = mv.loc[inbound_mask, "inbound_date"]
    else:
        inbound_mask = pd.Series(False, index=mv.index)

    if "arrival_date" in mv.columns:
        arrival_mask = (~inbound_mask) & mv["arrival_date"].notna()
        if arrival_mask.any():
            past_arrival = arrival_mask & (mv["arrival_date"] <= today)
            pred_end.loc[past_arrival] = mv.loc[past_arrival, "arrival_date"] + pd.Timedelta(days=lag_days)

            future_arrival = arrival_mask & (mv["arrival_date"] > today)
            pred_end.loc[future_arrival] = mv.loc[future_arrival, "arrival_date"]

    pred_end = pred_end.fillna(today + pd.Timedelta(days=1))
    mv["pred_end_date"] = pred_end

    carrier_mode = mv["carrier_mode"].str.upper() if "carrier_mode" in mv.columns else ""

    in_transit_series = pd.Series(dtype=float)
    if "onboard_date" in mv.columns:
        in_transit_mask = mv["onboard_date"].notna() & (mv["onboard_date"] <= today) & (today < mv["pred_end_date"])
        if "carrier_mode" in mv.columns:
            in_transit_mask &= carrier_mode != "WIP"
        if in_transit_mask.any():
            in_transit_series = (
                mv[in_transit_mask]
                .groupby(["resource_code", "to_center"], as_index=True)["qty_ea"]
                .sum()
            )

    wip_series = pd.Series(dtype=float)
    if "carrier_mode" in mv.columns and (carrier_mode == "WIP").any():
        wip_frame = mv[carrier_mode == "WIP"].copy()
        if not wip_frame.empty and "onboard_date" in wip_frame.columns:
            add = (
                wip_frame.dropna(subset=["onboard_date"])
                .set_index(["resource_code", "to_center", "onboard_date"])["qty_ea"]
            )
            rem = pd.Series(dtype=float)
            if "event_date" in wip_frame.columns:
                rem = (
                    wip_frame.dropna(subset=["event_date"])
                    .set_index(["resource_code", "to_center", "event_date"])["qty_ea"]
                    * -1
                )
            flow = pd.concat([add, rem]) if not rem.empty else add
            flow = flow.groupby(level=[0, 1, 2]).sum()
            flow = flow[flow.index.get_level_values(2) <= today]
            if not flow.empty:
                wip_series = (
                    flow.groupby(level=[0, 1])
                    .cumsum()
                    .groupby(level=[0, 1])
                    .last()
                    .clip(lower=0)
                )

    if not in_transit_series.empty:
        in_transit_series = in_transit_series.clip(lower=0).round().astype(int)
    if not wip_series.empty:
        wip_series = wip_series.clip(lower=0).round().astype(int)

    return in_transit_series, wip_series


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

    snapshot_view = snapshot.copy()
    if date_column not in snapshot_view.columns and "snapshot_date" in snapshot_view.columns:
        date_column = "snapshot_date"
    if date_column not in snapshot_view.columns:
        st.caption("스냅샷에 날짜 정보가 없어 KPI를 계산할 수 없습니다.")
        return pd.DataFrame()

    snapshot_view["date"] = pd.to_datetime(snapshot_view[date_column], errors="coerce").dt.normalize()
    snapshot_view = snapshot_view.dropna(subset=["date"])
    if snapshot_view.empty:
        st.caption("스냅샷에 유효한 날짜가 없어 KPI를 계산할 수 없습니다.")
        return pd.DataFrame()

    snapshot_view["center"] = snapshot_view["center"].astype(str)
    snapshot_view["resource_code"] = snapshot_view["resource_code"].astype(str)

    centers_list = [str(center) for center in centers if str(center).strip()]
    sku_list = [str(sku) for sku in skus if str(sku).strip()]
    centers_all = sorted(
        {
            str(center).strip()
            for center in snapshot_view["center"].unique()
            if str(center).strip()
        }
    )

    if not centers_list or not sku_list:
        st.caption("센터 또는 SKU 선택이 비어 있어 KPI를 계산할 수 없습니다.")
        return pd.DataFrame()

    filtered_snapshot = snapshot_view[
        snapshot_view["center"].isin(centers_list)
        & snapshot_view["resource_code"].isin(sku_list)
    ].copy()
    if filtered_snapshot.empty:
        st.caption("선택한 센터/SKU 조합에 해당하는 KPI 데이터가 없습니다.")
        return pd.DataFrame()

    global_latest_snapshot = snapshot_view["date"].max()
    if pd.isna(global_latest_snapshot):
        st.caption("최신 스냅샷 일자를 확인할 수 없어 KPI를 계산할 수 없습니다.")
        return pd.DataFrame()
    global_latest_snapshot_dt = pd.to_datetime(global_latest_snapshot).normalize()

    selected_latest_snapshot = filtered_snapshot["date"].max()
    if pd.isna(selected_latest_snapshot):
        st.caption("최신 스냅샷 일자를 확인할 수 없어 KPI를 계산할 수 없습니다.")
        return pd.DataFrame()

    if latest_snapshot is None or pd.isna(latest_snapshot):
        latest_snapshot_dt = pd.to_datetime(selected_latest_snapshot).normalize()
    else:
        latest_snapshot_dt = pd.to_datetime(latest_snapshot).normalize()

    latest_snapshot = latest_snapshot_dt

    if latest_snapshot is None or pd.isna(latest_snapshot):
        latest_snapshot_dt = pd.to_datetime(selected_latest_snapshot).normalize()
    else:
        latest_snapshot_dt = pd.to_datetime(latest_snapshot).normalize()

    latest_snapshot = latest_snapshot_dt

    name_map: Mapping[str, str] = {}
    if "resource_name" in filtered_snapshot.columns:
        name_rows = filtered_snapshot.dropna(subset=["resource_code", "resource_name"]).copy()
        if not name_rows.empty:
            name_rows["resource_code"] = name_rows["resource_code"].astype(str)
            name_rows["resource_name"] = name_rows["resource_name"].astype(str).str.strip()
            name_rows = name_rows[name_rows["resource_name"] != ""]
            if not name_rows.empty:
                name_map = dict(
                    name_rows.sort_values("date", ascending=False)[["resource_code", "resource_name"]]
                    .drop_duplicates(subset=["resource_code"])
                    .itertuples(index=False, name=None)
                )

    moves_view = moves.copy() if moves is not None else pd.DataFrame()
    moves_global = pd.DataFrame()
    if not moves_view.empty:
        if "carrier_mode" in moves_view.columns:
            moves_view["carrier_mode"] = moves_view["carrier_mode"].astype(str).str.upper()
        for column in ["resource_code", "to_center"]:
            if column in moves_view.columns:
                moves_view[column] = moves_view[column].astype(str)
        for column in ["inbound_date", "arrival_date", "onboard_date", "event_date"]:
            if column in moves_view.columns:
                moves_view[column] = pd.to_datetime(moves_view[column], errors="coerce")

        if "resource_code" in moves_view.columns:
            moves_view = moves_view[
                moves_view["resource_code"].isin(sku_list) | (moves_view["resource_code"] == "")
            ]
        moves_global = moves_view.copy()
        if "to_center" in moves_view.columns:
            moves_view = moves_view[
                moves_view["to_center"].isin(centers_list) | (moves_view["to_center"] == "")
            ]

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

    latest_snapshot_rows = filtered_snapshot[
        filtered_snapshot["date"] == latest_snapshot_dt
    ].copy()
    if "stock_qty" in latest_snapshot_rows.columns:
        latest_snapshot_rows["stock_qty"] = pd.to_numeric(
            latest_snapshot_rows["stock_qty"], errors="coerce"
        )

    global_snapshot_rows = snapshot_view[
        snapshot_view["date"] == global_latest_snapshot_dt
    ].copy()
    if "stock_qty" in global_snapshot_rows.columns:
        global_snapshot_rows["stock_qty"] = pd.to_numeric(
            global_snapshot_rows["stock_qty"], errors="coerce"
        )

    current_by_center = (
        latest_snapshot_rows.groupby(["resource_code", "center"])["stock_qty"].sum()
        if "stock_qty" in latest_snapshot_rows.columns
        else pd.Series(dtype=float)
    )
    current_totals = (
        current_by_center.groupby(level=0).sum()
        if not current_by_center.empty
        else pd.Series(dtype=float)
    )

    global_current_totals = (
        global_snapshot_rows.groupby("resource_code")["stock_qty"].sum()
        if "stock_qty" in global_snapshot_rows.columns and not global_snapshot_rows.empty
        else pd.Series(dtype=float)
    )

    daily_demand_series, total_demand_series = _extract_daily_demand(latest_snapshot_rows)

    in_transit_series, wip_series = _movement_breakdown_per_center(
        moves_view,
        centers_list,
        sku_list,
        today_dt,
        int(lag_days),
    )

    global_in_transit_series = pd.Series(dtype=float)
    global_wip_series = pd.Series(dtype=float)
    if centers_all:
        global_in_transit_series, global_wip_series = _movement_breakdown_per_center(
            moves_global,
            centers_all,
            sku_list,
            today_dt,
            int(lag_days),
        )

    global_in_transit_totals = (
        global_in_transit_series.groupby(level=0).sum()
        if not global_in_transit_series.empty
        else pd.Series(dtype=float)
    )
    global_wip_totals = (
        global_wip_series.groupby(level=0).sum()
        if not global_wip_series.empty
        else pd.Series(dtype=float)
    )

    _inject_responsive_styles()

    sku_cards_html: list[str] = []
    sku_min_width = max(280, int(1024 / max(chunk_size, 1))) if chunk_size else 320

    for sku in sku_list:
        display_name = name_map.get(sku, "") if isinstance(name_map, Mapping) else ""

        base_current = kpi_df.at[sku, "current"] if sku in kpi_df.index else 0
        total_current = int(current_totals.get(sku, base_current) or base_current)
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
        total_wip = int(kpi_df.at[sku, "wip"]) if sku in kpi_df.index else 0
        if not global_in_transit_totals.empty:
            transit_all_val = global_in_transit_totals.get(sku, float("nan"))
            total_transit_all = (
                int(round(transit_all_val))
                if pd.notna(transit_all_val)
                else total_transit
            )
        else:
            total_transit_all = total_transit

        if not global_wip_totals.empty:
            wip_all_val = global_wip_totals.get(sku, float("nan"))
            total_wip_all = (
                int(round(wip_all_val))
                if pd.notna(wip_all_val)
                else total_wip
            )
        else:
            total_wip_all = total_wip

        summary_cards = [
            _build_metric_card("전체 센터 재고 합계", _format_number(total_current_all)),
            _build_metric_card("선택 센터 재고 합계", _format_number(total_current)),
            _build_metric_card("전체 이동중 재고 합계", _format_number(total_transit_all)),
            _build_metric_card("전체 생산중 재고 합계", _format_number(total_wip_all)),
        ]
        summary_html = _build_grid(
            summary_cards,
            extra_class="kpi-grid--summary",
            min_width=220,
        )

        center_cards: list[str] = []
        for center in centers_list:
            center_current = (
                int(current_by_center.get((sku, center), 0)) if not current_by_center.empty else 0
            )
            center_transit = (
                int(in_transit_series.get((sku, center), 0)) if not in_transit_series.empty else 0
            )
            center_wip = int(wip_series.get((sku, center), 0)) if not wip_series.empty else 0
            center_demand = float(daily_demand_series.get((sku, center), float("nan")))
            center_depletion = center_depletion_map.get((sku, center), {})
            center_coverage = center_depletion.get("days_to_depletion")
            center_sellout_date = center_depletion.get("depletion_date")

            if center_coverage is None:
                center_coverage = _calculate_coverage_days(center_current, center_demand)
                center_sellout_date = _calculate_sellout_date(today_dt, center_coverage)
            center_cards.append(
                _build_center_card(
                    {
                        "center": center,
                        "current": center_current,
                        "in_transit": center_transit,
                        "wip": center_wip,
                        "coverage": center_coverage,
                        "sellout_date": center_sellout_date,
                        "show_in_transit": _should_show_in_transit(center, center_transit),
                    }
                )
            )

        center_cols, center_min_width, center_modifier = _center_grid_layout(len(center_cards))
        centers_html = _build_grid(
            center_cards,
            extra_class=f"kpi-grid--centers {center_modifier}".strip(),
            min_width=center_min_width,
            columns=center_cols,
            data_attrs={"data-center-count": len(center_cards)},
        )

        if display_name:
            title_html = (
                '<div class="kpi-sku-title">'
                f"{_escape(display_name)} "
                f'<span class="kpi-sku-code">{_escape(sku)}</span>'
                "</div>"
            )
        else:
            title_html = (
                '<div class="kpi-sku-title">'
                f'<span class="kpi-sku-code">{_escape(sku)}</span>'
                "</div>"
            )

        centers_section = (
            '<div class="kpi-section-title">센터별 상세</div>' + centers_html if centers_html else ""
        )

        sku_cards_html.append(
            '<div class="kpi-sku-card">'
            + title_html
            + summary_html
            + centers_section
            + "</div>"
        )

    cards_html = _build_grid(sku_cards_html, min_width=sku_min_width, extra_class="kpi-grid--sku")
    st.markdown(cards_html, unsafe_allow_html=True)
    st.caption(
        f"※ {pd.to_datetime(latest_snapshot).normalize():%Y-%m-%d} 스냅샷 기준 KPI이며, 현재 대표 시나리오 필터(센터/기간/SKU)가 반영되었습니다."
    )
    return kpi_df
