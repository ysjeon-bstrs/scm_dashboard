"""KPI 메트릭 계산 모듈.

재고 소진, 일평균 수요, 센터별 입출고 분석 등을 제공합니다.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import pandas as pd

from scm_dashboard_v9.analytics import kpi_breakdown_per_sku
from scm_dashboard_v9.core import build_timeline as build_core_timeline
from scm_dashboard_v9.forecast import apply_consumption_with_events

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




def extract_daily_demand(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
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




def movement_breakdown_per_center(
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



