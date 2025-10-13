"""KPI rendering helpers for the Streamlit dashboard."""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import pandas as pd
import streamlit as st

from scm_dashboard_v5.analytics import kpi_breakdown_per_sku


def _chunked(items: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for idx in range(0, len(items), size):
        yield items[idx : idx + size]


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

    if latest_snapshot is None or pd.isna(latest_snapshot):
        latest_snapshot = filtered_snapshot["date"].max()
    if pd.isna(latest_snapshot):
        st.caption("최신 스냅샷 일자를 확인할 수 없어 KPI를 계산할 수 없습니다.")
        return pd.DataFrame()

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

    latest_snapshot_rows = filtered_snapshot[filtered_snapshot["date"] == latest_snapshot_dt].copy()
    if "stock_qty" in latest_snapshot_rows.columns:
        latest_snapshot_rows["stock_qty"] = pd.to_numeric(
            latest_snapshot_rows["stock_qty"], errors="coerce"
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

    daily_demand_series, total_demand_series = _extract_daily_demand(latest_snapshot_rows)

    in_transit_series, wip_series = _movement_breakdown_per_center(
        moves_view,
        centers_list,
        sku_list,
        today_dt,
        int(lag_days),
    )

    for group in _chunked(sku_list, max(1, chunk_size)):
        cols = st.columns(len(group))
        for idx, sku in enumerate(group):
            with cols[idx].container(border=True):
                display_name = name_map.get(sku, "") if isinstance(name_map, Mapping) else ""
                if display_name:
                    st.markdown(f"**{display_name}**  \\\n`{sku}`")
                else:
                    st.markdown(f"`{sku}`")

                base_current = kpi_df.at[sku, "current"] if sku in kpi_df.index else 0
                total_current = int(current_totals.get(sku, base_current) or base_current)
                total_transit = int(kpi_df.at[sku, "in_transit"]) if sku in kpi_df.index else 0
                total_wip = int(kpi_df.at[sku, "wip"]) if sku in kpi_df.index else 0
                total_demand = float(total_demand_series.get(sku, 0.0))
                total_coverage = _calculate_coverage_days(total_current, total_demand)
                total_sellout_date = _calculate_sellout_date(today_dt, total_coverage)

                summary_cols = st.columns(5)
                summary_cols[0].metric("전체 센터 재고", _format_number(total_current))
                summary_cols[1].metric("전체 이동중", _format_number(total_transit))
                summary_cols[2].metric("전체 생산중", _format_number(total_wip))
                summary_cols[3].metric("예상 소진일수", _format_days(total_coverage))
                summary_cols[4].metric("소진 예상일", _format_date(total_sellout_date))

                st.markdown("**센터별 상세**")
                center_cards: list[dict[str, object]] = []
                for center in centers_list:
                    center_current = (
                        int(current_by_center.get((sku, center), 0)) if not current_by_center.empty else 0
                    )
                    center_transit = (
                        int(in_transit_series.get((sku, center), 0)) if not in_transit_series.empty else 0
                    )
                    center_wip = int(wip_series.get((sku, center), 0)) if not wip_series.empty else 0
                    center_demand = float(daily_demand_series.get((sku, center), float("nan")))
                    center_coverage = _calculate_coverage_days(center_current, center_demand)
                    center_sellout_date = _calculate_sellout_date(today_dt, center_coverage)
                    center_cards.append(
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

                for center_group in _chunked(center_cards, 2):
                    center_cols = st.columns(len(center_group))
                    for c_idx, center_info in enumerate(center_group):
                        with center_cols[c_idx].container(border=True):
                            st.markdown(f"**{center_info['center']}**")
                            metrics_row = st.columns(3)
                            metrics_row[0].metric("재고", _format_number(center_info["current"]))
                            if center_info["show_in_transit"]:
                                metrics_row[1].metric("이동중", _format_number(center_info["in_transit"]))
                            else:
                                metrics_row[1].metric("이동중", "-")
                            metrics_row[2].metric("생산중", _format_number(center_info["wip"]))

                            metrics_row2 = st.columns(2)
                            metrics_row2[0].metric("예상 소진일수", _format_days(center_info["coverage"]))
                            metrics_row2[1].metric("소진 예상일", _format_date(center_info["sellout_date"]))

    st.caption(
        f"※ {pd.to_datetime(latest_snapshot).normalize():%Y-%m-%d} 스냅샷 기준 KPI이며, 현재 대표 시나리오 필터(센터/기간/SKU)가 반영되었습니다."
    )
    return kpi_df
