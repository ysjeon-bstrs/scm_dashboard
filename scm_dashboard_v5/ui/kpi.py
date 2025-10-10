"""KPI rendering helpers for the Streamlit dashboard."""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import pandas as pd
import streamlit as st

from scm_dashboard_v5.analytics import kpi_breakdown_per_sku


def _chunked(items: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for idx in range(0, len(items), size):
        yield items[idx : idx + size]


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

    for group in _chunked(sku_list, max(1, chunk_size)):
        cols = st.columns(len(group))
        for idx, sku in enumerate(group):
            with cols[idx].container(border=True):
                display_name = name_map.get(sku, "") if isinstance(name_map, Mapping) else ""
                if display_name:
                    st.markdown(f"**{display_name}**  \\\n`{sku}`")
                else:
                    st.markdown(f"`{sku}`")
                c1, c2, c3 = st.columns(3)
                current_val = int(kpi_df.at[sku, "current"]) if sku in kpi_df.index else 0
                transit_val = int(kpi_df.at[sku, "in_transit"]) if sku in kpi_df.index else 0
                wip_val = int(kpi_df.at[sku, "wip"]) if sku in kpi_df.index else 0
                c1.metric("현재 재고", f"{current_val:,}")
                c2.metric("이동중", f"{transit_val:,}")
                c3.metric("생산중", f"{wip_val:,}")

    st.caption(
        f"※ {pd.to_datetime(latest_snapshot).normalize():%Y-%m-%d} 스냅샷 기준 KPI이며, 현재 대표 시나리오 필터(센터/기간/SKU)가 반영되었습니다."
    )
    return kpi_df
