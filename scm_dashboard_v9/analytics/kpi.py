"""KPI 계산 함수들."""

from __future__ import annotations

from typing import Sequence

import pandas as pd


def kpi_breakdown_per_sku(
    snap_long: pd.DataFrame,
    moves: pd.DataFrame,
    centers_sel: Sequence[str],
    skus_sel: Sequence[str],
    today: pd.Timestamp,
    snap_date_col: str,
    latest_snapshot: pd.Timestamp,
    lag_days: int,
) -> pd.DataFrame:
    """SKU별로 현재 재고, 입고 예정, WIP를 계산합니다.

    Args:
        snap_long: 스냅샷 long-format DataFrame
        moves: 이동(moves) DataFrame
        centers_sel: 대상 센터 리스트
        skus_sel: 대상 SKU 리스트
        today: 기준 날짜
        snap_date_col: 스냅샷 날짜 컬럼명
        latest_snapshot: 최신 스냅샷 날짜
        lag_days: 입고 지연 일수

    Returns:
        current, in_transit, wip 컬럼을 가진 DataFrame (index: resource_code)
    """
    # 현재 재고 계산 - 센터별 최신 날짜 사용 (센터별 스냅샷 생성 시간 차이 대응)
    if snap_long.empty:
        cur = pd.Series(dtype=float, name="resource_code")
    else:
        # 각 센터의 최신 날짜 데이터를 개별적으로 수집
        snapshot_parts = []
        for center in centers_sel:
            center_data = snap_long[
                (snap_long["center"] == center)
                & (snap_long["resource_code"].astype(str).isin(skus_sel))
            ]
            if center_data.empty:
                continue
            center_latest_date = center_data[snap_date_col].max()
            if pd.isna(center_latest_date):
                continue
            center_latest_data = center_data[center_data[snap_date_col] == center_latest_date]
            snapshot_parts.append(center_latest_data)

        if snapshot_parts:
            latest_snapshot_data = pd.concat(snapshot_parts, ignore_index=True)
            cur = latest_snapshot_data.groupby("resource_code", as_index=True)["stock_qty"].sum()
        else:
            cur = pd.Series(dtype=float, name="resource_code")

    # moves 복사 및 pred_end_date 계산
    # WIP: event_date 그대로, In-Transit: 리드타임 적용 (과거 arrival은 오늘+3일)
    mv_kpi = moves.copy()
    if not mv_kpi.empty:
        pred_end = pd.Series(pd.NaT, index=mv_kpi.index, dtype="datetime64[ns]")

        # carrier_mode 확인
        carrier_mode = mv_kpi.get("carrier_mode", pd.Series("", index=mv_kpi.index))
        is_wip = carrier_mode.astype(str).str.upper() == "WIP"

        # inbound_date가 있으면 우선 사용
        if "inbound_date" in mv_kpi.columns:
            mask_inb = mv_kpi["inbound_date"].notna()
            pred_end.loc[mask_inb] = mv_kpi.loc[mask_inb, "inbound_date"]
        else:
            mask_inb = pd.Series(False, index=mv_kpi.index)

        # WIP: event_date 그대로 사용
        wip_mask = is_wip & (~mask_inb)
        if wip_mask.any() and "event_date" in mv_kpi.columns:
            event_date_series = pd.to_datetime(mv_kpi["event_date"], errors="coerce")
            wip_with_event = wip_mask & event_date_series.notna()
            if wip_with_event.any():
                pred_end.loc[wip_with_event] = event_date_series.loc[wip_with_event]

        # In-Transit: arrival + 리드타임
        intransit_mask = (~is_wip) & (~mask_inb)
        if "arrival_date" in mv_kpi.columns:
            mask_arr = intransit_mask & mv_kpi["arrival_date"].notna()
            if mask_arr.any():
                # 과거 arrival: today + 3일
                past_arr = mask_arr & (mv_kpi["arrival_date"] <= today)
                pred_end.loc[past_arr] = today + pd.Timedelta(days=3)

                # 미래 arrival: arrival + lag_days
                fut_arr = mask_arr & (mv_kpi["arrival_date"] > today)
                pred_end.loc[fut_arr] = mv_kpi.loc[fut_arr, "arrival_date"] + pd.Timedelta(days=int(lag_days))

        pred_end = pred_end.fillna(today + pd.Timedelta(days=1))
        mv_kpi["pred_end_date"] = pred_end

    # In-Transit 계산
    it = (
        mv_kpi[
            (mv_kpi["carrier_mode"] != "WIP")
            & (mv_kpi["to_center"].isin(centers_sel))
            & (mv_kpi["resource_code"].isin(skus_sel))
            & (mv_kpi["onboard_date"].notna())
            & (mv_kpi["onboard_date"] <= today)
            & (today < mv_kpi["pred_end_date"])
        ].groupby("resource_code", as_index=True)["qty_ea"].sum()
    )

    # WIP 계산
    w = mv_kpi[
        (mv_kpi["carrier_mode"] == "WIP")
        & (mv_kpi["to_center"].isin(centers_sel))
        & (mv_kpi["resource_code"].isin(skus_sel))
    ].copy()
    if w.empty:
        wip = pd.Series(0, index=pd.Index(skus_sel, name="resource_code"))
    else:
        add = w.dropna(subset=["onboard_date"]).set_index(["resource_code", "onboard_date"])["qty_ea"]
        rem = w.dropna(subset=["event_date"]).set_index(["resource_code", "event_date"])["qty_ea"] * -1
        flow = pd.concat([add, rem]).groupby(level=[0, 1]).sum()
        flow = flow[flow.index.get_level_values(1) <= today]
        wip = flow.groupby(level=0).cumsum().groupby(level=0).last().clip(lower=0)

    out = pd.DataFrame({"current": cur, "in_transit": it, "wip": wip}).reindex(skus_sel).fillna(0).astype(int)
    return out
