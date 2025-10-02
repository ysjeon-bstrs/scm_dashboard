"""KPI helpers for the SCM dashboard."""

from __future__ import annotations

from typing import List

import pandas as pd


def kpi_breakdown_per_sku(
    snap_long: pd.DataFrame,
    moves: pd.DataFrame,
    centers_sel: List[str],
    skus_sel: List[str],
    today: pd.Timestamp,
    snap_date_col: str,
    latest_snapshot: pd.Timestamp,
    lag_days: int,
) -> pd.DataFrame:
    cur = (
        snap_long[
            (snap_long[snap_date_col] == latest_snapshot)
            & (snap_long["center"].isin(centers_sel))
            & (snap_long["resource_code"].astype(str).isin(skus_sel))
        ].groupby("resource_code", as_index=True)["stock_qty"].sum()
    )

    mv_kpi = moves.copy()
    if not mv_kpi.empty:
        pred_end = pd.Series(pd.NaT, index=mv_kpi.index, dtype="datetime64[ns]")

        mask_inb = mv_kpi["inbound_date"].notna()
        pred_end.loc[mask_inb] = mv_kpi.loc[mask_inb, "inbound_date"]

        mask_arr = (~mask_inb) & mv_kpi["arrival_date"].notna()
        if mask_arr.any():
            past_arr = mask_arr & (mv_kpi["arrival_date"] <= today)
            pred_end.loc[past_arr] = mv_kpi.loc[past_arr, "arrival_date"] + pd.Timedelta(days=int(lag_days))

            fut_arr = mask_arr & (mv_kpi["arrival_date"] > today)
            pred_end.loc[fut_arr] = mv_kpi.loc[fut_arr, "arrival_date"]

        pred_end = pred_end.fillna(today + pd.Timedelta(days=1))
        mv_kpi["pred_end_date"] = pred_end

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
