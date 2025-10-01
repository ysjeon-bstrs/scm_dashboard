"""Timeline generation logic for the SCM dashboard."""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd


def build_timeline(
    snap_long: pd.DataFrame,
    moves: pd.DataFrame,
    centers_sel: List[str],
    skus_sel: List[str],
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
    horizon_days: int = 0,
    today: Optional[pd.Timestamp] = None,
    lag_days: int = 7,
) -> pd.DataFrame:
    if today is None:
        today = pd.Timestamp.today().normalize()
    horizon_end = end_dt + pd.Timedelta(days=horizon_days)
    full_dates = pd.date_range(start_dt, horizon_end, freq="D")

    mv_all = moves.copy()

    base = snap_long[
        snap_long["center"].isin(centers_sel) & snap_long["resource_code"].isin(skus_sel)
    ].copy().rename(columns={"snapshot_date": "date"})
    if base.empty:
        return pd.DataFrame(columns=["date", "center", "resource_code", "stock_qty"])
    base = base[(base["date"] >= start_dt) & (base["date"] <= end_dt)]

    lines: List[pd.DataFrame] = []

    for (ct, sku), grp in base.groupby(["center", "resource_code"]):
        grp = grp.sort_values("date")
        last_dt = grp["date"].max()

        if horizon_days > 0:
            proj_dates = pd.date_range(last_dt + pd.Timedelta(days=1), horizon_end, freq="D")
            proj_df = pd.DataFrame({"date": proj_dates, "center": ct, "resource_code": sku, "stock_qty": np.nan})
            ts = pd.concat([grp[["date", "center", "resource_code", "stock_qty"]], proj_df], ignore_index=True)
        else:
            ts = grp[["date", "center", "resource_code", "stock_qty"]].copy()

        ts = ts.sort_values("date")
        ts["stock_qty"] = ts["stock_qty"].ffill()

        mv = mv_all[mv_all["resource_code"] == sku].copy()

        eff_minus = (
            mv[(mv["from_center"].astype(str) == str(ct)) & (mv["onboard_date"].notna()) & (mv["onboard_date"] > last_dt)]
            .groupby("onboard_date", as_index=False)["qty_ea"].sum()
            .rename(columns={"onboard_date": "date", "qty_ea": "delta"})
        )
        eff_minus["delta"] *= -1

        mv_center = mv[(mv["to_center"].astype(str) == str(ct))].copy()
        if not mv_center.empty:
            pred_inbound = pd.Series(pd.NaT, index=mv_center.index, dtype="datetime64[ns]")
            mask_inb = mv_center["inbound_date"].notna()
            pred_inbound.loc[mask_inb] = mv_center.loc[mask_inb, "inbound_date"]
            mask_arr = (~mask_inb) & mv_center["arrival_date"].notna()
            if mask_arr.any():
                past_arr = mask_arr & (mv_center["arrival_date"] <= today)
                pred_inbound.loc[past_arr] = mv_center.loc[past_arr, "arrival_date"] + pd.Timedelta(days=int(lag_days))
                fut_arr = mask_arr & (mv_center["arrival_date"] > today)
                pred_inbound.loc[fut_arr] = mv_center.loc[fut_arr, "arrival_date"]
            mv_center["pred_inbound_date"] = pred_inbound
            eff_plus = (
                mv_center[(mv_center["pred_inbound_date"].notna()) & (mv_center["pred_inbound_date"] > last_dt)]
                .groupby("pred_inbound_date", as_index=False)["qty_ea"].sum()
                .rename(columns={"pred_inbound_date": "date", "qty_ea": "delta"})
            )
        else:
            eff_plus = pd.DataFrame(columns=["date", "delta"])

        eff_all = pd.concat([eff_minus, eff_plus], ignore_index=True)
        if not eff_all.empty:
            delta_series = eff_all.groupby("date")["delta"].sum()
            delta_series = delta_series.reindex(ts["date"], fill_value=0).fillna(0)
            for date, delta in delta_series.items():
                if delta != 0:
                    ts.loc[ts["date"] >= date, "stock_qty"] = ts.loc[ts["date"] >= date, "stock_qty"] + delta

        ts["stock_qty"] = ts["stock_qty"].fillna(0).replace([np.inf, -np.inf], 0).clip(lower=0)
        lines.append(ts)

        wip_complete = moves[
            (moves["resource_code"] == sku)
            & (moves["carrier_mode"].astype(str).str.upper() == "WIP")
            & (moves["to_center"] == ct)
            & (moves["event_date"].notna())
        ].copy()
        if not wip_complete.empty:
            wip_add = (
                wip_complete.groupby("event_date", as_index=False)["qty_ea"].sum().rename(columns={"event_date": "date", "qty_ea": "delta"})
            )
            wip_delta_series = wip_add.groupby("date")["delta"].sum()
            wip_delta_series = wip_delta_series.reindex(ts["date"], fill_value=0).fillna(0)
            for date, delta in wip_delta_series.items():
                if delta != 0:
                    ts.loc[ts["date"] >= date, "stock_qty"] = ts.loc[ts["date"] >= date, "stock_qty"] + delta
            ts["stock_qty"] = ts["stock_qty"].fillna(0).replace([np.inf, -np.inf], 0).clip(lower=0)
            lines[-1] = ts

    moves_str = mv_all.copy()
    moves_str["from_center"] = moves_str["from_center"].astype(str)
    moves_str["to_center"] = moves_str["to_center"].astype(str)
    moves_str["carrier_mode"] = moves_str["carrier_mode"].astype(str).str.upper()

    mv_sel = moves_str[
        moves_str["resource_code"].isin(skus_sel)
        & (
            moves_str["from_center"].isin(centers_sel)
            | moves_str["to_center"].isin(centers_sel)
            | (moves_str["carrier_mode"] == "WIP")
        )
    ]

    for sku, g in mv_sel.groupby("resource_code"):
        g_nonwip = g[g["carrier_mode"] != "WIP"]
        if not g_nonwip.empty:
            g_selected = g_nonwip[g_nonwip["to_center"].isin(centers_sel)]
            if not g_selected.empty:
                idx = pd.date_range(start_dt, horizon_end, freq="D")
                today_norm = (today or pd.Timestamp.today()).normalize()

                end_eff = pd.Series(pd.NaT, index=g_selected.index, dtype="datetime64[ns]")

                mask_inb = g_selected["inbound_date"].notna()
                end_eff.loc[mask_inb] = g_selected.loc[mask_inb, "inbound_date"]

                mask_arr = (~mask_inb) & g_selected["arrival_date"].notna()
                if mask_arr.any():
                    past_arr = mask_arr & (g_selected["arrival_date"] <= today_norm)
                    end_eff.loc[past_arr] = g_selected.loc[past_arr, "arrival_date"] + pd.Timedelta(days=int(lag_days))

                    fut_arr = mask_arr & (g_selected["arrival_date"] > today_norm)
                    end_eff.loc[fut_arr] = g_selected.loc[fut_arr, "arrival_date"]

                end_eff = end_eff.fillna(min(today_norm + pd.Timedelta(days=1), idx[-1] + pd.Timedelta(days=1)))

                g_selected_with_end = g_selected.copy()
                g_selected_with_end["end_date"] = end_eff

                starts = g_selected_with_end.dropna(subset=["onboard_date"]).groupby("onboard_date")["qty_ea"].sum()
                ends = g_selected_with_end.groupby("end_date")["qty_ea"].sum() * -1

                delta = (
                    starts.rename_axis("date").to_frame("delta").add(ends.rename_axis("date").to_frame("delta"), fill_value=0)["delta"].sort_index()
                )

                s = delta.reindex(idx, fill_value=0).cumsum().clip(lower=0)

                carry_mask = (
                    g_selected["onboard_date"].notna()
                    & (g_selected["onboard_date"] < idx[0])
                    & (end_eff > idx[0])
                )
                carry = int(g_selected.loc[carry_mask, "qty_ea"].sum())
                if carry:
                    s = (s + carry).clip(lower=0)

                if s.any():
                    lines.append(
                        pd.DataFrame(
                            {
                                "date": s.index,
                                "center": "In-Transit",
                                "resource_code": sku,
                                "stock_qty": s.values.astype(int),
                            }
                        )
                    )

        g_wip = g[g["carrier_mode"] == "WIP"]
        if not g_wip.empty:
            s = pd.Series(0, index=pd.to_datetime(full_dates))

            add_onboard = (
                g_wip[g_wip["onboard_date"].notna()]
                .groupby("onboard_date", as_index=False)["qty_ea"].sum()
                .rename(columns={"onboard_date": "date", "qty_ea": "delta"})
            )
            add_event = (
                g_wip[g_wip["event_date"].notna()]
                .groupby("event_date", as_index=False)["qty_ea"].sum()
                .rename(columns={"event_date": "date", "qty_ea": "delta"})
            )
            add_event["delta"] *= -1
            deltas = pd.concat([add_onboard, add_event], ignore_index=True)

            if not deltas.empty:
                delta_series = deltas.groupby("date")["delta"].sum()
                delta_series = delta_series.reindex(s.index, fill_value=0).fillna(0)
                for date, delta in delta_series.items():
                    if delta != 0:
                        s.loc[s.index >= date] = s.loc[s.index >= date] + delta

                vdf = pd.DataFrame({"date": s.index, "center": "WIP", "resource_code": sku, "stock_qty": s.values})
                vdf["stock_qty"] = vdf["stock_qty"].fillna(0).replace([np.inf, -np.inf], 0).clip(lower=0)
                lines.append(vdf)

    if not lines:
        return pd.DataFrame(columns=["date", "center", "resource_code", "stock_qty"])

    out = pd.concat(lines, ignore_index=True)
    out = out[(out["date"] >= start_dt) & (out["date"] <= horizon_end)]

    out["stock_qty"] = pd.to_numeric(out["stock_qty"], errors="coerce")
    out["stock_qty"] = out["stock_qty"].fillna(0)
    out["stock_qty"] = out["stock_qty"].replace([np.inf, -np.inf], 0)
    out["stock_qty"] = out["stock_qty"].clip(lower=0).astype(int)

    return out
