"""Timeline generation logic for the SCM dashboard."""

from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np
import pandas as pd


DATE_COLUMNS = ("onboard_date", "arrival_date", "inbound_date", "event_date")


def normalize_move_dates(moves: pd.DataFrame, columns: Iterable[str] = DATE_COLUMNS) -> pd.DataFrame:
    """Return a copy of *moves* with the specified date columns normalised to midnight."""

    out = moves.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce").dt.normalize()
    return out


def annotate_move_schedule(
    moves: pd.DataFrame,
    today: pd.Timestamp,
    lag_days: int,
    horizon_end: pd.Timestamp,
    fallback_days: int = 1,
) -> pd.DataFrame:
    """Attach predicted inbound dates aligned with the centre inventory policy.

    The policy is:
    * Prefer the actual inbound completion date when available.
    * Otherwise fall back to the arrival/ETA date. Past arrivals stay in transit
      for ``lag_days`` after arrival to mirror receipt delays; future ETAs
      convert on the ETA itself.
    * Rows without any milestone drop on ``today + fallback_days`` (capped to
      the chart horizon) so they do not block the forecast indefinitely.
    """

    today_norm = pd.to_datetime(today).normalize()
    fallback_date = min(today_norm + pd.Timedelta(days=int(fallback_days)), horizon_end + pd.Timedelta(days=1))

    out = moves.copy()
    out["carrier_mode"] = out.get("carrier_mode", "").astype(str).str.upper()

    pred = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns]")


    has_inbound = out["inbound_date"].notna() if "inbound_date" in out else pd.Series(False, index=out.index)
    pred.loc[has_inbound] = out.loc[has_inbound, "inbound_date"]


    if "arrival_date" in out:
        arrival_col = out["arrival_date"]
    else:
        arrival_col = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns]")

    has_arrival = (~has_inbound) & arrival_col.notna()
    if has_arrival.any():

        arr_dates = arrival_col
        # Past arrivals remain in transit until the lagged receipt date.

        past_arrival = has_arrival & (arr_dates <= today_norm)
        if past_arrival.any():
            pred.loc[past_arrival] = out.loc[past_arrival, "arrival_date"] + pd.Timedelta(days=int(lag_days))
        # Future ETAs release inventory on the ETA itself.
        future_arrival = has_arrival & (arr_dates > today_norm)
        if future_arrival.any():
            pred.loc[future_arrival] = out.loc[future_arrival, "arrival_date"]

    # Shipments without any milestone fall back to a policy date (default: today + 1 day).
    pred = pred.fillna(fallback_date)
    out["pred_inbound_date"] = pd.to_datetime(pred).dt.normalize()
    out["pred_inbound_date"] = out["pred_inbound_date"].clip(upper=horizon_end + pd.Timedelta(days=1))
    out["in_transit_end_date"] = out["pred_inbound_date"]

    return out



def build_timeline(
    snap_long: pd.DataFrame,

    moves: pd.DataFrame,
    centers_sel: Iterable[str],
    skus_sel: Iterable[str],
    start_dt: pd.Timestamp,
    horizon_end: pd.Timestamp,
    today: pd.Timestamp,
    lag_days: int = 7,
) -> pd.DataFrame:
    """Build an in-transit daily timeseries synchronised with inventory receipts.

    The output provides a daily step series per SKU whose decrements align
    exactly with the receipt dates returned by :func:`annotate_move_schedule`.

    Note
    ----
    The Streamlit dashboard no longer renders this series directly in the
    inventory step chart, but the helper remains available for data export and
    offline diagnostics.
    """
    centers = {str(c) for c in centers_sel}
    skus = set(skus_sel)
    today_norm = pd.to_datetime(today).normalize()

    prepared = normalize_move_dates(moves)
    prepared = annotate_move_schedule(prepared, today_norm, lag_days, horizon_end)

    prepared = prepared[prepared.get("carrier_mode", "").astype(str).str.upper() != "WIP"].copy()
    if prepared.empty:
        return pd.DataFrame(columns=["date", "center", "resource_code", "stock_qty"])

    prepared = prepared[
        prepared["resource_code"].isin(skus)
        & prepared["to_center"].astype(str).isin(centers)
        & prepared["onboard_date"].notna()
    ].copy()
    if prepared.empty:
        return pd.DataFrame(columns=["date", "center", "resource_code", "stock_qty"])

    mv_all = normalize_move_dates(moves.copy())
    mv_all = annotate_move_schedule(mv_all, today, lag_days, horizon_end)


    mv_all = normalize_move_dates(moves.copy())
    mv_all = annotate_move_schedule(mv_all, today, lag_days, horizon_end)


    if not series_frames:
        return pd.DataFrame(columns=["date", "center", "resource_code", "stock_qty"])

    result = pd.concat(series_frames, ignore_index=True)
    return result


def generate_timeline(
    moves: pd.DataFrame,
    capacity: pd.DataFrame,
    mv_all: pd.DataFrame,
    product_master: pd.DataFrame,
    skus_sel: Iterable[str],
    centers_sel: Iterable[str],
    start_dt: pd.Timestamp,
    horizon_end: pd.Timestamp,
    today: pd.Timestamp,
    lag_days: int = 7,
) -> pd.DataFrame:
    """Merge capacity plan, transit data and WIP positions into a unified timeline.

    Returns
    -------
    pd.DataFrame
        Columns: date, center, resource_code, stock_qty
    """
    full_dates = pd.date_range(start_dt, horizon_end, freq="D")
    capacity = capacity.copy()
    capacity["date"] = pd.to_datetime(capacity["date"]).dt.normalize()
    capacity["center"] = capacity["center"].astype(str)

    capacity_filt = capacity[
        capacity["resource_code"].isin(skus_sel)
        & capacity["center"].isin(centers_sel)
        & (capacity["date"] >= start_dt)
        & (capacity["date"] <= horizon_end)
    ].copy()

    if capacity_filt.empty:
        cap_rows = pd.DataFrame(columns=["date", "center", "resource_code", "stock_qty"])
    else:
        cap_rows = capacity_filt[["date", "center", "resource_code", "stock_qty"]].copy()

    lines: List[pd.DataFrame] = []
    if not cap_rows.empty:
        lines.append(cap_rows)

    if moves.empty and mv_all.empty:
        if not lines:
            return pd.DataFrame(columns=["date", "center", "resource_code", "stock_qty"])
        out = pd.concat(lines, ignore_index=True)
        out = out[(out["date"] >= start_dt) & (out["date"] <= horizon_end)]

        out["stock_qty"] = pd.to_numeric(out["stock_qty"], errors="coerce")
        out["stock_qty"] = out["stock_qty"].fillna(0)
        out["stock_qty"] = out["stock_qty"].replace([np.inf, -np.inf], 0)
        out["stock_qty"] = out["stock_qty"].clip(lower=0).astype(int)


        eff_minus = (
            mv[(mv["from_center"].astype(str) == str(ct)) & (mv["onboard_date"].notna()) & (mv["onboard_date"] > last_dt)]
            .groupby("onboard_date", as_index=False)["qty_ea"].sum()
            .rename(columns={"onboard_date": "date", "qty_ea": "delta"})
        )
        eff_minus["delta"] *= -1

        mv_center = mv[(mv["to_center"].astype(str) == str(ct))].copy()
        if not mv_center.empty:
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