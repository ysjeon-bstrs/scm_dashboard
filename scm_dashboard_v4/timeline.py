"""Timeline generation logic for the SCM dashboard."""

from __future__ import annotations

from typing import Iterable

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
    """Attach predicted inbound dates aligned with the centre inventory policy."""

    today_norm = pd.to_datetime(today).normalize()
    fallback_date = min(today_norm + pd.Timedelta(days=int(fallback_days)), horizon_end + pd.Timedelta(days=1))

    out = moves.copy()
    out["carrier_mode"] = out.get("carrier_mode", "").astype(str).str.upper()
    actual_onboard = pd.to_datetime(out.get("onboard_date"), errors="coerce").dt.normalize()
    out["_onboard_date_actual"] = actual_onboard

    pred = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns]")

    inbound_col = out.get("inbound_date")
    has_inbound = inbound_col.notna() if inbound_col is not None else pd.Series(False, index=out.index)
    if has_inbound.any():
        pred.loc[has_inbound] = inbound_col.loc[has_inbound]

    if "arrival_date" in out:
        arrival_col = out["arrival_date"]
    else:
        arrival_col = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns]")

    has_arrival = (~has_inbound) & arrival_col.notna()
    if has_arrival.any():
        arr_dates = arrival_col
        past_arrival = has_arrival & (arr_dates <= today_norm)
        if past_arrival.any():
            pred.loc[past_arrival] = out.loc[past_arrival, "arrival_date"] + pd.Timedelta(days=int(lag_days))

        future_arrival = has_arrival & (arr_dates > today_norm)
        if future_arrival.any():
            pred.loc[future_arrival] = out.loc[future_arrival, "arrival_date"]

    pred = pred.fillna(fallback_date)
    out["pred_inbound_date"] = pd.to_datetime(pred).dt.normalize()
    out["pred_inbound_date"] = out["pred_inbound_date"].clip(upper=horizon_end + pd.Timedelta(days=1))
    out["in_transit_end_date"] = out["pred_inbound_date"]

    return out


def compute_in_transit_series(
    moves: pd.DataFrame,
    centers_sel: Iterable[str],
    skus_sel: Iterable[str],
    start_dt: pd.Timestamp,
    horizon_end: pd.Timestamp,
    today: pd.Timestamp,
    lag_days: int = 7,
) -> pd.DataFrame:
    """Return the non-WIP in-transit series grouped by SKU."""

    if moves is None or moves.empty:
        return pd.DataFrame(columns=["date", "center", "resource_code", "stock_qty"])

    start_dt = pd.to_datetime(start_dt).normalize()
    horizon_end = pd.to_datetime(horizon_end).normalize()
    today_norm = pd.to_datetime(today).normalize()

    centers = {str(c) for c in centers_sel}
    skus = {str(s) for s in skus_sel}
    if not centers or not skus:
        return pd.DataFrame(columns=["date", "center", "resource_code", "stock_qty"])

    prepared = normalize_move_dates(moves)
    prepared = annotate_move_schedule(prepared, today_norm, lag_days, horizon_end)
    prepared["carrier_mode"] = prepared.get("carrier_mode", "").astype(str).str.upper()
    prepared["resource_code"] = prepared.get("resource_code", "").astype(str)
    prepared["to_center"] = prepared.get("to_center", "").astype(str)
    prepared["from_center"] = prepared.get("from_center", "").astype(str)
    prepared["qty_ea"] = pd.to_numeric(prepared.get("qty_ea", 0), errors="coerce").fillna(0)
    start_col = "_onboard_date_actual" if "_onboard_date_actual" in prepared.columns else "onboard_date"

    filtered = prepared[
        (prepared["carrier_mode"] != "WIP")
        & prepared["resource_code"].isin(skus)
        & prepared["to_center"].isin(centers)
    ].copy()

    if filtered.empty:
        return pd.DataFrame(columns=["date", "center", "resource_code", "stock_qty"])

    idx = pd.date_range(start_dt, horizon_end, freq="D")
    lines = []
    for sku, grp in filtered.groupby("resource_code"):
        starts = grp.dropna(subset=[start_col]).groupby(start_col)["qty_ea"].sum()
        ends = grp.dropna(subset=["in_transit_end_date"]).groupby("in_transit_end_date")["qty_ea"].sum() * -1
        delta = (
            starts.rename_axis("date").to_frame("delta")
            .add(ends.rename_axis("date").to_frame("delta"), fill_value=0)
            ["delta"].sort_index()
        )
        delta = delta.reindex(idx, fill_value=0.0)
        series = delta.cumsum().clip(lower=0)

        carry_mask = (
            grp[start_col].notna()
            & (grp[start_col] < idx[0])
            & (grp["in_transit_end_date"].fillna(horizon_end + pd.Timedelta(days=1)) > idx[0])
        )
        carry = int(grp.loc[carry_mask, "qty_ea"].sum())
        if carry:
            series = (series + carry).clip(lower=0)

        if series.any():
            lines.append(
                pd.DataFrame(
                    {
                        "date": series.index,
                        "center": "In-Transit",
                        "resource_code": sku,
                        "stock_qty": series.values,
                    }
                )
            )

    if not lines:
        return pd.DataFrame(columns=["date", "center", "resource_code", "stock_qty"])

    out = pd.concat(lines, ignore_index=True)
    out["stock_qty"] = out["stock_qty"].round().astype(int)
    return out


def _compute_wip_series(
    moves: pd.DataFrame,
    skus_sel: Iterable[str],
    start_dt: pd.Timestamp,
    horizon_end: pd.Timestamp,
) -> pd.DataFrame:
    if moves is None or moves.empty:
        return pd.DataFrame(columns=["date", "center", "resource_code", "stock_qty"])

    start_dt = pd.to_datetime(start_dt).normalize()
    horizon_end = pd.to_datetime(horizon_end).normalize()

    skus = {str(s) for s in skus_sel}
    moves = moves.copy()
    moves["carrier_mode"] = moves.get("carrier_mode", "").astype(str).str.upper()
    moves["resource_code"] = moves.get("resource_code", "").astype(str)
    moves["qty_ea"] = pd.to_numeric(moves.get("qty_ea", 0), errors="coerce").fillna(0)

    start_col = "_onboard_date_actual" if "_onboard_date_actual" in moves.columns else "onboard_date"

    wip = moves[(moves["carrier_mode"] == "WIP") & moves["resource_code"].isin(skus)]
    if wip.empty:
        return pd.DataFrame(columns=["date", "center", "resource_code", "stock_qty"])

    idx = pd.date_range(start_dt, horizon_end, freq="D")
    lines = []
    for sku, grp in wip.groupby("resource_code"):
        deltas = []
        onboard = (
            grp[grp[start_col].notna()]
            .groupby(start_col, as_index=False)["qty_ea"].sum()
            .rename(columns={start_col: "date", "qty_ea": "delta"})
        )
        if not onboard.empty:
            deltas.append(onboard)

        if "event_date" in grp.columns:
            events = (
                grp[grp["event_date"].notna()]
                .groupby("event_date", as_index=False)["qty_ea"].sum()
                .rename(columns={"event_date": "date", "qty_ea": "delta"})
            )
            if not events.empty:
                events["delta"] *= -1
                deltas.append(events)

        if not deltas:
            continue

        delta = pd.concat(deltas, ignore_index=True)
        delta_series = delta.groupby("date")["delta"].sum().reindex(idx, fill_value=0.0)
        series = delta_series.cumsum().clip(lower=0)
        if not series.any():
            continue

        lines.append(
            pd.DataFrame(
                {
                    "date": series.index,
                    "center": "WIP",
                    "resource_code": sku,
                    "stock_qty": series.values,
                }
            )
        )

    if not lines:
        return pd.DataFrame(columns=["date", "center", "resource_code", "stock_qty"])

    out = pd.concat(lines, ignore_index=True)
    out["stock_qty"] = out["stock_qty"].round().astype(int)
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
    """Build an in-transit daily timeseries synchronised with inventory receipts."""

    today_norm = pd.to_datetime(today).normalize()
    start_dt = pd.to_datetime(start_dt).normalize()
    horizon_end = pd.to_datetime(horizon_end).normalize()

    centers = {str(c) for c in centers_sel}
    skus = {str(s) for s in skus_sel}
    if not centers or not skus:
        return pd.DataFrame(columns=["date", "center", "resource_code", "stock_qty"])

    snap_cols = {c.lower(): c for c in snap_long.columns}
    date_col = snap_cols.get("date") or snap_cols.get("snapshot_date")
    if not date_col:
        raise KeyError("snap_long must contain a 'date' or 'snapshot_date' column")

    snap = snap_long.copy()
    snap["date"] = pd.to_datetime(snap[date_col], errors="coerce").dt.normalize()
    snap["center"] = snap["center"].astype(str)
    snap["resource_code"] = snap["resource_code"].astype(str)
    snap["stock_qty"] = pd.to_numeric(snap["stock_qty"], errors="coerce")

    snap = snap[
        snap["center"].isin(centers)
        & snap["resource_code"].isin(skus)
        & snap["date"].notna()
    ].copy()

    if snap.empty:
        return pd.DataFrame(columns=["date", "center", "resource_code", "stock_qty"])

    snap = snap[(snap["date"] >= start_dt) & (snap["date"] <= horizon_end)]
    if snap.empty:
        return pd.DataFrame(columns=["date", "center", "resource_code", "stock_qty"])

    idx = pd.date_range(start_dt, horizon_end, freq="D")

    mv_all = normalize_move_dates(moves.copy())
    mv_all = annotate_move_schedule(mv_all, today_norm, lag_days, horizon_end)
    mv_all["carrier_mode"] = mv_all.get("carrier_mode", "").astype(str).str.upper()
    mv_all["resource_code"] = mv_all.get("resource_code", "").astype(str)
    mv_all["from_center"] = mv_all.get("from_center", "").astype(str)
    mv_all["to_center"] = mv_all.get("to_center", "").astype(str)
    mv_all["qty_ea"] = pd.to_numeric(mv_all.get("qty_ea", 0), errors="coerce").fillna(0)
    ship_start_col = "_onboard_date_actual" if "_onboard_date_actual" in mv_all.columns else "onboard_date"

    lines: list[pd.DataFrame] = []
    for (ct, sku), grp in snap.groupby(["center", "resource_code"]):
        grp = grp.sort_values("date")
        last_dt = grp["date"].max()

        ts = pd.DataFrame({"date": idx})
        ts["center"] = ct
        ts["resource_code"] = sku
        stock_series = grp.set_index("date")["stock_qty"].astype(float)
        ts = ts.merge(stock_series.rename("stock_qty"), on="date", how="left")
        ts["stock_qty"] = ts["stock_qty"].ffill().fillna(0.0)

        mv = mv_all[mv_all["resource_code"] == sku]
        if not mv.empty:
            eff_minus = (
                mv[
                    (mv["from_center"] == ct)
                    & mv[ship_start_col].notna()
                    & (mv[ship_start_col] > last_dt)
                ]
                .groupby(ship_start_col, as_index=False)["qty_ea"].sum()
                .rename(columns={ship_start_col: "date", "qty_ea": "delta"})
            )
            eff_minus["delta"] *= -1

            mv_center = mv[(mv["to_center"] == ct) & (mv["carrier_mode"] != "WIP")]
            if not mv_center.empty:
                eff_plus = (
                    mv_center[
                        mv_center["pred_inbound_date"].notna()
                        & (mv_center["pred_inbound_date"] > last_dt)
                    ]
                    .groupby("pred_inbound_date", as_index=False)["qty_ea"].sum()
                    .rename(columns={"pred_inbound_date": "date", "qty_ea": "delta"})
                )
            else:
                eff_plus = pd.DataFrame(columns=["date", "delta"])

            eff_all = pd.concat([eff_minus, eff_plus], ignore_index=True)
            if not eff_all.empty:
                delta_series = eff_all.groupby("date")["delta"].sum().reindex(idx, fill_value=0.0)
                ts["stock_qty"] = ts["stock_qty"] + delta_series.cumsum().values

        if "event_date" in mv_all.columns:
            wip_mask = (
                (mv_all["resource_code"] == sku)
                & (mv_all["carrier_mode"] == "WIP")
                & (mv_all["to_center"] == ct)
                & mv_all["event_date"].notna()
            )
            wip_complete = mv_all[wip_mask]
        else:
            wip_complete = pd.DataFrame(columns=mv_all.columns)
        if not wip_complete.empty:
            wip_add = (
                wip_complete.groupby("event_date", as_index=False)["qty_ea"].sum()
                .rename(columns={"event_date": "date", "qty_ea": "delta"})
            )
            delta_series = wip_add.groupby("date")["delta"].sum().reindex(idx, fill_value=0.0)
            ts["stock_qty"] = ts["stock_qty"] + delta_series.cumsum().values

        ts["stock_qty"] = ts["stock_qty"].fillna(0)
        ts["stock_qty"] = ts["stock_qty"].replace([np.inf, -np.inf], 0)
        ts["stock_qty"] = ts["stock_qty"].clip(lower=0)
        lines.append(ts)

    in_transit = compute_in_transit_series(mv_all, centers, skus, start_dt, horizon_end, today_norm, lag_days)
    if not in_transit.empty:
        lines.append(in_transit)

    wip_series = _compute_wip_series(mv_all, skus, start_dt, horizon_end)
    if not wip_series.empty:
        lines.append(wip_series)

    if not lines:
        return pd.DataFrame(columns=["date", "center", "resource_code", "stock_qty"])

    out = pd.concat(lines, ignore_index=True)
    out = out[(out["date"] >= start_dt) & (out["date"] <= horizon_end)]
    out["stock_qty"] = pd.to_numeric(out["stock_qty"], errors="coerce")
    out["stock_qty"] = out["stock_qty"].fillna(0)
    out["stock_qty"] = out["stock_qty"].replace([np.inf, -np.inf], 0)
    out["stock_qty"] = out["stock_qty"].clip(lower=0).astype(int)
    return out
