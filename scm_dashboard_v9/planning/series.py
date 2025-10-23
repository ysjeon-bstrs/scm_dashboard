"""Series builders for centre, in-transit, and WIP timelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Set

import numpy as np
import pandas as pd


TIMELINE_COLUMNS = ["date", "center", "resource_code", "stock_qty"]


def _empty_timeline() -> pd.DataFrame:
    """Return a shared empty frame matching the timeline schema."""

    return pd.DataFrame(columns=TIMELINE_COLUMNS)


def _normalise_selection(values: Iterable[str]) -> Set[str]:
    """Return a cleaned-up selection set used across builders."""

    normalised = {str(value).strip() for value in values if value is not None}
    return {value for value in normalised if value}


def _resolve_onboard_column(frame: pd.DataFrame) -> str:
    """Pick the effective onboarding date column present in *frame*."""

    if "_onboard_date_actual" in frame.columns:
        return "_onboard_date_actual"
    return "onboard_date"


@dataclass(frozen=True)
class SeriesIndex:
    start: pd.Timestamp
    end: pd.Timestamp

    @property
    def range(self) -> pd.DatetimeIndex:
        return pd.date_range(self.start, self.end, freq="D")


def build_center_series(
    snapshot: pd.DataFrame,
    moves: pd.DataFrame,
    *,
    centers: Iterable[str],
    skus: Iterable[str],
    index: SeriesIndex,
) -> pd.DataFrame:
    centers_set = _normalise_selection(centers)
    skus_set = _normalise_selection(skus)
    if snapshot.empty or not centers_set or not skus_set:
        return _empty_timeline()

    idx = index.range
    mv = moves.copy()
    ship_start_col = _resolve_onboard_column(mv)

    lines = []
    for (ct, sku), grp in snapshot.groupby(["center", "resource_code"]):
        if ct not in centers_set or sku not in skus_set:
            continue
        grp = grp.dropna(subset=["date"]).sort_values("date")
        if grp.empty:
            continue
        grp = (
            grp.groupby("date", as_index=False)["stock_qty"].last().sort_values("date")
        )
        grp = grp.drop_duplicates(subset="date", keep="last")
        last_dt = grp["date"].max()

        ts = pd.DataFrame(index=idx)
        ts["center"] = ct
        ts["resource_code"] = sku
        stock_series = grp.set_index("date")["stock_qty"].astype(float)
        stock_series = stock_series[~stock_series.index.duplicated(keep="last")]
        stock_series = stock_series.reindex(idx)
        stock_series = stock_series.ffill().fillna(0.0)
        ts["stock_qty"] = stock_series

        mv_sku = mv[mv["resource_code"] == sku]
        if not mv_sku.empty:
            eff_minus = (
                mv_sku[
                    (mv_sku["from_center"] == ct)
                    & mv_sku[ship_start_col].notna()
                    & (mv_sku[ship_start_col] > last_dt)
                ]
                .groupby(ship_start_col, as_index=False)["qty_ea"].sum()
                .rename(columns={ship_start_col: "date", "qty_ea": "delta"})
            )
            eff_minus["delta"] *= -1

            mv_center = mv_sku[(mv_sku["to_center"] == ct) & (mv_sku["carrier_mode"] != "WIP")]
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

            frames_to_concat = [df for df in [eff_minus, eff_plus] if not df.empty]
            if frames_to_concat:
                eff_all = pd.concat(frames_to_concat, ignore_index=True)
                delta_series = (
                    eff_all.groupby("date")["delta"].sum().reindex(idx, fill_value=0.0)
                )
                ts["stock_qty"] = ts["stock_qty"].add(delta_series.cumsum(), fill_value=0.0)

        if "event_date" in mv_sku.columns:
            wip_mask = (
                (mv_sku["carrier_mode"] == "WIP")
                & (mv_sku["to_center"] == ct)
                & mv_sku["event_date"].notna()
            )
            wip_complete = mv_sku[wip_mask]
        else:
            wip_complete = pd.DataFrame(columns=mv_sku.columns)
        if not wip_complete.empty:
            wip_add = (
                wip_complete.groupby("event_date", as_index=False)["qty_ea"].sum()
                .rename(columns={"event_date": "date", "qty_ea": "delta"})
            )
            delta_series = (
                wip_add.groupby("date")["delta"].sum().reindex(idx, fill_value=0.0)
            )
            ts["stock_qty"] = ts["stock_qty"].add(delta_series.cumsum(), fill_value=0.0)

        ts["stock_qty"] = ts["stock_qty"].fillna(0).astype(float)
        ts["stock_qty"] = ts["stock_qty"].replace([np.inf, -np.inf], 0.0)
        ts["stock_qty"] = ts["stock_qty"].clip(lower=0.0)
        lines.append(ts.reset_index().rename(columns={"index": "date"}))

    if not lines:
        return _empty_timeline()
    out = pd.concat(lines, ignore_index=True)
    return out[TIMELINE_COLUMNS]


def build_in_transit_series(
    moves: pd.DataFrame,
    *,
    centers: Iterable[str],
    skus: Iterable[str],
    index: SeriesIndex,
    today: pd.Timestamp,
    lag_days: int,
) -> pd.DataFrame:
    if moves.empty:
        return _empty_timeline()

    centers_set = _normalise_selection(centers)
    skus_set = _normalise_selection(skus)
    if not centers_set or not skus_set:
        return _empty_timeline()

    start_dt = index.start
    horizon_end = index.end
    idx = index.range
    start_col = _resolve_onboard_column(moves)

    filtered = moves[
        (moves["carrier_mode"] != "WIP")
        & moves["resource_code"].isin(skus_set)
        & moves["to_center"].isin(centers_set)
    ].copy()
    if filtered.empty:
        return _empty_timeline()

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
        return _empty_timeline()

    out = pd.concat(lines, ignore_index=True)
    out["stock_qty"] = out["stock_qty"].round().astype(int)
    return out[TIMELINE_COLUMNS]


def build_wip_series(
    moves: pd.DataFrame,
    *,
    skus: Iterable[str],
    index: SeriesIndex,
) -> pd.DataFrame:
    if moves.empty:
        return _empty_timeline()

    skus_set = _normalise_selection(skus)
    if not skus_set:
        return _empty_timeline()

    idx = index.range
    start_col = _resolve_onboard_column(moves)

    wip = moves[(moves["carrier_mode"] == "WIP") & moves["resource_code"].isin(skus_set)]
    if wip.empty:
        return _empty_timeline()

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

        onboard_dates = pd.to_datetime(grp[start_col], errors="coerce")
        event_dates = (
            pd.to_datetime(grp.get("event_date"), errors="coerce")
            if "event_date" in grp
            else pd.Series(pd.NaT, index=grp.index, dtype="datetime64[ns]")
        )
        carry_mask = (
            onboard_dates.notna()
            & (onboard_dates < idx[0])
            & event_dates.fillna(pd.Timestamp.max).ge(idx[0])
        )
        if carry_mask.any():
            carry = int(
                pd.to_numeric(grp.loc[carry_mask, "qty_ea"], errors="coerce")
                .fillna(0)
                .sum()
            )
            if carry:
                series = (series + carry).clip(lower=0)
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
        return _empty_timeline()

    out = pd.concat(lines, ignore_index=True)
    out["stock_qty"] = out["stock_qty"].round().astype(int)
    return out[TIMELINE_COLUMNS]
