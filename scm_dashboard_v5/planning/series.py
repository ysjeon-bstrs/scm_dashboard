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

    # Pre-normalise and pre-aggregate move deltas once to avoid filtering
    # the full frame repeatedly inside the SKU loop.
    if not mv.empty:
        mv["resource_code"] = mv.get("resource_code", "").astype(str)
        for c in ("from_center", "to_center"):
            if c in mv.columns:
                mv[c] = mv[c].astype(str)
        # date-like columns
        if ship_start_col in mv.columns:
            mv[ship_start_col] = pd.to_datetime(mv[ship_start_col], errors="coerce").dt.normalize()
        if "pred_inbound_date" in mv.columns:
            mv["pred_inbound_date"] = pd.to_datetime(mv["pred_inbound_date"], errors="coerce").dt.normalize()
        if "event_date" in mv.columns:
            mv["event_date"] = pd.to_datetime(mv["event_date"], errors="coerce").dt.normalize()
        mv["qty_ea"] = pd.to_numeric(mv.get("qty_ea"), errors="coerce").fillna(0.0)

        # Outbound deltas by (from_center, sku)
        outbound_map: dict[tuple[str, str], pd.Series] = {}
        outbound_df = mv[(mv[ship_start_col].notna()) & mv["resource_code"].isin(skus_set)]
        if "from_center" in mv.columns and not outbound_df.empty:
            g = (
                outbound_df.groupby(["from_center", "resource_code", ship_start_col], as_index=False)[
                    "qty_ea"
                ]
                .sum()
                .rename(columns={ship_start_col: "date"})
            )
            g["delta"] = -g["qty_ea"].astype(float)
            for (fc, sku), chunk in g.groupby(["from_center", "resource_code"], dropna=True):
                outbound_map[(str(fc), str(sku))] = (
                    chunk.set_index("date")["delta"].sort_index()
                )
        else:
            outbound_map = {}

        # Inbound (non-WIP) deltas by (to_center, sku)
        inbound_map: dict[tuple[str, str], pd.Series] = {}
        inbound_filter = (
            (mv.get("carrier_mode") != "WIP")
            & mv["resource_code"].isin(skus_set)
            & ("to_center" in mv.columns)
            & mv.get("pred_inbound_date").notna()
        )
        inbound_df = mv[inbound_filter] if inbound_filter is not None else pd.DataFrame(columns=mv.columns)
        if not inbound_df.empty:
            g = (
                inbound_df.groupby(["to_center", "resource_code", "pred_inbound_date"], as_index=False)[
                    "qty_ea"
                ]
                .sum()
                .rename(columns={"pred_inbound_date": "date"})
            )
            g["delta"] = g["qty_ea"].astype(float)
            for (tc, sku), chunk in g.groupby(["to_center", "resource_code"], dropna=True):
                inbound_map[(str(tc), str(sku))] = (
                    chunk.set_index("date")["delta"].sort_index()
                )

        # WIP completion deltas by (to_center, sku)
        wip_map: dict[tuple[str, str], pd.Series] = {}
        if "event_date" in mv.columns:
            wip_df = mv[(mv.get("carrier_mode") == "WIP") & mv["event_date"].notna()]
            if not wip_df.empty and "to_center" in wip_df.columns:
                g = (
                    wip_df.groupby(["to_center", "resource_code", "event_date"], as_index=False)[
                        "qty_ea"
                    ]
                    .sum()
                    .rename(columns={"event_date": "date"})
                )
                g["delta"] = g["qty_ea"].astype(float)
                for (tc, sku), chunk in g.groupby(["to_center", "resource_code"], dropna=True):
                    wip_map[(str(tc), str(sku))] = (
                        chunk.set_index("date")["delta"].sort_index()
                    )
        else:
            wip_map = {}
    else:
        outbound_map = {}
        inbound_map = {}
        wip_map = {}

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

        # Apply pre-aggregated move deltas for the (center, sku)
        # Outbound from this center
        eff_minus_series = outbound_map.get((str(ct), str(sku)))
        # Inbound to this center (excluding WIP)
        eff_plus_series = inbound_map.get((str(ct), str(sku)))

        deltas_parts: list[pd.Series] = []
        if eff_minus_series is not None and not eff_minus_series.empty:
            deltas_parts.append(eff_minus_series[eff_minus_series.index > last_dt])
        if eff_plus_series is not None and not eff_plus_series.empty:
            deltas_parts.append(eff_plus_series[eff_plus_series.index > last_dt])
        if deltas_parts:
            eff_all_series = (
                pd.concat(deltas_parts).groupby(level=0).sum().reindex(idx, fill_value=0.0)
            )
            ts["stock_qty"] = ts["stock_qty"].add(eff_all_series.cumsum(), fill_value=0.0)

        # WIP completion adds to stock for this center
        wip_series = wip_map.get((str(ct), str(sku)))
        if wip_series is not None and not wip_series.empty:
            wip_delta = wip_series.reindex(idx, fill_value=0.0)
            ts["stock_qty"] = ts["stock_qty"].add(wip_delta.cumsum(), fill_value=0.0)

        ts["stock_qty"] = ts["stock_qty"].fillna(0)
        ts["stock_qty"] = ts["stock_qty"].replace([np.inf, -np.inf], 0)
        ts["stock_qty"] = ts["stock_qty"].clip(lower=0)
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
