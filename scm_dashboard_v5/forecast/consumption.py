"""Forecasting adapters that reuse the proven v4 consumption logic."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import pandas as pd

from scm_dashboard_v4 import consumption as v4_consumption


def estimate_daily_consumption(sales: pd.DataFrame, *, window: int = 28) -> pd.DataFrame:
    """Delegate to the stable v4 estimator while keeping the new namespace."""

    return v4_consumption.estimate_daily_consumption(sales, window=window)


def apply_consumption_with_events(
    timeline: pd.DataFrame,
    snapshot: pd.DataFrame,
    *,
    centers: Sequence[str],
    skus: Sequence[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    lookback_days: int = 28,
    events: Optional[Iterable[dict]] = None,
    cons_start: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Delegate to the v4 helper while controlling the consumption window."""

    centers_list = list(centers)
    skus_list = list(skus)
    start_norm = pd.to_datetime(start).normalize()
    end_norm = pd.to_datetime(end).normalize()
    events_list = list(events) if events else None

    baseline_source = timeline.copy()
    if "date" in baseline_source.columns:
        baseline_source["date"] = pd.to_datetime(
            baseline_source["date"], errors="coerce"
        ).dt.normalize()

    timeline_copy = baseline_source.copy()
    if "date" in timeline_copy.columns:
        timeline_copy["date"] = pd.to_datetime(
            timeline_copy["date"], errors="coerce"
        ).dt.normalize()

    stock_col = "stock_qty" if "stock_qty" in timeline_copy.columns else None
    key_cols = [
        col
        for col in ["center", "resource_code"]
        if col in timeline_copy.columns
    ]
    if stock_col and key_cols and "date" in timeline_copy.columns:
        timeline_copy = timeline_copy.sort_values(key_cols + ["date"])
        timeline_copy[stock_col] = pd.to_numeric(
            timeline_copy[stock_col], errors="coerce"
        )
        timeline_copy[stock_col] = (
            timeline_copy.groupby(key_cols)[stock_col].ffill()
        )
        timeline_copy[stock_col] = timeline_copy[stock_col].fillna(0.0)

    result = v4_consumption.apply_consumption_with_events(
        timeline_copy,
        snapshot,
        centers_list,
        skus_list,
        start_norm,
        end_norm,
        int(lookback_days),
        events_list,
    )

    if "date" not in result.columns:
        return result

    result = result.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.normalize()

    snap_cols = {c.lower(): c for c in snapshot.columns}
    date_col = snap_cols.get("date") or snap_cols.get("snapshot_date")
    latest_snap = None
    if date_col is not None and not snapshot.empty:
        latest_vals = pd.to_datetime(snapshot[date_col], errors="coerce").dropna()
        if not latest_vals.empty:
            latest_snap = latest_vals.max().normalize()

    if cons_start is not None:
        cons_start_norm = pd.to_datetime(cons_start).normalize()
    elif latest_snap is not None:
        cons_start_norm = max(latest_snap + pd.Timedelta(days=1), start_norm)
    else:
        cons_start_norm = None

    if cons_start_norm is not None and stock_col and key_cols:
        groups = []
        for _, grp in result.groupby(key_cols, sort=False):
            grp = grp.sort_values("date").copy()
            grp[stock_col] = pd.to_numeric(grp[stock_col], errors="coerce")
            mask = grp["date"] >= cons_start_norm
            if mask.any():
                if (~mask).any() and grp.loc[~mask, stock_col].notna().any():
                    last_real = grp.loc[~mask, stock_col].dropna().iloc[-1]
                else:
                    last_real = 0.0
                future = grp.loc[mask, stock_col]
                grp.loc[mask, stock_col] = future.fillna(last_real)
            groups.append(grp)
        result = pd.concat(groups, ignore_index=True)

    if "date" not in baseline_source.columns:
        return result

    align_keys = [
        col
        for col in ["date", "center", "resource_code"]
        if col in result.columns and col in baseline_source.columns
    ]
    if not align_keys:
        return result

    orig = (
        baseline_source[align_keys + ["stock_qty"]]
        .copy()
        .rename(columns={"stock_qty": "_orig_stock_qty"})
    )

    orig = orig.drop_duplicates(subset=align_keys, keep="last")

    merged = result.merge(orig, on=align_keys, how="left")
    if cons_start_norm is not None:
        mask = merged["date"] < cons_start_norm
        if mask.any():
            restored = merged.loc[mask, "_orig_stock_qty"].fillna(
                merged.loc[mask, "stock_qty"]
            )
            merged.loc[mask, "stock_qty"] = restored.values

    merged = merged.drop(columns=["_orig_stock_qty"], errors="ignore")

    desired_cols = list(baseline_source.columns)
    remaining_cols = [c for c in merged.columns if c not in desired_cols]
    ordered_cols = desired_cols + remaining_cols
    merged = merged[[c for c in ordered_cols if c in merged.columns]]

    return merged
