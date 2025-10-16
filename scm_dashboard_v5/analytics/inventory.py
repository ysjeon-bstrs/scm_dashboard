"""Analytics adapters exposing the v4 inventory helpers under the new structure."""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from scm_dashboard_v4 import inventory as v4_inventory
from scm_dashboard_v4.config import CENTER_COL


def pivot_inventory_cost_from_raw(
    raw: pd.DataFrame,
    latest_dt: pd.Timestamp,
    centers: List[str],
    center_latest_dates: Optional[Dict[str, pd.Timestamp]] = None,
) -> pd.DataFrame:
    """Return cost pivots while remaining compatible with older helper signatures."""

    if center_latest_dates:
        try:
            return v4_inventory.pivot_inventory_cost_from_raw(
                raw,
                latest_dt,
                centers,
                center_latest_dates=center_latest_dates,
            )
        except TypeError:
            # Older deployments may still expose the three-argument helper. Fall
            # back to a local implementation that mirrors the new behaviour so
            # valuations stay aligned with per-center snapshot dates instead of
            # crashing entirely.
            return _pivot_inventory_cost_from_raw_per_center(
                raw, latest_dt, centers, center_latest_dates
            )

    return v4_inventory.pivot_inventory_cost_from_raw(raw, latest_dt, centers)


def _pivot_inventory_cost_from_raw_per_center(
    raw: pd.DataFrame,
    latest_dt: pd.Timestamp,
    centers: List[str],
    center_latest_dates: Dict[str, pd.Timestamp],
) -> pd.DataFrame:
    """Compute inventory cost when v4 helper lacks the latest signature."""

    if raw is None or raw.empty:
        return pd.DataFrame(columns=["resource_code"] + [f"{c}_재고자산" for c in centers])

    df = raw.copy()
    cols = {str(c).strip().lower(): c for c in df.columns}
    col_date = cols.get("snapshot_date") or cols.get("date")
    col_sku = (
        cols.get("resource_code")
        or cols.get("sku")
        or cols.get("상품코드")
        or cols.get("option1")
    )
    col_cogs = cols.get("cogs")

    if not all([col_date, col_sku, col_cogs]):
        return pd.DataFrame(columns=["resource_code"] + [f"{c}_재고자산" for c in centers])

    df[col_date] = pd.to_datetime(df[col_date], errors="coerce").dt.normalize()
    df[col_sku] = df[col_sku].astype(str)
    df[col_cogs] = pd.to_numeric(df[col_cogs], errors="coerce").fillna(0).clip(lower=0)

    normalized_center_dates: Dict[str, pd.Timestamp] = {}
    for center in centers:
        if center not in center_latest_dates:
            continue
        dt = pd.to_datetime(center_latest_dates.get(center), errors="coerce")
        if pd.isna(dt):
            continue
        normalized_center_dates[center] = dt.normalize()

    if not normalized_center_dates:
        fallback_dt = pd.to_datetime(latest_dt, errors="coerce")
        if pd.isna(fallback_dt):
            fallback_dt = pd.Timestamp.today().normalize()
        else:
            fallback_dt = fallback_dt.normalize()
        return v4_inventory.pivot_inventory_cost_from_raw(raw, fallback_dt, centers)

    relevant_dates = list({dt for dt in normalized_center_dates.values() if pd.notna(dt)})
    sub = df[df[col_date].isin(relevant_dates)].copy()
    if sub.empty:
        return pd.DataFrame(columns=["resource_code"] + [f"{c}_재고자산" for c in centers])

    cost_cols = {}
    for center in centers:
        src_col = CENTER_COL.get(center)
        if not src_col or src_col not in sub.columns:
            continue
        target_date = normalized_center_dates.get(center)
        if pd.isna(target_date):
            continue
        center_subset = sub[sub[col_date] == target_date].copy()
        if center_subset.empty:
            continue
        qty = pd.to_numeric(center_subset[src_col], errors="coerce").fillna(0).clip(lower=0)
        cost = center_subset[col_cogs] * qty
        g = center_subset[[col_sku]].copy()
        g[f"{center}_재고자산"] = cost
        cost_cols[center] = g.groupby(col_sku, as_index=False)[f"{center}_재고자산"].sum()

    if not cost_cols:
        return pd.DataFrame(columns=["resource_code"] + [f"{c}_재고자산" for c in centers])

    base = pd.DataFrame({"resource_code": pd.unique(sub[col_sku])})
    for center, grouped in cost_cols.items():
        base = base.merge(
            grouped.rename(columns={col_sku: "resource_code"}),
            on="resource_code",
            how="left",
        )

    num_cols = [c for c in base.columns if c.endswith("_재고자산")]
    base[num_cols] = base[num_cols].fillna(0).round().astype(int)
    return base
