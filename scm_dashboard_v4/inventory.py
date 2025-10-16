"""Inventory related helpers for the SCM dashboard."""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from .config import CENTER_COL


def pivot_inventory_cost_from_raw(
    snap_raw: pd.DataFrame,
    _latest_dt: pd.Timestamp,
    centers: List[str],
    center_latest_dates: Optional[Dict[str, pd.Timestamp]] = None,
) -> pd.DataFrame:
    if snap_raw is None or snap_raw.empty:
        return pd.DataFrame(columns=["resource_code"] + [f"{c}_재고자산" for c in centers])

    df = snap_raw.copy()
    cols = {str(c).strip().lower(): c for c in df.columns}
    col_date = cols.get("snapshot_date") or cols.get("date")
    col_sku = cols.get("resource_code") or cols.get("sku") or cols.get("상품코드") or cols.get("option1")
    col_cogs = cols.get("cogs")

    if not all([col_date, col_sku, col_cogs]):
        return pd.DataFrame(columns=["resource_code"] + [f"{c}_재고자산" for c in centers])

    df[col_date] = pd.to_datetime(df[col_date], errors="coerce").dt.normalize()
    df[col_sku] = df[col_sku].astype(str)
    df[col_cogs] = pd.to_numeric(df[col_cogs], errors="coerce").fillna(0).clip(lower=0)

    normalized_center_dates: Dict[str, pd.Timestamp] = {}
    if center_latest_dates:
        for ct, dt in center_latest_dates.items():
            if pd.isna(dt):
                continue
            normalized_center_dates[ct] = pd.to_datetime(dt).normalize()

    if normalized_center_dates:
        relevant_dates = list({dt for dt in normalized_center_dates.values() if pd.notna(dt)})
        sub = df[df[col_date].isin(relevant_dates)].copy()
    else:
        today = pd.Timestamp.today().normalize()
        sub = df[df[col_date] == today].copy()
        if sub.empty:
            latest_date = df[col_date].max()
            sub = df[df[col_date] == latest_date].copy()

    if sub.empty:
        return pd.DataFrame(columns=["resource_code"] + [f"{c}_재고자산" for c in centers])

    cost_cols = {}
    for ct in centers:
        src_col = CENTER_COL.get(ct)
        if not src_col or src_col not in sub.columns:
            continue
        if normalized_center_dates:
            target_date = normalized_center_dates.get(ct)
            if pd.isna(target_date):
                continue
            center_sub = sub[sub[col_date] == target_date].copy()
        else:
            center_sub = sub.copy()
        if center_sub.empty:
            continue
        qty = pd.to_numeric(center_sub[src_col], errors="coerce").fillna(0).clip(lower=0)
        if qty.empty:
            continue
        cost = center_sub[col_cogs] * qty
        g = center_sub[[col_sku]].copy()
        g[f"{ct}_재고자산"] = cost
        cost_cols[ct] = g.groupby(col_sku, as_index=False)[f"{ct}_재고자산"].sum()

    if not cost_cols:
        return pd.DataFrame(columns=["resource_code"] + [f"{c}_재고자산" for c in centers])

    base = pd.DataFrame({"resource_code": pd.unique(sub[col_sku])})
    for ct, g in cost_cols.items():
        base = base.merge(g.rename(columns={col_sku: "resource_code"}), on="resource_code", how="left")
    num_cols = [c for c in base.columns if c.endswith("_재고자산")]
    base[num_cols] = base[num_cols].fillna(0).round().astype(int)
    return base
