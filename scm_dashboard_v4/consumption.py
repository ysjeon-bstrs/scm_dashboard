"""Consumption modelling utilities for the SCM dashboard."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st


@st.cache_data(ttl=3600)
def estimate_daily_consumption(
    snap_long: pd.DataFrame,
    centers_sel: List[str],
    skus_sel: List[str],
    asof_dt: pd.Timestamp,
    lookback_days: int = 28,
) -> Dict[Tuple[str, str], float]:
    snap = snap_long.rename(columns={"snapshot_date": "date"}).copy()
    snap["date"] = pd.to_datetime(snap["date"], errors="coerce").dt.normalize()
    start = (asof_dt - pd.Timedelta(days=int(lookback_days) - 1)).normalize()

    hist = snap[(snap["date"] >= start) & (snap["date"] <= asof_dt) & (snap["center"].isin(centers_sel)) & (snap["resource_code"].isin(skus_sel))]

    rates: Dict[Tuple[str, str], float] = {}
    if hist.empty:
        return rates

    for (ct, sku), g in hist.groupby(["center", "resource_code"]):
        series = (
            g.dropna(subset=["date"])  # drop rows without a usable date
            .sort_values("date")
            .groupby("date", as_index=False)["stock_qty"]
            .last()
        )
        if series.empty:
            continue

        ts = (
            series.set_index("date")["stock_qty"].astype(float).asfreq("D").ffill()
        )
        if ts.dropna().shape[0] < max(7, lookback_days // 2):
            continue
        x = np.arange(len(ts), dtype=float)
        y = ts.values.astype(float)
        rate_reg = max(0.0, -np.polyfit(x, y, 1)[0])
        dec = (-np.diff(y)).clip(min=0)
        rate_dec = dec.mean() if len(dec) else 0.0
        rate = max(rate_reg, rate_dec)
        if rate > 0:
            rates[(ct, sku)] = float(rate)
    return rates


@st.cache_data(ttl=1800)
def apply_consumption_with_events(
    timeline: pd.DataFrame,
    snap_long: pd.DataFrame,
    centers_sel: List[str],
    skus_sel: List[str],
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
    lookback_days: int = 28,
    events: Optional[List[Dict]] = None,
) -> pd.DataFrame:
    out = timeline.copy()
    if out.empty:
        return out
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()

    snap_cols = {c.lower(): c for c in snap_long.columns}
    date_col = snap_cols.get("date") or snap_cols.get("snapshot_date")
    if date_col is None:
        raise KeyError("snap_long에는 'date' 또는 'snapshot_date' 컬럼이 필요합니다.")

    latest_snap = pd.to_datetime(snap_long[date_col]).max().normalize()
    cons_start = max(latest_snap + pd.Timedelta(days=1), start_dt)
    if cons_start > end_dt:
        return out

    idx = pd.date_range(cons_start, end_dt, freq="D")
    uplift = pd.Series(1.0, index=idx)
    if events:
        for event in events:
            s = pd.to_datetime(event.get("start"), errors="coerce")
            t = pd.to_datetime(event.get("end"), errors="coerce")
            u = min(3.0, max(-1.0, float(event.get("uplift", 0.0))))
            if pd.notna(s) and pd.notna(t):
                s = s.normalize()
                t = t.normalize()
                s = max(s, idx[0])
                t = min(t, idx[-1])
                if s <= t:
                    uplift.loc[s:t] = uplift.loc[s:t] * (1.0 + u)

    rates = estimate_daily_consumption(snap_long, centers_sel, skus_sel, latest_snap, int(lookback_days))

    chunks: list[pd.DataFrame] = []
    for (ct, sku), g in out.groupby(["center", "resource_code"]):
        if ct in ("In-Transit", "WIP"):
            chunks.append(g)
            continue

        rate = float(rates.get((ct, sku), 0.0))
        if rate <= 0:
            chunks.append(g)
            continue

        g = g.sort_values("date").copy()
        g["stock_qty"] = pd.to_numeric(g["stock_qty"], errors="coerce")
        mask = g["date"] >= cons_start
        if not mask.any():
            chunks.append(g)
            continue

        last_real = (
            g.loc[~mask, "stock_qty"].dropna().iloc[-1]
            if (~mask).any() and g.loc[~mask, "stock_qty"].notna().any()
            else 0.0
        )
        future_stock = pd.to_numeric(g.loc[mask, "stock_qty"], errors="coerce")
        g.loc[mask, "stock_qty"] = future_stock.fillna(last_real)

        daily = g.loc[mask, "date"].map(uplift).fillna(1.0).values * rate
        stk = g.loc[mask, "stock_qty"].astype(float).values
        for i in range(len(stk)):
            dec = daily[i]
            stk[i:] = np.maximum(0.0, stk[i:] - dec)
        g.loc[mask, "stock_qty"] = stk
        chunks.append(g)

    if not chunks:
        return out

    out = pd.concat(chunks, ignore_index=True)
    out["stock_qty"] = pd.to_numeric(out["stock_qty"], errors="coerce")
    out["stock_qty"] = out["stock_qty"].fillna(0)
    out["stock_qty"] = out["stock_qty"].replace([np.inf, -np.inf], 0)
    out["stock_qty"] = out["stock_qty"].round().clip(lower=0).astype(int)
    return out
