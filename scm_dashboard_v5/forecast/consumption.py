"""Forecast-period helpers that mirror the proven v4 logic for v5."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from scm_dashboard_v4 import consumption as v4_consumption


def load_amazon_daily_sales_from_snapshot_raw(
    snapshot_raw: pd.DataFrame,
    centers: Tuple[str, ...] = ("AMZUS",),
    skus: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Return daily FBA outbound volumes from ``snapshot_raw``.

    The legacy ``snapshot_raw`` sheet occasionally exposes the FBA outbound
    quantity column with different labels.  This helper normalises the
    structure so the downstream sales simulator can reuse the values as
    ground-truth daily sales.
    """

    if snapshot_raw is None or snapshot_raw.empty:
        return pd.DataFrame(columns=["date", "center", "resource_code", "sales_ea"])

    df = snapshot_raw.copy()

    rename_map: dict[str, str] = {}
    for col in df.columns:
        key = str(col).strip().lower()
        if key in {"snapshot_date", "date", "스냅샷일자", "스냅샷 일자"}:
            rename_map[col] = "date"
        elif key in {"center", "창고", "창고명"}:
            rename_map[col] = "center"
        elif key in {"resource_code", "sku", "상품코드", "product_code"}:
            rename_map[col] = "resource_code"
        elif "fba_output_stock" in key:
            rename_map[col] = "fba_output_stock"

    df = df.rename(columns=rename_map)
    required = {"date", "center", "resource_code", "fba_output_stock"}
    missing = required - set(df.columns)
    if missing:
        return pd.DataFrame(columns=["date", "center", "resource_code", "sales_ea"])

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["fba_output_stock"] = pd.to_numeric(df["fba_output_stock"], errors="coerce").fillna(0)
    df = df[df["fba_output_stock"] >= 0]

    if centers:
        center_set = {str(ct) for ct in centers}
        df = df[df["center"].astype(str).isin(center_set)]

    if skus:
        sku_set = {str(sku) for sku in skus}
        df = df[df["resource_code"].astype(str).isin(sku_set)]

    grouped = (
        df.groupby(["date", "center", "resource_code"], as_index=False)["fba_output_stock"]
        .sum()
        .rename(columns={"fba_output_stock": "sales_ea"})
    )
    grouped["sales_ea"] = grouped["sales_ea"].fillna(0).astype(float)
    return grouped


def forecast_sales_and_inventory(
    daily_sales: pd.DataFrame,
    timeline_center: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    lookback_days: int = 28,
    uplift_events: Optional[Iterable[dict]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate sales and inventory simultaneously for Amazon centres.

    The simulator ensures that forecast sales never exceed the available
    inventory (including inbound events present on the centre timeline).  Once
    the simulated stock reaches zero, the sales forecast is automatically
    capped at zero, keeping the OOS timing aligned between charts and KPIs.
    """

    if timeline_center is None or timeline_center.empty:
        empty_sales = pd.DataFrame(columns=["date", "center", "resource_code", "sales_ea"])
        empty_inv = pd.DataFrame(columns=["date", "center", "resource_code", "stock_qty"])
        return empty_sales, empty_inv

    if daily_sales is None or daily_sales.empty:
        sales_history = pd.DataFrame(columns=["date", "center", "resource_code", "sales_ea"])
    else:
        sales_history = daily_sales.copy()
        rename_map = {col.lower(): col for col in sales_history.columns}
        if "date" not in sales_history.columns and "snapshot_date" in rename_map:
            sales_history = sales_history.rename(columns={rename_map["snapshot_date"]: "date"})
        if "center" not in sales_history.columns and "center" in rename_map:
            sales_history = sales_history.rename(columns={rename_map["center"]: "center"})
        if "resource_code" not in sales_history.columns and "resource_code" in rename_map:
            sales_history = sales_history.rename(columns={rename_map["resource_code"]: "resource_code"})
        sales_history["date"] = pd.to_datetime(sales_history["date"], errors="coerce").dt.normalize()
        if "sales_ea" not in sales_history.columns:
            value_col = None
            for candidate in sales_history.columns:
                if str(candidate).lower().endswith("sales_ea"):
                    value_col = candidate
                    break
            if value_col is not None:
                sales_history = sales_history.rename(columns={value_col: "sales_ea"})
        sales_history["sales_ea"] = pd.to_numeric(
            sales_history.get("sales_ea"), errors="coerce"
        ).fillna(0)
        sales_history = sales_history.dropna(subset=["date"])

    timeline = timeline_center.copy()
    timeline["date"] = pd.to_datetime(timeline["date"], errors="coerce").dt.normalize()
    timeline["center"] = timeline.get("center", "").astype(str)
    timeline["resource_code"] = timeline.get("resource_code", "").astype(str)
    timeline["stock_qty"] = pd.to_numeric(timeline.get("stock_qty"), errors="coerce").fillna(0)

    center_set = timeline["center"].dropna().astype(str).unique().tolist()
    sku_set = timeline["resource_code"].dropna().astype(str).unique().tolist()
    if not center_set or not sku_set:
        empty_sales = pd.DataFrame(columns=["date", "center", "resource_code", "sales_ea"])
        empty_inv = pd.DataFrame(columns=["date", "center", "resource_code", "stock_qty"])
        return empty_sales, empty_inv

    start_norm = pd.to_datetime(start).normalize()
    end_norm = pd.to_datetime(end).normalize()
    if end_norm < start_norm:
        empty_sales = pd.DataFrame(columns=["date", "center", "resource_code", "sales_ea"])
        empty_inv = pd.DataFrame(columns=["date", "center", "resource_code", "stock_qty"])
        return empty_sales, empty_inv

    index = pd.date_range(start_norm, end_norm, freq="D")

    sku_mask = sales_history.get("resource_code")
    if sku_mask is None:
        filtered_sales = pd.DataFrame(columns=["date", "center", "resource_code", "sales_ea"])
    else:
        sku_mask = sales_history["resource_code"].astype(str).isin(sku_set)
        if "center" in sales_history.columns:
            center_mask = sales_history["center"].astype(str).isin(center_set)
        else:
            center_mask = pd.Series(True, index=sales_history.index)
        filtered_sales = sales_history[sku_mask & center_mask].copy()
        if "center" not in filtered_sales.columns:
            inferred_center = center_set[0] if len(center_set) == 1 else ""
            filtered_sales["center"] = inferred_center

    pivot = (
        filtered_sales.groupby(["resource_code", "date"], as_index=False)["sales_ea"].sum()
        .pivot(index="date", columns="resource_code", values="sales_ea")
        .sort_index()
        .asfreq("D", fill_value=0)
    )

    if lookback_days <= 0:
        lookback_days = 1
    mean_rate = pivot.tail(int(lookback_days)).mean().clip(lower=0)

    uplift = pd.Series(1.0, index=index)
    if uplift_events:
        for event in uplift_events:
            s = pd.to_datetime(event.get("start"), errors="coerce")
            t = pd.to_datetime(event.get("end"), errors="coerce")
            u = float(event.get("uplift", 0.0))
            if pd.notna(s) and pd.notna(t):
                s_norm = max(s.normalize(), index[0])
                t_norm = min(t.normalize(), index[-1])
                if s_norm <= t_norm:
                    uplift.loc[s_norm:t_norm] = uplift.loc[s_norm:t_norm] * (1.0 + u)

    timeline_rows: list[pd.DataFrame] = []
    sales_rows: list[pd.DataFrame] = []

    for (center_value, sku), group in timeline.groupby(["center", "resource_code"]):
        group = group.sort_values("date").copy()
        group = group.set_index("date")
        series = group["stock_qty"].asfreq("D")
        series = series.ffill()
        if series.empty:
            continue

        anchor_date = min(series.index.min(), start_norm - pd.Timedelta(days=1))
        full_index = pd.date_range(anchor_date, index[-1], freq="D")
        series = series.reindex(full_index, method="ffill").fillna(0.0)

        cons_start = max(start_norm, series.index.min())
        inv_start = float(series.get(cons_start - pd.Timedelta(days=1), series.iloc[-1]))

        deltas = series.diff().fillna(0.0)
        inbound_future = deltas.loc[cons_start:].clip(lower=0)

        base_rate = float(mean_rate.get(sku, 0.0))
        if base_rate < 0:
            base_rate = 0.0

        uplift_slice = uplift.reindex(index, fill_value=1.0)
        daily_target = pd.Series(base_rate, index=index) * uplift_slice

        inventory = inv_start
        inv_curve: list[float] = []
        sales_curve: list[float] = []

        for day in index:
            inventory += float(inbound_future.get(day, 0.0))
            desired = float(daily_target.loc[day])
            sell = min(inventory, desired)
            inventory = max(0.0, inventory - sell)
            sales_curve.append(sell)
            inv_curve.append(inventory)

        df_sales = pd.DataFrame(
            {
                "date": index,
                "center": center_value,
                "resource_code": sku,
                "sales_ea": np.array(sales_curve, dtype=float),
            }
        )
        df_inv = pd.DataFrame(
            {
                "date": index,
                "center": center_value,
                "resource_code": sku,
                "stock_qty": np.array(inv_curve, dtype=float),
            }
        )

        sales_rows.append(df_sales)
        timeline_rows.append(df_inv)

    forecast_sales = (
        pd.concat(sales_rows, ignore_index=True)
        if sales_rows
        else pd.DataFrame(columns=["date", "center", "resource_code", "sales_ea"])
    )
    forecast_inv = (
        pd.concat(timeline_rows, ignore_index=True)
        if timeline_rows
        else pd.DataFrame(columns=["date", "center", "resource_code", "stock_qty"])
    )

    forecast_sales["date"] = pd.to_datetime(forecast_sales["date"], errors="coerce").dt.normalize()
    forecast_inv["date"] = pd.to_datetime(forecast_inv["date"], errors="coerce").dt.normalize()

    forecast_sales["sales_ea"] = forecast_sales["sales_ea"].fillna(0).round().clip(lower=0).astype(int)
    forecast_inv["stock_qty"] = forecast_inv["stock_qty"].fillna(0).round().clip(lower=0).astype(int)

    history = filtered_sales[["date", "center", "resource_code", "sales_ea"]].copy()
    history["sales_ea"] = history["sales_ea"].fillna(0).round().clip(lower=0).astype(int)

    combined_sales = pd.concat([history, forecast_sales], ignore_index=True)
    combined_sales = combined_sales.sort_values(["resource_code", "date"]).drop_duplicates(
        subset=["date", "center", "resource_code"], keep="last"
    )

    return combined_sales.reset_index(drop=True), forecast_inv.reset_index(drop=True)

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
    centers_list = list(centers)
    skus_list = list(skus)
    start_norm = pd.to_datetime(start).normalize()
    end_norm = pd.to_datetime(end).normalize()
    events_list = list(events) if events else None

    out = timeline.copy()
    if out.empty:
        return out

    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()

    snap_cols = {c.lower(): c for c in snapshot.columns}
    date_col = snap_cols.get("date") or snap_cols.get("snapshot_date")
    if date_col is None:
        raise KeyError("snapshot must include a 'date' or 'snapshot_date' column")

    latest_vals = pd.to_datetime(snapshot[date_col], errors="coerce").dropna()
    latest_snap = latest_vals.max().normalize() if not latest_vals.empty else pd.NaT

    if cons_start is not None:
        cons_start_norm = pd.to_datetime(cons_start).normalize()
        cons_start_norm = max(cons_start_norm, start_norm)
        if not pd.isna(latest_snap):
            cons_start_norm = max(cons_start_norm, latest_snap + pd.Timedelta(days=1))
    elif pd.isna(latest_snap):
        cons_start_norm = start_norm
    else:
        cons_start_norm = max(latest_snap + pd.Timedelta(days=1), start_norm)

    if cons_start_norm > end_norm:
        return out

    idx = pd.date_range(cons_start_norm, end_norm, freq="D")
    uplift = pd.Series(1.0, index=idx)
    if events_list:
        for event in events_list:
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

    rates = {}
    if not pd.isna(latest_snap):
        rates = v4_consumption.estimate_daily_consumption(
            snapshot,
            centers_list,
            skus_list,
            latest_snap,
            int(lookback_days),
        )

    chunks: list[pd.DataFrame] = []
    for (ct, sku), grp in out.groupby(["center", "resource_code"]):
        g = grp.sort_values("date").copy()
        g["stock_qty"] = pd.to_numeric(g.get("stock_qty"), errors="coerce")

        if ct in ("In-Transit", "WIP"):
            chunks.append(g)
            continue

        g["stock_qty"] = g["stock_qty"].ffill()

        rate = float(rates.get((ct, sku), 0.0)) if rates else 0.0
        if rate > 0:
            mask = g["date"] >= cons_start_norm
            if mask.any():
                daily = g.loc[mask, "date"].map(uplift).fillna(1.0).values * rate
                stk = g.loc[mask, "stock_qty"].astype(float).values
                for i in range(len(stk)):
                    dec = daily[i]
                    stk[i:] = np.maximum(0.0, stk[i:] - dec)
                g.loc[mask, "stock_qty"] = stk

        chunks.append(g)

    if not chunks:
        return out

    combined = pd.concat(chunks, ignore_index=True)
    combined = combined.sort_values(["center", "resource_code", "date"])
    combined["stock_qty"] = pd.to_numeric(combined["stock_qty"], errors="coerce")
    ffill_mask = ~combined["center"].isin(["In-Transit", "WIP"])
    combined.loc[ffill_mask, "stock_qty"] = (
        combined.loc[ffill_mask]
        .groupby(["center", "resource_code"])["stock_qty"]
        .ffill()
    )
    combined["stock_qty"] = combined["stock_qty"].fillna(0)
    combined["stock_qty"] = combined["stock_qty"].replace([np.inf, -np.inf], 0)
    combined["stock_qty"] = (
        combined["stock_qty"].round().clip(lower=0).astype(int)
    )

    return combined
