"""재고 및 판매 예측 함수.

재고 소진을 고려한 판매 예측 로직을 제공합니다.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .sales import make_forecast_sales_capped

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
        if "sales_ea" in sales_history.columns:
            coerced = pd.to_numeric(sales_history["sales_ea"], errors="coerce")
        else:
            coerced = pd.Series(0.0, index=sales_history.index, dtype=float)
        sales_history["sales_ea"] = coerced.fillna(0.0)
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

    if filtered_sales.empty:
        pivot = pd.DataFrame(index=index)
        pivot.index.name = "date"
        pivot.columns = pd.MultiIndex(
            levels=[[], []], codes=[[], []], names=["center", "resource_code"]
        )
    else:
        filtered_sales["date"] = pd.to_datetime(
            filtered_sales["date"], errors="coerce"
        ).dt.normalize()
        grouped_sales = (
            filtered_sales.groupby(["date", "center", "resource_code"], as_index=False)[
                "sales_ea"
            ]
            .sum()
            .sort_values("date")
        )
        pivot = (
            grouped_sales.pivot_table(
                index="date",
                columns=["center", "resource_code"],
                values="sales_ea",
                aggfunc="sum",
            )
            .sort_index()
        )
        pivot.index = pd.DatetimeIndex(pivot.index)
        pivot = pivot.asfreq("D", fill_value=0)

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

        base_rate = float(mean_rate.get((center_value, sku), 0.0))
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
    history["sales_ea"] = pd.to_numeric(history["sales_ea"], errors="coerce").fillna(0)
    history["sales_ea"] = history["sales_ea"].round().clip(lower=0).astype(int)

    combined_sales = pd.concat([history, forecast_sales], ignore_index=True)
    combined_sales = combined_sales.sort_values(["resource_code", "date"]).drop_duplicates(
        subset=["date", "center", "resource_code"], keep="last"
    )

    return combined_sales.reset_index(drop=True), forecast_inv.reset_index(drop=True)



