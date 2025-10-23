"""Scheduling helpers for inbound move projections."""

from __future__ import annotations

import pandas as pd

from ..domain.normalization import normalize_dates


def annotate_move_schedule(
    moves: pd.DataFrame,
    *,
    today: pd.Timestamp,
    lag_days: int,
    horizon_end: pd.Timestamp,
    fallback_days: int = 1,
) -> pd.DataFrame:
    """Attach predicted inbound dates aligned with the centre inventory policy."""

    today_norm = pd.to_datetime(today).normalize()
    horizon_end = pd.to_datetime(horizon_end).normalize()
    fallback_date = min(
        today_norm + pd.Timedelta(days=int(fallback_days)),
        horizon_end + pd.Timedelta(days=1),
    )

    out = normalize_dates(moves)
    out["carrier_mode"] = out.get("carrier_mode", "").astype(str).str.upper()
    actual_onboard = pd.to_datetime(out.get("onboard_date"), errors="coerce").dt.normalize()
    out["_onboard_date_actual"] = actual_onboard

    pred = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns]")
    inbound_col = out.get("inbound_date")
    has_inbound = inbound_col.notna() if inbound_col is not None else pd.Series(False, index=out.index)
    if has_inbound.any():
        pred.loc[has_inbound] = inbound_col.loc[has_inbound]

    arrival_raw = out.get("arrival_date")
    if isinstance(arrival_raw, pd.Series):
        arrival_series = pd.to_datetime(arrival_raw, errors="coerce").dt.normalize()
    else:
        arrival_series = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns]")

    eta_raw = out.get("eta_date")
    if isinstance(eta_raw, pd.Series):
        eta_series = pd.to_datetime(eta_raw, errors="coerce").dt.normalize()
    else:
        eta_series = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns]")
    effective_arrival = arrival_series.fillna(eta_series)
    has_arrival = (~has_inbound) & effective_arrival.notna()
    if has_arrival.any():
        arr_dates = effective_arrival
        past_arrival = has_arrival & (arr_dates <= today_norm)
        if past_arrival.any():
            pred.loc[past_arrival] = today_norm + pd.Timedelta(days=int(lag_days))

        future_arrival = has_arrival & (arr_dates > today_norm)
        if future_arrival.any():
            pred.loc[future_arrival] = arr_dates.loc[future_arrival] + pd.Timedelta(
                days=int(lag_days)
            )

    has_signal = has_inbound | has_arrival
    need_fallback = has_signal & pred.isna()
    if need_fallback.any():
        pred.loc[need_fallback] = fallback_date

    pred = pred.where(has_signal, pd.NaT)
    out["pred_inbound_date"] = pd.to_datetime(pred).dt.normalize()
    out["pred_inbound_date"] = out["pred_inbound_date"].clip(upper=horizon_end + pd.Timedelta(days=1))
    out["in_transit_end_date"] = out["pred_inbound_date"]
    return out
