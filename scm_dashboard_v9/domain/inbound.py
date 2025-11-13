"""Inbound domain utilities."""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Tuple

import pandas as pd

LeadtimeKey = Tuple[str, str, str]


def _normalize_key(value: object) -> str:
    """Normalize key components to string for lookup consistency."""

    if pd.isna(value):
        return ""
    return str(value)


def build_lt_inbound_map(leadtime_df: pd.DataFrame | None) -> Dict[LeadtimeKey, float]:
    """Build a lookup map for 출발→입고 평균 리드타임."""

    if leadtime_df is None or leadtime_df.empty:
        return {}

    if "avg_lt_depart_to_inbound" not in leadtime_df.columns:
        return {}

    required_cols: Iterable[str] = ("from_center", "to_center", "carrier_mode")
    if not set(required_cols).issubset(leadtime_df.columns):
        return {}

    map_dict: Dict[LeadtimeKey, float] = {}
    for _, row in leadtime_df.dropna(subset=["avg_lt_depart_to_inbound"]).iterrows():
        key = tuple(_normalize_key(row[col]) for col in required_cols)
        map_dict[key] = float(row["avg_lt_depart_to_inbound"])

    return map_dict


def assign_expected_inbound_dates(
    frame: pd.DataFrame,
    leadtime_map: Mapping[LeadtimeKey, float],
    *,
    from_col: str = "from_center",
    to_col: str = "to_center",
    carrier_col: str = "carrier_mode",
    onboard_col: str = "onboard_date",
    target_col: str = "expected_inbound_date",
) -> pd.DataFrame:
    """Assign expected inbound dates based on leadtime map."""

    if frame.empty:
        result = frame.copy()
        result[target_col] = pd.NaT
        return result

    result = frame.copy()

    if not leadtime_map:
        result[target_col] = pd.NaT
        return result

    key_series = pd.Series(
        list(
            zip(
                result[from_col].map(_normalize_key),
                result[to_col].map(_normalize_key),
                result[carrier_col].map(_normalize_key),
            )
        ),
        index=result.index,
    )

    leadtime_series = key_series.map(leadtime_map)
    onboard_series = pd.to_datetime(result[onboard_col], errors="coerce")
    timedelta_series = pd.to_timedelta(leadtime_series, unit="D")
    expected_series = onboard_series + timedelta_series
    expected_series = expected_series.where(
        leadtime_series.notna() & onboard_series.notna(),
        pd.NaT,
    )

    result[target_col] = expected_series
    return result
