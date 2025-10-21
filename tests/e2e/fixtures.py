"""Shared fixtures for end-to-end pipeline tests."""

from __future__ import annotations

import pandas as pd

DEFAULT_CENTERS = ["C1", "AMZUS"]
DEFAULT_SKUS = ["SKU1", "SKU2"]
DEFAULT_START = pd.Timestamp("2024-01-01")
DEFAULT_END = pd.Timestamp("2024-01-10")
DEFAULT_TODAY = pd.Timestamp("2024-01-04")
DEFAULT_LOOKBACK_DAYS = 7
DEFAULT_LAG_DAYS = 2


def load_sample_snapshot() -> pd.DataFrame:
    """Return a deterministic snapshot sample spanning two centres."""

    data = [
        {"snapshot_date": "2023-12-30", "center": "C1", "resource_code": "SKU1", "stock_qty": 110},
        {"snapshot_date": "2023-12-31", "center": "C1", "resource_code": "SKU1", "stock_qty": 105},
        {"snapshot_date": "2024-01-01", "center": "C1", "resource_code": "SKU1", "stock_qty": 100},
        {"snapshot_date": "2024-01-02", "center": "C1", "resource_code": "SKU1", "stock_qty": 95},
        {"snapshot_date": "2024-01-03", "center": "C1", "resource_code": "SKU1", "stock_qty": 90},
        {"snapshot_date": "2024-01-04", "center": "C1", "resource_code": "SKU1", "stock_qty": 85},
        {"snapshot_date": "2023-12-31", "center": "C1", "resource_code": "SKU2", "stock_qty": 60},
        {"snapshot_date": "2024-01-01", "center": "C1", "resource_code": "SKU2", "stock_qty": 58},
        {"snapshot_date": "2024-01-02", "center": "C1", "resource_code": "SKU2", "stock_qty": 56},
        {"snapshot_date": "2024-01-03", "center": "C1", "resource_code": "SKU2", "stock_qty": 53},
        {"snapshot_date": "2024-01-04", "center": "C1", "resource_code": "SKU2", "stock_qty": 50},
        {"snapshot_date": "2023-12-30", "center": "AMZUS", "resource_code": "SKU1", "stock_qty": 44},
        {"snapshot_date": "2023-12-31", "center": "AMZUS", "resource_code": "SKU1", "stock_qty": 42},
        {"snapshot_date": "2024-01-01", "center": "AMZUS", "resource_code": "SKU1", "stock_qty": 40},
        {"snapshot_date": "2024-01-02", "center": "AMZUS", "resource_code": "SKU1", "stock_qty": 38},
        {"snapshot_date": "2024-01-03", "center": "AMZUS", "resource_code": "SKU1", "stock_qty": 37},
        {"snapshot_date": "2024-01-04", "center": "AMZUS", "resource_code": "SKU1", "stock_qty": 36},
        {"snapshot_date": "2023-12-31", "center": "AMZUS", "resource_code": "SKU2", "stock_qty": 26},
        {"snapshot_date": "2024-01-01", "center": "AMZUS", "resource_code": "SKU2", "stock_qty": 24},
        {"snapshot_date": "2024-01-02", "center": "AMZUS", "resource_code": "SKU2", "stock_qty": 23},
        {"snapshot_date": "2024-01-03", "center": "AMZUS", "resource_code": "SKU2", "stock_qty": 21},
        {"snapshot_date": "2024-01-04", "center": "AMZUS", "resource_code": "SKU2", "stock_qty": 20},
    ]
    return pd.DataFrame(data)


def load_sample_moves() -> pd.DataFrame:
    """Return representative movement records including WIP and fallback flows."""

    data = [
        {
            "resource_code": "SKU1",
            "qty_ea": 15,
            "from_center": "C1",
            "to_center": "AMZUS",
            "carrier_mode": "AIR",
            "onboard_date": "2024-01-03",
            "arrival_date": "2024-01-05",
        },
        {
            "resource_code": "SKU1",
            "qty_ea": 25,
            "from_center": "SUP",
            "to_center": "C1",
            "carrier_mode": "SEA",
            "onboard_date": "2024-01-06",
            "arrival_date": "2024-01-08",
            "inbound_date": "2024-01-09",
        },
        {
            "resource_code": "SKU2",
            "qty_ea": 18,
            "from_center": "SUP",
            "to_center": "AMZUS",
            "carrier_mode": "SEA",
            "onboard_date": "2023-12-30",
            "arrival_date": "2024-01-02",
            "inbound_date": "2024-01-03",
        },
        {
            "resource_code": "SKU2",
            "qty_ea": 5,
            "from_center": "Factory",
            "to_center": "C1",
            "carrier_mode": "WIP",
            "onboard_date": "2024-01-01",
            "event_date": "2024-01-07",
        },
        {
            "resource_code": "SKU1",
            "qty_ea": 12,
            "from_center": "SUP",
            "to_center": "AMZUS",
            "carrier_mode": "SEA",
            "onboard_date": "2024-01-05",
        },
    ]
    frame = pd.DataFrame(data)
    date_columns = [
        "onboard_date",
        "arrival_date",
        "inbound_date",
        "event_date",
    ]
    for column in date_columns:
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    return frame


def load_sample_snapshot_long() -> pd.DataFrame:
    """Return the snapshot sample with a ``date`` column for consumption calls."""

    snap = load_sample_snapshot()
    snap_long = snap.rename(columns={"snapshot_date": "date"}).copy()
    snap_long["date"] = pd.to_datetime(snap_long["date"], errors="coerce")
    return snap_long


def load_sample_snapshot_raw() -> pd.DataFrame:
    """Return Amazon-specific raw snapshot rows containing daily FBA outbounds."""

    data = [
        {"snapshot_date": "2023-12-28", "center": "AMZUS", "resource_code": "SKU1", "fba_output_stock": 6},
        {"snapshot_date": "2023-12-29", "center": "AMZUS", "resource_code": "SKU1", "fba_output_stock": 7},
        {"snapshot_date": "2023-12-30", "center": "AMZUS", "resource_code": "SKU1", "fba_output_stock": 5},
        {"snapshot_date": "2023-12-31", "center": "AMZUS", "resource_code": "SKU1", "fba_output_stock": 8},
        {"snapshot_date": "2024-01-01", "center": "AMZUS", "resource_code": "SKU1", "fba_output_stock": 9},
        {"snapshot_date": "2024-01-02", "center": "AMZUS", "resource_code": "SKU1", "fba_output_stock": 6},
        {"snapshot_date": "2023-12-29", "center": "AMZUS", "resource_code": "SKU2", "fba_output_stock": 4},
        {"snapshot_date": "2023-12-30", "center": "AMZUS", "resource_code": "SKU2", "fba_output_stock": 3},
        {"snapshot_date": "2023-12-31", "center": "AMZUS", "resource_code": "SKU2", "fba_output_stock": 5},
        {"snapshot_date": "2024-01-01", "center": "AMZUS", "resource_code": "SKU2", "fba_output_stock": 4},
        {"snapshot_date": "2024-01-02", "center": "AMZUS", "resource_code": "SKU2", "fba_output_stock": 5},
    ]
    return pd.DataFrame(data)

