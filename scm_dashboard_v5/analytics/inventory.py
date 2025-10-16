"""Analytics adapters exposing the v4 inventory helpers under the new structure."""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from scm_dashboard_v4 import inventory as v4_inventory


def pivot_inventory_cost_from_raw(
    raw: pd.DataFrame,
    latest_dt: pd.Timestamp,
    centers: List[str],
    center_latest_dates: Optional[Dict[str, pd.Timestamp]] = None,
) -> pd.DataFrame:
    return v4_inventory.pivot_inventory_cost_from_raw(
        raw, latest_dt, centers, center_latest_dates
    )
