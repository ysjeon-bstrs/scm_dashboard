"""Analytics adapters exposing the v4 inventory helpers under the new structure."""

from __future__ import annotations

import pandas as pd

from scm_dashboard_v4 import inventory as v4_inventory


def pivot_inventory_cost_from_raw(raw: pd.DataFrame) -> pd.DataFrame:
    return v4_inventory.pivot_inventory_cost_from_raw(raw)
