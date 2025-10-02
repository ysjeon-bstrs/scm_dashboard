"""KPI adapters that map to the existing v4 KPI calculators."""

from __future__ import annotations

import pandas as pd

from scm_dashboard_v4 import kpi as v4_kpi


def kpi_breakdown_per_sku(snapshot: pd.DataFrame, *, recent_days: int = 28) -> pd.DataFrame:
    return v4_kpi.kpi_breakdown_per_sku(snapshot, recent_days=recent_days)
