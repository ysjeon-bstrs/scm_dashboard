from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class DashboardSourceData:
    """Normalized data bundle loaded from an external source."""

    moves: pd.DataFrame
    snapshot: pd.DataFrame
    wip: pd.DataFrame
    snapshot_raw: Optional[pd.DataFrame] = None
