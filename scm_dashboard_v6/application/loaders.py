"""Thin forwarding wrappers for the existing data loader implementations."""

from __future__ import annotations

from typing import Optional, Tuple

import pandas as pd

from scm_dashboard_v4.loaders import (
    load_from_excel as _load_from_excel,
    load_from_gsheet_api as _load_from_gsheet_api,
    load_snapshot_raw as _load_snapshot_raw,
)

__all__ = [
    "load_from_excel",
    "load_from_gsheet_api",
    "load_snapshot_raw",
]


def load_from_gsheet_api() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return Google Sheets frames exactly as provided by the v4 loader."""

    return _load_from_gsheet_api()


def load_from_excel(
    file,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Return Excel frames exactly as provided by the v4 loader."""

    return _load_from_excel(file)


def load_snapshot_raw() -> pd.DataFrame:
    """Return the cached snapshot raw frame via the v4 loader implementation."""

    return _load_snapshot_raw()
