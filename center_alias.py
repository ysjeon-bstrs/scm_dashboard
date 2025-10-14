"""Utilities for normalising center names across the dashboard."""

from __future__ import annotations

from typing import Any, Optional

import math
import pandas as pd

# Centralised alias mapping so that all inputs converge to a single canonical
# name. Extend this dictionary when new aliases appear in upstream systems.
CENTER_ALIAS: dict[str, str] = {
    "어크로스비US": "AcrossBUS",  # Hangul alias → English canonical name
    "AcrossBUS": "AcrossBUS",
}

# Center values that should be ignored when building filter options. The
# comparison is case-insensitive to guard against inconsistent capitalisation
# in source files.
_IGNORED_CENTER_VALUES = {
    "",
    "nan",
    "none",
    "wip",
    "in-transit",
    "transit",
    "생산중",
    "production",
}
_IGNORED_CENTER_VALUES_CI = {value.casefold() for value in _IGNORED_CENTER_VALUES}


def normalize_center_series(series: pd.Series) -> pd.Series:
    """Return *series* with known center aliases replaced by canonical names."""

    normalized = series.astype(str).str.strip()
    return normalized.replace(CENTER_ALIAS)


def normalize_center_value(value: Any) -> Optional[str]:
    """Normalise a single center name for use in filters and lookups.

    The function trims whitespace, applies alias mapping and drops ignored
    placeholders (such as ``WIP`` or ``In-Transit``). ``None`` is returned for
    values that should not participate in downstream filters.
    """

    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if pd.isna(value):
        return None

    text = str(value).strip()
    if not text:
        return None

    normalized = CENTER_ALIAS.get(text, text)
    if normalized.casefold() in _IGNORED_CENTER_VALUES_CI:
        return None
    return normalized
