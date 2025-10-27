"""Utilities for normalising center names across the dashboard."""

from __future__ import annotations

from typing import Any, Optional

import math
import pandas as pd

# Centralised alias mapping so that all inputs converge to a single canonical
# name. Extend this dictionary when new aliases appear in upstream systems.
CENTER_ALIAS: dict[str, str] = {
    # Amazon US
    "AMZUS": "AMZUS",
    "아마존US": "AMZUS",
    "AmazonUS": "AMZUS",
    "Amazon US": "AMZUS",
    "AMZ US": "AMZUS",
    "AMZ-US": "AMZUS",
    "AMZ_US": "AMZUS",
    # Across B US fulfilment centre
    "AcrossBUS": "AcrossBUS",
    "Across B US": "AcrossBUS",
    "Across-B US": "AcrossBUS",
    "Across_B US": "AcrossBUS",
    "어크로스비US": "AcrossBUS",
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


def _alias_key(value: str) -> str:
    """Create a normalised lookup key by stripping separators and lowering case."""

    return "".join(ch for ch in value.casefold() if ch.isalnum())


_CENTER_ALIAS_LOOKUP = {
    _alias_key(key): canonical for key, canonical in CENTER_ALIAS.items()
}


def _apply_alias(text: str) -> str:
    key = _alias_key(text)
    return _CENTER_ALIAS_LOOKUP.get(key, text)


def normalize_center_series(series: pd.Series) -> pd.Series:
    """Return *series* with known center aliases replaced by canonical names."""

    normalized = series.astype(str).str.strip()
    mask_placeholder = normalized.str.casefold().isin({"nan", "none", "null", "<na>"})
    normalized.loc[mask_placeholder] = ""
    return normalized.map(_apply_alias)


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

    lowered = text.casefold()
    if lowered in {"", "nan", "none", "null", "<na>"}:
        return None

    normalized = _CENTER_ALIAS_LOOKUP.get(_alias_key(text))
    if not normalized:
        normalized = text
    if normalized.casefold() in _IGNORED_CENTER_VALUES_CI:
        return None
    return normalized
