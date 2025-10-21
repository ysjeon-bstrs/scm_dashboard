"""Thin application-level wrappers around the v5 timeline pipeline."""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from scm_dashboard_v5.pipeline import BuildInputs as _BuildInputs
from scm_dashboard_v5.pipeline import build_timeline_bundle as _build_timeline_bundle

BuildInputs = _BuildInputs

__all__ = ["BuildInputs", "build_timeline_bundle"]


def build_timeline_bundle(
    inputs: BuildInputs,
    *,
    centers: Iterable[str],
    skus: Iterable[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    today: pd.Timestamp,
    lag_days: int = 7,
    horizon_days: int = 0,
    move_fallback_days: int = 1,
):
    """Delegate to the v5 timeline pipeline without altering behaviour."""

    return _build_timeline_bundle(
        inputs,
        centers=centers,
        skus=skus,
        start=start,
        end=end,
        today=today,
        lag_days=lag_days,
        horizon_days=horizon_days,
        move_fallback_days=move_fallback_days,
    )
