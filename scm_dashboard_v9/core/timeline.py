"""Wrapper utilities for constructing timeline data frames."""

from __future__ import annotations

from typing import Sequence

import pandas as pd

from ..pipeline import BuildInputs, build_timeline_bundle


def build_timeline(
    snapshot: pd.DataFrame,
    moves: pd.DataFrame,
    *,
    centers: Sequence[str],
    skus: Sequence[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    today: pd.Timestamp,
    lag_days: int = 5,
    horizon_days: int = 0,
    move_fallback_days: int = 1,
) -> pd.DataFrame:
    """Return the concatenated timeline bundle for the given filters."""

    inputs = BuildInputs(snapshot=snapshot, moves=moves)
    bundle = build_timeline_bundle(
        inputs,
        centers=list(centers),
        skus=list(skus),
        start=start,
        end=end,
        today=today,
        lag_days=lag_days,
        horizon_days=int(max(0, horizon_days)),
        move_fallback_days=int(max(0, move_fallback_days)),
    )
    timeline = bundle.concat()
    if timeline.empty:
        return timeline
    # Normalise timestamps to midnight for consistent downstream filtering.
    timeline = timeline.copy()
    if "date" in timeline.columns:
        timeline["date"] = pd.to_datetime(timeline["date"], errors="coerce").dt.normalize()
    return timeline
