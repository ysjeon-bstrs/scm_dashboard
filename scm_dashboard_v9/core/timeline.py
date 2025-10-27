"""Wrapper utilities for constructing timeline data frames."""

from __future__ import annotations

import logging
from typing import Sequence

import pandas as pd

from ..common.performance import measure_time
from ..pipeline import BuildInputs, build_timeline_bundle

logger = logging.getLogger(__name__)


@measure_time
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

    logger.info(
        f"Building timeline: {len(centers)} centers, {len(skus)} SKUs, "
        f"period {start.date()} to {end.date()}, lag_days={lag_days}"
    )

    inputs = BuildInputs(snapshot=snapshot, moves=moves)
    logger.debug(f"Input data: {len(snapshot)} snapshot rows, {len(moves)} move rows")

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
        logger.warning("Timeline is empty - no data for given filters")
        return timeline

    logger.info(f"Timeline built successfully: {len(timeline)} rows")

    # Normalise timestamps to midnight for consistent downstream filtering.
    timeline = timeline.copy()
    if "date" in timeline.columns:
        timeline["date"] = pd.to_datetime(timeline["date"], errors="coerce").dt.normalize()

    return timeline
