"""End-to-end orchestration helpers for the restructured SCM dashboard."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from .planning.timeline import TimelineBuilder, TimelineContext, prepare_moves, prepare_snapshot

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BuildInputs:
    snapshot: pd.DataFrame
    moves: pd.DataFrame


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
    logger.debug("Building timeline bundle with TimelineBuilder")

    context = TimelineContext(
        centers=list(centers),
        skus=list(skus),
        start=start,
        end=end,
        today=today,
        lag_days=lag_days,
        horizon_days=int(max(0, horizon_days)),
    )

    builder = TimelineBuilder(context)

    logger.debug("Preparing snapshot and move tables")
    snapshot_table = prepare_snapshot(inputs.snapshot)
    fallback_days = int(max(0, move_fallback_days))
    move_table = prepare_moves(
        inputs.moves,
        context=context,
        fallback_days=fallback_days,
    )

    logger.debug("Building timeline bundle")
    bundle = builder.build(snapshot_table, move_table)
    logger.debug(
        f"Bundle created: {len(bundle.center_lines)} center, "
        f"{len(bundle.in_transit_lines)} transit, {len(bundle.wip_lines)} WIP rows"
    )

    return bundle
