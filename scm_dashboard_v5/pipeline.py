"""End-to-end orchestration helpers for the restructured SCM dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from .planning.timeline import TimelineBuilder, TimelineContext, prepare_moves, prepare_snapshot


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
):
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
    snapshot_table = prepare_snapshot(inputs.snapshot)
    move_table = prepare_moves(inputs.moves, context=context)
    return builder.build(snapshot_table, move_table)
