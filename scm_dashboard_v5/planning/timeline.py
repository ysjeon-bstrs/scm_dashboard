"""High level orchestration for building dashboard timelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from ..domain.models import MoveTable, SnapshotTable, TimelineBundle
from ..domain.normalization import normalize_moves, normalize_snapshot
from .schedule import annotate_move_schedule
from .series import SeriesIndex, build_center_series, build_in_transit_series, build_wip_series


@dataclass(frozen=True)
class TimelineContext:
    centers: Iterable[str]
    skus: Iterable[str]
    start: pd.Timestamp
    end: pd.Timestamp
    today: pd.Timestamp
    lag_days: int = 7

    @property
    def index(self) -> SeriesIndex:
        return SeriesIndex(
            start=pd.to_datetime(self.start).normalize(),
            end=pd.to_datetime(self.end).normalize(),
        )


class TimelineBuilder:
    """Constructs the centre, in-transit, and WIP stock time series."""

    def __init__(self, context: TimelineContext) -> None:
        self.context = context

    def build(
        self,
        snapshot: SnapshotTable,
        moves: MoveTable,
    ) -> TimelineBundle:
        index = self.context.index
        centers = list(self.context.centers)
        skus = list(self.context.skus)

        center_lines = build_center_series(
            snapshot=snapshot.filter(
                centers=centers,
                skus=skus,
                start=index.start,
                end=index.end,
            ),
            moves=moves.data,
            centers=centers,
            skus=skus,
            index=index,
        )

        in_transit = build_in_transit_series(
            moves=moves.data,
            centers=centers,
            skus=skus,
            index=index,
            today=self.context.today,
            lag_days=self.context.lag_days,
        )

        wip = build_wip_series(
            moves=moves.data,
            skus=skus,
            index=index,
        )

        return TimelineBundle(center_lines=center_lines, in_transit_lines=in_transit, wip_lines=wip)


def prepare_moves(
    moves: pd.DataFrame,
    *,
    context: TimelineContext,
    fallback_days: int = 1,
) -> MoveTable:
    normalized = normalize_moves(moves)
    scheduled = annotate_move_schedule(
        normalized,
        today=context.today,
        lag_days=context.lag_days,
        horizon_end=context.end,
        fallback_days=fallback_days,
    )
    return MoveTable(scheduled)


def prepare_snapshot(snapshot: pd.DataFrame) -> SnapshotTable:
    normalized = normalize_snapshot(snapshot)
    return SnapshotTable(normalized)
