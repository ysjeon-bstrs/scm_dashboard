"""Domain models representing normalized SCM dashboard tables."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import pandas as pd


@dataclass(frozen=True)
class SnapshotTable:
    """Normalized view of the daily inventory snapshot feed."""

    data: pd.DataFrame

    @classmethod
    def from_dataframe(
        cls,
        frame: pd.DataFrame,
        *,
        date_column: str = "date",
        center_column: str = "center",
        sku_column: str = "resource_code",
        qty_column: str = "stock_qty",
    ) -> "SnapshotTable":
        snap = frame.copy()
        if date_column not in snap.columns and "snapshot_date" in snap.columns:
            date_column = "snapshot_date"
        if date_column not in snap.columns:
            raise KeyError("snapshot data must include a date or snapshot_date column")

        snap["date"] = pd.to_datetime(snap[date_column], errors="coerce").dt.normalize()
        snap["center"] = snap[center_column].astype(str)
        snap["resource_code"] = snap[sku_column].astype(str)
        snap["stock_qty"] = pd.to_numeric(snap[qty_column], errors="coerce")
        snap = snap.dropna(subset=["date"])
        return cls(snap[["date", "center", "resource_code", "stock_qty"]])

    def filter(
        self,
        *,
        centers: Iterable[str],
        skus: Iterable[str],
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> pd.DataFrame:
        centers_set = {str(c) for c in centers}
        skus_set = {str(s) for s in skus}
        if not centers_set or not skus_set:
            return pd.DataFrame(columns=["date", "center", "resource_code", "stock_qty"])

        start_dt = pd.to_datetime(start).normalize()
        end_dt = pd.to_datetime(end).normalize()
        filtered = self.data[
            self.data["center"].isin(centers_set)
            & self.data["resource_code"].isin(skus_set)
            & (self.data["date"] >= start_dt)
            & (self.data["date"] <= end_dt)
        ]
        if filtered.empty:
            return pd.DataFrame(columns=["date", "center", "resource_code", "stock_qty"])
        return filtered.copy()


@dataclass(frozen=True)
class MoveTable:
    """Normalized movement table enriched with schedule metadata."""

    data: pd.DataFrame

    def filter(self, *, skus: Sequence[str]) -> pd.DataFrame:
        skus_set = {str(s) for s in skus}
        if not skus_set:
            return pd.DataFrame(columns=self.data.columns)
        return self.data[self.data["resource_code"].isin(skus_set)].copy()

    def slice_by_center(
        self,
        *,
        center: str,
        include_wip: bool = True,
    ) -> pd.DataFrame:
        frame = self.data.copy()
        mask = frame["to_center"].eq(str(center))
        if not include_wip:
            mask &= frame["carrier_mode"] != "WIP"
        return frame[mask]


@dataclass(frozen=True)
class TimelineBundle:
    """Grouped timeline outputs for centers, in-transit, and WIP lines."""

    center_lines: pd.DataFrame
    in_transit_lines: pd.DataFrame
    wip_lines: pd.DataFrame

    def concat(self) -> pd.DataFrame:
        frames = [df for df in (self.center_lines, self.in_transit_lines, self.wip_lines) if not df.empty]
        if not frames:
            return pd.DataFrame(columns=["date", "center", "resource_code", "stock_qty"])
        combined = pd.concat(frames, ignore_index=True)
        combined["stock_qty"] = pd.to_numeric(combined["stock_qty"], errors="coerce").fillna(0)
        combined["stock_qty"] = combined["stock_qty"].clip(lower=0).round().astype(int)
        return combined
