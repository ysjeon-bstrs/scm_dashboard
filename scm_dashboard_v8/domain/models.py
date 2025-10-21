"""v8 도메인 계층의 핵심 테이블 모델 정의."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import pandas as pd


@dataclass(frozen=True)
class SnapshotTable:
    """정규화된 재고 스냅샷을 보관하는 불변 데이터 클래스."""

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
        """다양한 컬럼 명을 허용하면서 내부 표준 스키마(date/center/resource_code/stock_qty)로 변환한다."""

        snap = frame.copy()
        if date_column not in snap.columns and "snapshot_date" in snap.columns:
            date_column = "snapshot_date"
        if date_column not in snap.columns:
            raise KeyError("snapshot 데이터에는 날짜 컬럼(date 또는 snapshot_date)이 필요합니다.")

        # ✅ 날짜·센터·SKU·재고 수량 컬럼을 표준화한다.
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
        """센터/상품/기간 조건으로 부분집합을 추출한다."""

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
    """정규화된 입출고(move) 데이터를 저장하는 래퍼."""

    data: pd.DataFrame

    def filter(self, *, skus: Sequence[str]) -> pd.DataFrame:
        """주어진 SKU 목록만 포함하는 서브셋을 반환한다."""

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
        """특정 센터로 향하는 move 데이터만 선택한다."""

        frame = self.data.copy()
        mask = frame["to_center"].eq(str(center))
        if not include_wip:
            mask &= frame["carrier_mode"] != "WIP"
        return frame[mask]


@dataclass(frozen=True)
class TimelineBundle:
    """센터/이동경로별 타임라인 결과를 묶어서 제공하는 컨테이너."""

    center_lines: pd.DataFrame
    in_transit_lines: pd.DataFrame
    wip_lines: pd.DataFrame

    def concat(self) -> pd.DataFrame:
        """세 가지 타임라인을 하나의 DataFrame으로 결합하여 분석이 용이하도록 반환한다."""

        frames = [df for df in (self.center_lines, self.in_transit_lines, self.wip_lines) if not df.empty]
        if not frames:
            return pd.DataFrame(columns=["date", "center", "resource_code", "stock_qty"])
        combined = pd.concat(frames, ignore_index=True)
        # ✅ 수량 컬럼을 안전하게 정수형으로 정규화한다.
        combined["stock_qty"] = pd.to_numeric(combined["stock_qty"], errors="coerce").fillna(0)
        combined["stock_qty"] = combined["stock_qty"].clip(lower=0).round().astype(int)
        return combined
