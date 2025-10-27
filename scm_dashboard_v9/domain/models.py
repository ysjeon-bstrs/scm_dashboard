"""
도메인 모델: SCM 대시보드의 핵심 데이터 구조

이 모듈은 SCM 대시보드에서 사용하는 정규화된 데이터 모델을 정의합니다.
모든 모델은 불변(frozen) 데이터클래스로 구현되어 안전한 데이터 전달을 보장합니다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import pandas as pd


@dataclass(frozen=True)
class SnapshotTable:
    """
    일별 재고 스냅샷 데이터를 나타내는 정규화된 테이블.

    스냅샷 데이터는 특정 날짜/센터/SKU 조합에서의 재고 수량을 나타냅니다.
    원본 데이터의 다양한 컬럼명을 표준화하여 일관된 스키마를 제공합니다.

    Attributes:
        data: date, center, resource_code, stock_qty 컬럼을 포함한 정규화된 데이터프레임

    Examples:
        >>> raw_df = pd.read_csv("snapshot.csv")
        >>> snapshot = SnapshotTable.from_dataframe(raw_df)
        >>> filtered = snapshot.filter(
        ...     centers=["태광KR", "AMZUS"],
        ...     skus=["BA00021"],
        ...     start=pd.Timestamp("2024-01-01"),
        ...     end=pd.Timestamp("2024-01-31")
        ... )
    """

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
        """
        원본 데이터프레임으로부터 정규화된 스냅샷 테이블을 생성합니다.

        다양한 원본 컬럼명을 표준화된 스키마로 변환합니다:
        - date: 날짜 (datetime64, 시간 부분 제거)
        - center: 센터명 (문자열)
        - resource_code: SKU 코드 (문자열)
        - stock_qty: 재고 수량 (숫자)

        Args:
            frame: 원본 데이터프레임
            date_column: 날짜 컬럼명 (기본값: "date")
            center_column: 센터 컬럼명 (기본값: "center")
            sku_column: SKU 컬럼명 (기본값: "resource_code")
            qty_column: 수량 컬럼명 (기본값: "stock_qty")

        Returns:
            정규화된 스냅샷 테이블 인스턴스

        Raises:
            KeyError: date 또는 snapshot_date 컬럼이 없을 경우
        """
        snap = frame.copy()

        # ========================================
        # 날짜 컬럼 찾기 및 정규화
        # ========================================
        # 'date' 컬럼이 없으면 'snapshot_date'를 대안으로 사용
        if date_column not in snap.columns and "snapshot_date" in snap.columns:
            date_column = "snapshot_date"
        if date_column not in snap.columns:
            raise KeyError("snapshot data must include a date or snapshot_date column")

        # ========================================
        # 컬럼 타입 정규화
        # ========================================
        # 날짜: datetime64로 변환하고 시간 부분을 자정(00:00:00)으로 정규화
        snap["date"] = pd.to_datetime(snap[date_column], errors="coerce").dt.normalize()

        # 센터: 문자열로 변환 (공백 제거는 상위 레이어에서 수행)
        snap["center"] = snap[center_column].astype(str)

        # SKU: 문자열로 변환
        snap["resource_code"] = snap[sku_column].astype(str)

        # 수량: 숫자로 변환 (변환 실패 시 NaN)
        snap["stock_qty"] = pd.to_numeric(snap[qty_column], errors="coerce")

        # ========================================
        # 유효하지 않은 행 제거
        # ========================================
        # 날짜가 NaT인 행은 제거 (필수 필드)
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
        """
        센터, SKU, 날짜 범위로 스냅샷 데이터를 필터링합니다.

        Args:
            centers: 필터링할 센터명 목록
            skus: 필터링할 SKU 목록
            start: 시작 날짜 (포함)
            end: 종료 날짜 (포함)

        Returns:
            필터링된 데이터프레임. 조건에 맞는 데이터가 없으면 빈 데이터프레임 반환.
        """
        # ========================================
        # 입력 검증
        # ========================================
        centers_set = {str(c) for c in centers}
        skus_set = {str(s) for s in skus}

        # 센터나 SKU가 비어있으면 빈 결과 반환
        if not centers_set or not skus_set:
            return pd.DataFrame(
                columns=["date", "center", "resource_code", "stock_qty"]
            )

        # ========================================
        # 날짜 정규화
        # ========================================
        start_dt = pd.to_datetime(start).normalize()
        end_dt = pd.to_datetime(end).normalize()

        # ========================================
        # 조건부 필터링
        # ========================================
        filtered = self.data[
            self.data["center"].isin(centers_set)
            & self.data["resource_code"].isin(skus_set)
            & (self.data["date"] >= start_dt)
            & (self.data["date"] <= end_dt)
        ]

        if filtered.empty:
            return pd.DataFrame(
                columns=["date", "center", "resource_code", "stock_qty"]
            )

        return filtered.copy()


@dataclass(frozen=True)
class MoveTable:
    """
    재고 이동 원장 테이블 (스케줄 메타데이터 포함).

    이동 원장은 센터 간 재고 이동, WIP(생산중), 입고 등을 기록합니다.
    스케줄링 로직을 통해 예상 입고일 등의 메타데이터가 추가됩니다.

    Attributes:
        data: 정규화된 이동 원장 데이터프레임.
              주요 컬럼: from_center, to_center, resource_code, carrier_mode,
                        qty_ea, onboard_date, arrival_date, inbound_date, event_date

    Examples:
        >>> raw_moves = pd.read_csv("moves.csv")
        >>> moves = MoveTable(raw_moves)
        >>> amzus_moves = moves.slice_by_center(center="AMZUS", include_wip=False)
    """

    data: pd.DataFrame

    def filter(self, *, skus: Sequence[str]) -> pd.DataFrame:
        """
        특정 SKU 목록으로 이동 원장을 필터링합니다.

        Args:
            skus: 필터링할 SKU 코드 목록

        Returns:
            필터링된 이동 원장 데이터프레임.
            SKU가 비어있으면 빈 데이터프레임 반환.
        """
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
        """
        특정 목적지 센터로 향하는 이동만 추출합니다.

        Args:
            center: 목적지 센터명
            include_wip: WIP(생산중) 이동 포함 여부. False면 일반 운송만 포함.

        Returns:
            해당 센터로 향하는 이동 데이터프레임.

        Examples:
            >>> # 태광KR로 향하는 모든 이동 (WIP 포함)
            >>> moves.slice_by_center(center="태광KR", include_wip=True)

            >>> # AMZUS로 향하는 운송만 (WIP 제외)
            >>> moves.slice_by_center(center="AMZUS", include_wip=False)
        """
        frame = self.data.copy()

        # ========================================
        # 목적지 센터 필터링
        # ========================================
        mask = frame["to_center"].eq(str(center))

        # ========================================
        # WIP 제외 옵션 처리
        # ========================================
        # include_wip=False면 carrier_mode가 "WIP"인 행 제외
        if not include_wip:
            mask &= frame["carrier_mode"] != "WIP"

        return frame[mask]


@dataclass(frozen=True)
class TimelineBundle:
    """
    타임라인 빌드 결과를 그룹화한 번들.

    타임라인은 세 가지 라인으로 구분됩니다:
    1. center_lines: 센터 내 재고 (입고 완료된 재고)
    2. in_transit_lines: 운송 중인 재고 (출발했으나 아직 입고 전)
    3. wip_lines: 생산 중인 재고 (WIP, Work In Process)

    Attributes:
        center_lines: 센터 재고 타임라인
        in_transit_lines: 운송 중 재고 타임라인
        wip_lines: 생산 중 재고 타임라인

    Examples:
        >>> bundle = TimelineBundle(
        ...     center_lines=center_df,
        ...     in_transit_lines=transit_df,
        ...     wip_lines=wip_df
        ... )
        >>> combined = bundle.concat()  # 모든 라인을 하나의 데이터프레임으로 통합
    """

    center_lines: pd.DataFrame
    in_transit_lines: pd.DataFrame
    wip_lines: pd.DataFrame

    def concat(self) -> pd.DataFrame:
        """
        세 가지 타임라인을 하나의 데이터프레임으로 결합합니다.

        빈 데이터프레임은 자동으로 제외되며, 수량은 정수로 반올림됩니다.
        음수는 0으로 클리핑되어 재고가 음수가 되지 않도록 방지합니다.

        Returns:
            통합된 타임라인 데이터프레임.
            date, center, resource_code, stock_qty 컬럼 포함.
        """
        # ========================================
        # 비어있지 않은 데이터프레임만 수집
        # ========================================
        frames = [
            df
            for df in (self.center_lines, self.in_transit_lines, self.wip_lines)
            if not df.empty
        ]

        # ========================================
        # 모든 라인이 비어있으면 빈 데이터프레임 반환
        # ========================================
        if not frames:
            return pd.DataFrame(
                columns=["date", "center", "resource_code", "stock_qty"]
            )

        # ========================================
        # 데이터프레임 결합 및 정규화
        # ========================================
        combined = pd.concat(frames, ignore_index=True)

        # 수량을 숫자로 변환 (변환 실패 시 0)
        combined["stock_qty"] = pd.to_numeric(
            combined["stock_qty"], errors="coerce"
        ).fillna(0)

        # 음수 제거 및 정수 변환
        combined["stock_qty"] = combined["stock_qty"].clip(lower=0).round().astype(int)

        return combined
