"""Series builders for centre, in-transit, and WIP timelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Set

import numpy as np
import pandas as pd


TIMELINE_COLUMNS = ["date", "center", "resource_code", "stock_qty"]


def _empty_timeline() -> pd.DataFrame:
    """Return a shared empty frame matching the timeline schema."""

    return pd.DataFrame(columns=TIMELINE_COLUMNS)


def _normalise_selection(values: Iterable[str]) -> Set[str]:
    """Return a cleaned-up selection set used across builders."""

    normalised = {str(value).strip() for value in values if value is not None}
    return {value for value in normalised if value}


def _resolve_onboard_column(frame: pd.DataFrame) -> str:
    """Pick the effective onboarding date column present in *frame*."""

    if "_onboard_date_actual" in frame.columns:
        return "_onboard_date_actual"
    return "onboard_date"


@dataclass(frozen=True)
class SeriesIndex:
    start: pd.Timestamp
    end: pd.Timestamp

    @property
    def range(self) -> pd.DatetimeIndex:
        return pd.date_range(self.start, self.end, freq="D")


def build_center_series(
    snapshot: pd.DataFrame,
    moves: pd.DataFrame,
    *,
    centers: Iterable[str],
    skus: Iterable[str],
    index: SeriesIndex,
) -> pd.DataFrame:
    """센터별 재고 타임라인을 생성하고 예정된 입출고를 반영합니다.

    이 함수는 다음 단계로 재고 시계열을 구축합니다:
    1. 과거 스냅샷 데이터로부터 센터별 재고 기준선 생성
    2. 예정 출고 (from_center) 이벤트 반영 (재고 감소)
    3. 예정 입고 (to_center) 이벤트 반영 (재고 증가)
    4. WIP 완료 이벤트 반영 (제조 완료 시 재고 증가)
    5. 누적 변화량을 시계열에 적용하여 미래 재고 예측

    Args:
        snapshot: 과거 재고 스냅샷 DataFrame
            - 필수 컬럼: center, resource_code, date, stock_qty
        moves: 재고 이동 계획 DataFrame
            - 필수 컬럼: resource_code, from_center, to_center, qty_ea
            - 필수 컬럼: ship_start_date/onboard_date (중 하나)
            - 선택 컬럼: pred_inbound_date, event_date, carrier_mode
        centers: 대상 센터 리스트
        skus: 대상 SKU (resource_code) 리스트
        index: 타임라인 인덱스 (SeriesIndex 객체)
            - index.range: 날짜 범위 (pd.DatetimeIndex)

    Returns:
        센터별 재고 타임라인 DataFrame
        - 컬럼: date, center, resource_code, stock_qty
        - stock_qty: 0 이상 실수 (음수 제거됨)
        - 빈 결과 시: 동일 스키마의 빈 DataFrame

    Notes:
        - 스냅샷의 마지막 날짜 이후의 moves만 반영됨
        - 출고는 ship_start_date에 차감, 입고는 pred_inbound_date에 추가
        - WIP 모드는 event_date에 재고 추가
        - 재고량은 forward fill 후 0 이하 값 제거

    Examples:
        >>> idx = SeriesIndex(
        ...     start=pd.Timestamp("2024-01-01"),
        ...     end=pd.Timestamp("2024-01-31")
        ... )
        >>> timeline = build_center_series(
        ...     snapshot=snapshot_df,
        ...     moves=moves_df,
        ...     centers=["center1", "center2"],
        ...     skus=["SKU001"],
        ...     index=idx,
        ... )
    """
    centers_set = _normalise_selection(centers)
    skus_set = _normalise_selection(skus)
    if snapshot.empty or not centers_set or not skus_set:
        return _empty_timeline()

    idx = index.range
    mv = moves.copy()
    ship_start_col = _resolve_onboard_column(mv)

    lines = []
    for (ct, sku), grp in snapshot.groupby(["center", "resource_code"]):
        if ct not in centers_set or sku not in skus_set:
            continue
        grp = grp.dropna(subset=["date"]).sort_values("date")
        if grp.empty:
            continue
        grp = (
            grp.groupby("date", as_index=False)["stock_qty"].last().sort_values("date")
        )
        grp = grp.drop_duplicates(subset="date", keep="last")
        last_dt = grp["date"].max()

        ts = pd.DataFrame(index=idx)
        ts["center"] = ct
        ts["resource_code"] = sku
        stock_series = grp.set_index("date")["stock_qty"].astype(float)
        stock_series = stock_series[~stock_series.index.duplicated(keep="last")]
        stock_series = stock_series.reindex(idx)
        stock_series = stock_series.ffill().fillna(0.0)
        ts["stock_qty"] = stock_series

        mv_sku = mv[mv["resource_code"] == sku]
        if not mv_sku.empty:
            eff_minus = (
                mv_sku[
                    (mv_sku["from_center"] == ct)
                    & mv_sku[ship_start_col].notna()
                    & (mv_sku[ship_start_col] > last_dt)
                ]
                .groupby(ship_start_col, as_index=False)["qty_ea"]
                .sum()
                .rename(columns={ship_start_col: "date", "qty_ea": "delta"})
            )
            eff_minus["delta"] *= -1

            mv_center = mv_sku[
                (mv_sku["to_center"] == ct) & (mv_sku["carrier_mode"] != "WIP")
            ]
            if not mv_center.empty:
                eff_plus = (
                    mv_center[
                        mv_center["pred_inbound_date"].notna()
                        & (mv_center["pred_inbound_date"] > last_dt)
                    ]
                    .groupby("pred_inbound_date", as_index=False)["qty_ea"]
                    .sum()
                    .rename(columns={"pred_inbound_date": "date", "qty_ea": "delta"})
                )
            else:
                eff_plus = pd.DataFrame(columns=["date", "delta"])

            frames_to_concat = [df for df in [eff_minus, eff_plus] if not df.empty]
            if frames_to_concat:
                eff_all = pd.concat(frames_to_concat, ignore_index=True)
                delta_series = (
                    eff_all.groupby("date")["delta"].sum().reindex(idx, fill_value=0.0)
                )
                ts["stock_qty"] = ts["stock_qty"].add(
                    delta_series.cumsum(), fill_value=0.0
                )

        if "event_date" in mv_sku.columns:
            wip_mask = (
                (mv_sku["carrier_mode"] == "WIP")
                & (mv_sku["to_center"] == ct)
                & mv_sku["event_date"].notna()
            )
            wip_complete = mv_sku[wip_mask]
        else:
            wip_complete = pd.DataFrame(columns=mv_sku.columns)
        if not wip_complete.empty:
            wip_add = (
                wip_complete.groupby("event_date", as_index=False)["qty_ea"]
                .sum()
                .rename(columns={"event_date": "date", "qty_ea": "delta"})
            )
            delta_series = (
                wip_add.groupby("date")["delta"].sum().reindex(idx, fill_value=0.0)
            )
            ts["stock_qty"] = ts["stock_qty"].add(delta_series.cumsum(), fill_value=0.0)

        ts["stock_qty"] = ts["stock_qty"].fillna(0).astype(float)
        ts["stock_qty"] = ts["stock_qty"].replace([np.inf, -np.inf], 0.0)
        ts["stock_qty"] = ts["stock_qty"].clip(lower=0.0)
        lines.append(ts.reset_index().rename(columns={"index": "date"}))

    if not lines:
        return _empty_timeline()
    out = pd.concat(lines, ignore_index=True)
    return out[TIMELINE_COLUMNS]


def build_in_transit_series(
    moves: pd.DataFrame,
    *,
    centers: Iterable[str],
    skus: Iterable[str],
    index: SeriesIndex,
    today: pd.Timestamp,
    lag_days: int,
) -> pd.DataFrame:
    """In-Transit (운송 중) 재고 타임라인을 생성합니다.

    이 함수는 다음 로직으로 운송 중 재고를 추적합니다:
    1. 출고 시작일 (ship_start_date/onboard_date)에 In-Transit 재고 증가
    2. 입고 예정일 (in_transit_end_date)에 In-Transit 재고 감소
    3. 타임라인 시작 전에 출고되어 아직 입고되지 않은 건(carry) 반영
    4. 누적 변화량을 적용하여 일자별 In-Transit 재고 계산

    Args:
        moves: 재고 이동 계획 DataFrame
            - 필수 컬럼: resource_code, to_center, qty_ea, carrier_mode
            - 필수 컬럼: ship_start_date/onboard_date (중 하나)
            - 필수 컬럼: in_transit_end_date
        centers: 대상 센터 리스트 (to_center 필터링용)
        skus: 대상 SKU 리스트
        index: 타임라인 인덱스 (SeriesIndex 객체)
        today: 현재 날짜 (사용 안 함, 호환성 유지)
        lag_days: 지연 일수 (사용 안 함, 호환성 유지)

    Returns:
        In-Transit 재고 타임라인 DataFrame
        - 컬럼: date, center="In-Transit", resource_code, stock_qty
        - stock_qty: 0 이상 정수 (음수 제거됨)
        - 빈 결과 시: 동일 스키마의 빈 DataFrame

    Notes:
        - WIP 모드 (carrier_mode="WIP")는 제외됨
        - carry: 타임라인 시작 전 출고되어 아직 입고 안 된 물량
        - 각 SKU별로 독립적으로 계산됨

    Examples:
        >>> idx = SeriesIndex(
        ...     start=pd.Timestamp("2024-01-01"),
        ...     end=pd.Timestamp("2024-01-31")
        ... )
        >>> in_transit = build_in_transit_series(
        ...     moves=moves_df,
        ...     centers=["center1"],
        ...     skus=["SKU001"],
        ...     index=idx,
        ...     today=pd.Timestamp("2024-01-15"),
        ...     lag_days=0,
        ... )
    """
    if moves.empty:
        return _empty_timeline()

    centers_set = _normalise_selection(centers)
    skus_set = _normalise_selection(skus)
    if not centers_set or not skus_set:
        return _empty_timeline()

    start_dt = index.start
    horizon_end = index.end
    idx = index.range
    start_col = _resolve_onboard_column(moves)

    filtered = moves[
        (moves["carrier_mode"] != "WIP")
        & moves["resource_code"].isin(skus_set)
        & moves["to_center"].isin(centers_set)
    ].copy()
    if filtered.empty:
        return _empty_timeline()

    lines = []
    for sku, grp in filtered.groupby("resource_code"):
        starts = grp.dropna(subset=[start_col]).groupby(start_col)["qty_ea"].sum()
        ends = (
            grp.dropna(subset=["in_transit_end_date"])
            .groupby("in_transit_end_date")["qty_ea"]
            .sum()
            * -1
        )
        delta = (
            starts.rename_axis("date")
            .to_frame("delta")
            .add(ends.rename_axis("date").to_frame("delta"), fill_value=0)["delta"]
            .sort_index()
        )
        delta = delta.reindex(idx, fill_value=0.0)
        series = delta.cumsum().clip(lower=0)

        carry_mask = (
            grp[start_col].notna()
            & (grp[start_col] < idx[0])
            & (
                grp["in_transit_end_date"].fillna(horizon_end + pd.Timedelta(days=1))
                > idx[0]
            )
        )
        carry = int(grp.loc[carry_mask, "qty_ea"].sum())
        if carry:
            series = (series + carry).clip(lower=0)

        if series.any():
            lines.append(
                pd.DataFrame(
                    {
                        "date": series.index,
                        "center": "In-Transit",
                        "resource_code": sku,
                        "stock_qty": series.values,
                    }
                )
            )

    if not lines:
        return _empty_timeline()

    out = pd.concat(lines, ignore_index=True)
    out["stock_qty"] = out["stock_qty"].round().astype(int)
    return out[TIMELINE_COLUMNS]


def build_wip_series(
    moves: pd.DataFrame,
    *,
    skus: Iterable[str],
    index: SeriesIndex,
) -> pd.DataFrame:
    """WIP (Work In Progress, 제조 중) 재고 타임라인을 생성합니다.

    이 함수는 다음 로직으로 제조 중 재고를 추적합니다:
    1. 제조 시작일 (ship_start_date/onboard_date)에 WIP 재고 증가
    2. 제조 완료일 (event_date)에 WIP 재고 감소
    3. 누적 변화량을 적용하여 일자별 WIP 재고 계산
    4. 타임라인 시작 전에 시작된 WIP 건(carry) 반영

    Args:
        moves: 재고 이동 계획 DataFrame
            - 필수 컬럼: resource_code, qty_ea, carrier_mode
            - 필수 컬럼: ship_start_date/onboard_date (중 하나)
            - 선택 컬럼: event_date (WIP 완료일)
            - carrier_mode="WIP"인 행만 처리됨
        skus: 대상 SKU 리스트
        index: 타임라인 인덱스 (SeriesIndex 객체)

    Returns:
        WIP 재고 타임라인 DataFrame
        - 컬럼: date, center="WIP", resource_code, stock_qty
        - stock_qty: 0 이상 정수 (음수 제거됨)
        - 빈 결과 시: 동일 스키마의 빈 DataFrame

    Notes:
        - carrier_mode="WIP"인 이동 건만 처리됨
        - event_date 누락 시 완료되지 않은 것으로 간주 (재고 유지)
        - carry: 타임라인 시작 전 제조 시작되어 아직 완료 안 된 물량
        - 각 SKU별로 독립적으로 계산됨

    Examples:
        >>> idx = SeriesIndex(
        ...     start=pd.Timestamp("2024-01-01"),
        ...     end=pd.Timestamp("2024-01-31")
        ... )
        >>> wip_timeline = build_wip_series(
        ...     moves=moves_df,
        ...     skus=["SKU001", "SKU002"],
        ...     index=idx,
        ... )
    """
    if moves.empty:
        return _empty_timeline()

    skus_set = _normalise_selection(skus)
    if not skus_set:
        return _empty_timeline()

    idx = index.range
    start_col = _resolve_onboard_column(moves)

    wip = moves[
        (moves["carrier_mode"] == "WIP") & moves["resource_code"].isin(skus_set)
    ]
    if wip.empty:
        return _empty_timeline()

    lines = []
    for sku, grp in wip.groupby("resource_code"):
        deltas = []
        onboard = (
            grp[grp[start_col].notna()]
            .groupby(start_col, as_index=False)["qty_ea"]
            .sum()
            .rename(columns={start_col: "date", "qty_ea": "delta"})
        )
        if not onboard.empty:
            deltas.append(onboard)

        if "event_date" in grp.columns:
            events = (
                grp[grp["event_date"].notna()]
                .groupby("event_date", as_index=False)["qty_ea"]
                .sum()
                .rename(columns={"event_date": "date", "qty_ea": "delta"})
            )
            if not events.empty:
                events["delta"] *= -1
                deltas.append(events)

        if not deltas:
            continue

        delta = pd.concat(deltas, ignore_index=True)
        delta_series = delta.groupby("date")["delta"].sum().reindex(idx, fill_value=0.0)
        series = delta_series.cumsum().clip(lower=0)

        onboard_dates = pd.to_datetime(grp[start_col], errors="coerce")
        event_dates = (
            pd.to_datetime(grp.get("event_date"), errors="coerce")
            if "event_date" in grp
            else pd.Series(pd.NaT, index=grp.index, dtype="datetime64[ns]")
        )
        carry_mask = (
            onboard_dates.notna()
            & (onboard_dates < idx[0])
            & event_dates.fillna(pd.Timestamp.max).ge(idx[0])
        )
        if carry_mask.any():
            carry = int(
                pd.to_numeric(grp.loc[carry_mask, "qty_ea"], errors="coerce")
                .fillna(0)
                .sum()
            )
            if carry:
                series = (series + carry).clip(lower=0)
        if not series.any():
            continue

        lines.append(
            pd.DataFrame(
                {
                    "date": series.index,
                    "center": "WIP",
                    "resource_code": sku,
                    "stock_qty": series.values,
                }
            )
        )

    if not lines:
        return _empty_timeline()

    out = pd.concat(lines, ignore_index=True)
    out["stock_qty"] = out["stock_qty"].round().astype(int)
    return out[TIMELINE_COLUMNS]
