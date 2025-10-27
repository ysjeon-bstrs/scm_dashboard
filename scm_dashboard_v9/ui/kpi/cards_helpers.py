"""KPI 카드 렌더링 헬퍼 함수 모듈.

render_sku_summary_cards 함수를 작은 함수들로 분해한 헬퍼들입니다.
"""

from __future__ import annotations

from typing import Dict, Mapping, Sequence, Tuple, NamedTuple

import pandas as pd

# WIP 파이프라인 계산 상수
WIP_WINDOW_DAYS = 30  # 30일 내 완료 예정 WIP 계산 기간


class AggregatedMetrics(NamedTuple):
    """집계된 메트릭들을 담는 컨테이너."""

    current_by_center: pd.Series
    current_totals: pd.Series
    global_current_totals: pd.Series
    daily_demand_series: pd.Series
    in_transit_series: pd.Series
    global_in_transit_totals: pd.Series


def validate_and_prepare_snapshot(
    snapshot: pd.DataFrame,
    centers: Sequence[str],
    skus: Sequence[str],
    date_column: str,
    latest_snapshot: pd.Timestamp | None,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    list[str],
    list[str],
    list[str],
    pd.Timestamp,
    pd.Timestamp,
    Mapping[str, str],
]:
    """스냅샷 데이터를 검증하고 준비합니다.

    Returns:
        tuple: (snapshot_view, filtered_snapshot, centers_list, sku_list, centers_all,
                latest_snapshot_dt, global_latest_snapshot_dt, name_map)
        에러 시 첫 번째 요소가 empty DataFrame
    """
    snapshot_view = snapshot.copy()

    # Date column 검증
    if (
        date_column not in snapshot_view.columns
        and "snapshot_date" in snapshot_view.columns
    ):
        date_column = "snapshot_date"
    if date_column not in snapshot_view.columns:
        return pd.DataFrame(), pd.DataFrame(), [], [], [], pd.NaT, pd.NaT, {}

    # Date 정규화
    snapshot_view["date"] = pd.to_datetime(
        snapshot_view[date_column], errors="coerce"
    ).dt.normalize()
    snapshot_view = snapshot_view.dropna(subset=["date"])
    if snapshot_view.empty:
        return pd.DataFrame(), pd.DataFrame(), [], [], [], pd.NaT, pd.NaT, {}

    snapshot_view["center"] = snapshot_view["center"].astype(str)
    snapshot_view["resource_code"] = snapshot_view["resource_code"].astype(str)

    # 센터 및 SKU 리스트 준비
    centers_list = [str(center).strip() for center in centers if str(center).strip()]
    sku_list = [str(sku).strip() for sku in skus if str(sku).strip()]
    centers_all = sorted(
        {
            str(center).strip()
            for center in snapshot_view["center"].unique()
            if str(center).strip()
        }
    )

    if not centers_list or not sku_list:
        return pd.DataFrame(), pd.DataFrame(), [], [], [], pd.NaT, pd.NaT, {}

    # 필터링
    filtered_snapshot = snapshot_view[
        snapshot_view["center"].isin(centers_list)
        & snapshot_view["resource_code"].isin(sku_list)
    ].copy()

    if filtered_snapshot.empty:
        return pd.DataFrame(), pd.DataFrame(), [], [], [], pd.NaT, pd.NaT, {}

    # Latest snapshot 계산
    global_latest_snapshot = snapshot_view["date"].max()
    if pd.isna(global_latest_snapshot):
        return pd.DataFrame(), pd.DataFrame(), [], [], [], pd.NaT, pd.NaT, {}
    global_latest_snapshot_dt = pd.to_datetime(global_latest_snapshot).normalize()

    selected_latest_snapshot = filtered_snapshot["date"].max()
    if pd.isna(selected_latest_snapshot):
        return pd.DataFrame(), pd.DataFrame(), [], [], [], pd.NaT, pd.NaT, {}

    # latest_snapshot_dt 설정 (참조용)
    # 주의: 실제 재고 계산(kpi_breakdown_per_sku, aggregate_metrics)은 센터별 최신 데이터를
    # 개별적으로 가져오므로, 이 값은 주로 표시/로깅 목적으로 사용됩니다.
    if latest_snapshot is None or pd.isna(latest_snapshot):
        latest_snapshot_dt = pd.to_datetime(selected_latest_snapshot).normalize()
    else:
        requested_dt = pd.to_datetime(latest_snapshot).normalize()
        # 모든 선택된 센터가 요청한 날짜에 데이터를 가지고 있는지 확인
        centers_with_data_at_requested = set(
            filtered_snapshot[filtered_snapshot["date"] == requested_dt][
                "center"
            ].unique()
        )
        all_centers_have_data = set(centers_list).issubset(
            centers_with_data_at_requested
        )

        if all_centers_have_data:
            latest_snapshot_dt = requested_dt
        else:
            # 일부 센터에 요청 날짜 데이터가 없으면 센터별 최신 날짜의 최소값 사용
            # (모든 센터가 데이터를 가진 가장 최근 날짜)
            center_latest_dates = filtered_snapshot.groupby("center")["date"].max()
            if not center_latest_dates.empty:
                latest_snapshot_dt = pd.to_datetime(
                    center_latest_dates.min()
                ).normalize()
            else:
                latest_snapshot_dt = pd.to_datetime(
                    selected_latest_snapshot
                ).normalize()

    # Name map 생성
    name_map: Mapping[str, str] = {}
    if "resource_name" in filtered_snapshot.columns:
        name_rows = filtered_snapshot.dropna(
            subset=["resource_code", "resource_name"]
        ).copy()
        if not name_rows.empty:
            name_rows["resource_code"] = name_rows["resource_code"].astype(str)
            name_rows["resource_name"] = (
                name_rows["resource_name"].astype(str).str.strip()
            )
            name_rows = name_rows[name_rows["resource_name"] != ""]
            if not name_rows.empty:
                name_map = dict(
                    name_rows.sort_values("date", ascending=False)[
                        ["resource_code", "resource_name"]
                    ]
                    .drop_duplicates(subset=["resource_code"])
                    .itertuples(index=False, name=None)
                )

    return (
        snapshot_view,
        filtered_snapshot,
        centers_list,
        sku_list,
        centers_all,
        latest_snapshot_dt,
        global_latest_snapshot_dt,
        name_map,
    )


def prepare_moves_data(
    moves: pd.DataFrame,
    centers_list: list[str],
    sku_list: list[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Moves 데이터를 준비합니다.

    Returns:
        tuple: (moves_view, moves_global)
    """
    moves_view = moves.copy() if moves is not None else pd.DataFrame()
    moves_global = pd.DataFrame()

    if not moves_view.empty:
        if "carrier_mode" in moves_view.columns:
            moves_view["carrier_mode"] = (
                moves_view["carrier_mode"].astype(str).str.upper()
            )
        for column in ["resource_code", "to_center"]:
            if column in moves_view.columns:
                moves_view[column] = moves_view[column].astype(str)
        for column in ["inbound_date", "arrival_date", "onboard_date", "event_date"]:
            if column in moves_view.columns:
                moves_view[column] = pd.to_datetime(moves_view[column], errors="coerce")

        if "resource_code" in moves_view.columns:
            moves_view = moves_view[
                moves_view["resource_code"].isin(sku_list)
                | (moves_view["resource_code"] == "")
            ]
        moves_global = moves_view.copy()
        if "to_center" in moves_view.columns:
            moves_view = moves_view[
                moves_view["to_center"].isin(centers_list)
                | (moves_view["to_center"] == "")
            ]

    return moves_view, moves_global


def calculate_wip_pipeline(
    moves: pd.DataFrame,
    sku_list: list[str],
    centers_list: list[str],
    today_dt: pd.Timestamp,
) -> Tuple[Dict[str, int], Dict[tuple[str, str], int]]:
    """WIP 파이프라인 및 30일 내 완료 예정을 계산합니다.

    Returns:
        tuple: (wip_pipeline_totals, wip_30d_by_center)
    """
    wip_pipeline_totals: Dict[str, int] = {}
    wip_30d_by_center: Dict[tuple[str, str], int] = {}

    window_end = today_dt + pd.Timedelta(days=WIP_WINDOW_DAYS)

    if moves is None or moves.empty or not sku_list:
        return wip_pipeline_totals, wip_30d_by_center

    wf = moves.copy()
    for column in ["onboard_date", "event_date"]:
        if column in wf.columns:
            wf[column] = pd.to_datetime(wf[column], errors="coerce")
        else:
            wf[column] = pd.NaT
    if "carrier_mode" in wf.columns:
        wf["carrier_mode"] = wf["carrier_mode"].astype(str).str.upper()
    else:
        wf["carrier_mode"] = ""
    if "to_center" in wf.columns:
        wf["to_center"] = wf["to_center"].astype(str).str.strip()
    else:
        wf["to_center"] = ""
    if "resource_code" in wf.columns:
        wf["resource_code"] = wf["resource_code"].astype(str).str.strip()
    else:
        wf["resource_code"] = ""
    if "qty_ea" in wf.columns:
        wf["qty_ea"] = (
            pd.to_numeric(wf["qty_ea"], errors="coerce").fillna(0).astype(int)
        )
    else:
        wf["qty_ea"] = 0
    if "status" in wf.columns:
        wf = wf[wf["status"].astype(str).str.upper() != "CANCEL"]

    wf = wf[wf["resource_code"].isin(sku_list)]

    if not wf.empty:
        pipeline_mask = (wf["carrier_mode"] == "WIP") & (wf["event_date"] > today_dt)
        wip_pipeline_series = (
            wf.loc[pipeline_mask].groupby("resource_code")["qty_ea"].sum()
        )

        center_mask = (
            (wf["carrier_mode"] == "WIP")
            & (wf["to_center"].isin(centers_list))
            & (wf["event_date"] >= today_dt)
            & (wf["event_date"] <= window_end)
        )
        wip_30d_series = (
            wf.loc[center_mask]
            .groupby(["resource_code", "to_center"], as_index=True)["qty_ea"]
            .sum()
        )
        pipeline_dict = (
            wip_pipeline_series.astype(int).to_dict()
            if not wip_pipeline_series.empty
            else {}
        )
        wip_30d_dict = (
            wip_30d_series.astype(int).to_dict() if not wip_30d_series.empty else {}
        )

        if "__TOTAL__" in sku_list:
            total_pipeline_sum = int(
                sum(value for key, value in pipeline_dict.items() if key != "__TOTAL__")
            )
            pipeline_dict["__TOTAL__"] = total_pipeline_sum
            total_center_series = (
                wip_30d_series.groupby(level=1).sum()
                if not wip_30d_series.empty
                else pd.Series(dtype=int)
            )
            for center_name in centers_list:
                center_total_value = int(total_center_series.get(center_name, 0))
                wip_30d_dict[("__TOTAL__", center_name)] = center_total_value

        wip_pipeline_totals = pipeline_dict
        wip_30d_by_center = wip_30d_dict

    return wip_pipeline_totals, wip_30d_by_center


def aggregate_metrics(
    filtered_snapshot: pd.DataFrame,
    snapshot_view: pd.DataFrame,
    latest_snapshot_dt: pd.Timestamp,
    global_latest_snapshot_dt: pd.Timestamp,
    moves_view: pd.DataFrame,
    moves_global: pd.DataFrame,
    centers_list: list[str],
    centers_all: list[str],
    sku_list: list[str],
    today_dt: pd.Timestamp,
    lag_days: int,
) -> AggregatedMetrics:
    """재고, 이동중, 수요 등의 메트릭을 집계합니다.

    Returns:
        AggregatedMetrics: 집계된 메트릭들
    """
    from .metrics import extract_daily_demand, movement_breakdown_per_center

    # Latest snapshot rows 필터링 - 센터별 최신 날짜 사용
    if not filtered_snapshot.empty:
        center_latest_dates = filtered_snapshot.groupby("center")["date"].max()
        latest_snapshot_parts = []
        for center, latest_date in center_latest_dates.items():
            if center not in centers_list:
                continue
            center_latest_data = filtered_snapshot[
                (filtered_snapshot["center"] == center)
                & (filtered_snapshot["date"] == latest_date)
            ]
            latest_snapshot_parts.append(center_latest_data)
        latest_snapshot_rows = (
            pd.concat(latest_snapshot_parts, ignore_index=True)
            if latest_snapshot_parts
            else pd.DataFrame()
        )
    else:
        latest_snapshot_rows = pd.DataFrame()

    if "stock_qty" in latest_snapshot_rows.columns:
        latest_snapshot_rows["stock_qty"] = pd.to_numeric(
            latest_snapshot_rows["stock_qty"], errors="coerce"
        )

    # Global snapshot rows 필터링 - 센터별 스냅샷 생성 시간이 다를 수 있으므로
    # 각 센터의 최신 날짜 데이터를 사용 (고정된 global_latest_snapshot_dt 사용 안 함)
    if not snapshot_view.empty:
        center_latest_dates = snapshot_view.groupby("center")["date"].max()
        global_snapshot_parts = []
        for center, latest_date in center_latest_dates.items():
            center_latest_data = snapshot_view[
                (snapshot_view["center"] == center)
                & (snapshot_view["date"] == latest_date)
            ]
            global_snapshot_parts.append(center_latest_data)
        global_snapshot_rows = (
            pd.concat(global_snapshot_parts, ignore_index=True)
            if global_snapshot_parts
            else pd.DataFrame()
        )
    else:
        global_snapshot_rows = pd.DataFrame()

    if "stock_qty" in global_snapshot_rows.columns:
        global_snapshot_rows["stock_qty"] = pd.to_numeric(
            global_snapshot_rows["stock_qty"], errors="coerce"
        )

    # Current stock 집계
    current_by_center = (
        latest_snapshot_rows.groupby(["resource_code", "center"])["stock_qty"].sum()
        if "stock_qty" in latest_snapshot_rows.columns
        else pd.Series(dtype=float)
    )
    current_totals = (
        current_by_center.groupby(level=0).sum()
        if not current_by_center.empty
        else pd.Series(dtype=float)
    )

    # Global current stock 집계
    global_current_totals = (
        global_snapshot_rows.groupby("resource_code")["stock_qty"].sum()
        if "stock_qty" in global_snapshot_rows.columns
        and not global_snapshot_rows.empty
        else pd.Series(dtype=float)
    )

    # Daily demand 추출
    daily_demand_series, _ = extract_daily_demand(latest_snapshot_rows)

    # In-transit 계산
    in_transit_series, _ = movement_breakdown_per_center(
        moves_view,
        centers_list,
        sku_list,
        today_dt,
        int(lag_days),
    )

    # Global in-transit 계산
    global_in_transit_series = pd.Series(dtype=float)
    if centers_all:
        global_in_transit_series, _ = movement_breakdown_per_center(
            moves_global,
            centers_all,
            sku_list,
            today_dt,
            int(lag_days),
        )

    global_in_transit_totals = (
        global_in_transit_series.groupby(level=0).sum()
        if not global_in_transit_series.empty
        else pd.Series(dtype=float)
    )

    return AggregatedMetrics(
        current_by_center=current_by_center,
        current_totals=current_totals,
        global_current_totals=global_current_totals,
        daily_demand_series=daily_demand_series,
        in_transit_series=in_transit_series,
        global_in_transit_totals=global_in_transit_totals,
    )
