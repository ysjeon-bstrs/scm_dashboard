"""
테이블 렌더링 모듈

이 모듈은 SCM 대시보드의 테이블 렌더링 로직을 담당합니다.
입고 예정, WIP, 재고 현황, 로트 상세 등의 테이블을 렌더링합니다.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
import streamlit as st

from center_alias import normalize_center_value
from scm_dashboard_v9.analytics.inventory import pivot_inventory_cost_from_raw
from scm_dashboard_v9.core.config import CENTER_COL, CONFIG
from scm_dashboard_v9.data_sources.loaders import load_snapshot_raw
from scm_dashboard_v9.domain.filters import (
    filter_by_centers,
    is_empty_or_none,
    safe_to_datetime,
)
from scm_dashboard_v9.planning.schedule import calculate_predicted_inbound_date


def build_resource_name_map(snapshot: pd.DataFrame) -> dict[str, str]:
    """
    스냅샷 데이터에서 SKU → 품명 매핑 딕셔너리를 생성합니다.

    Args:
        snapshot: 스냅샷 데이터프레임 (resource_code, resource_name 컬럼 포함)

    Returns:
        {"BA00021": "제품A", "BA00022": "제품B", ...} 형식의 매핑 딕셔너리.
        resource_name 컬럼이 없으면 빈 딕셔너리 반환.

    Examples:
        >>> resource_name_map = build_resource_name_map(snapshot_df)
        >>> resource_name_map["BA00021"]
        '제품A'
    """
    # ========================================
    # 1단계: resource_name 컬럼 확인
    # ========================================
    if "resource_name" not in snapshot.columns:
        return {}

    # ========================================
    # 2단계: 유효한 품명 데이터만 추출
    # ========================================
    name_rows = snapshot.loc[
        snapshot["resource_name"].notna(),
        [
            "resource_code",
            "resource_name",
        ],
    ].copy()

    # ========================================
    # 3단계: 타입 정규화 및 공백 제거
    # ========================================
    name_rows["resource_code"] = name_rows["resource_code"].astype(str)
    name_rows["resource_name"] = name_rows["resource_name"].astype(str).str.strip()

    # 빈 문자열 제거
    name_rows = name_rows[name_rows["resource_name"] != ""]

    if name_rows.empty:
        return {}

    # ========================================
    # 4단계: 중복 제거 및 딕셔너리 변환
    # ========================================
    # SKU가 중복되면 첫 번째 값 사용
    resource_name_map = (
        name_rows.drop_duplicates("resource_code")
        .set_index("resource_code")["resource_name"]
        .to_dict()
    )

    return resource_name_map


def render_inbound_and_wip_tables(
    moves: pd.DataFrame,
    snapshot: pd.DataFrame,
    selected_centers: list[str],
    selected_skus: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    lag_days: int,
    today: pd.Timestamp,
) -> None:
    """
    입고 예정 테이블과 WIP 테이블을 렌더링합니다.

    렌더링되는 테이블:
    1. ✅ 입고 예정 현황 (Confirmed / In-transit Inbound): 운송 중인 재고
    2. 🛠 생산 진행 현황 (Manufacturing WIP Status)

    Args:
        moves: 이동 원장 데이터프레임
        snapshot: 스냅샷 데이터프레임 (품명 매핑용)
        selected_centers: 선택된 센터 목록
        selected_skus: 선택된 SKU 목록
        start: 시작 날짜
        end: 종료 날짜
        lag_days: 입고 반영 리드타임 (일)
        today: 오늘 날짜

    Notes:
        - confirmed_inbound: arrival_date 또는 onboard_date 기준
        - WIP: event_date 기준, 태광KR만 표시
    """
    window_start = start
    window_end = end
    today = pd.to_datetime(today).normalize()

    # ========================================
    # 1단계: 이동 원장 컬럼 보완
    # ========================================
    # 필요한 컬럼이 없으면 빈 값으로 채움
    moves_view = moves.copy()
    for col in [
        "carrier_mode",
        "to_center",
        "resource_code",
        "inbound_date",
        "arrival_date",
        "eta_date",
        "onboard_date",
        "event_date",
        "lot",
    ]:
        if col not in moves_view.columns:
            if "date" in col:
                moves_view[col] = pd.Series(
                    pd.NaT, index=moves_view.index, dtype="datetime64[ns]"
                )
            else:
                moves_view[col] = pd.Series("", index=moves_view.index, dtype="object")

    # ========================================
    # 2단계: 예상 입고일 계산 (pred_inbound_date)
    # ========================================
    # 공통 함수 사용 (중복 코드 제거)
    moves_view = calculate_predicted_inbound_date(
        moves_view, today=today, lag_days=lag_days
    )

    # ========================================
    # 3단계: 입고 예정 현황 필터링 (운송 중)
    # ========================================
    # selected_centers를 정규화 (normalize_moves에서 to_center가 정규화되므로)
    normalized_selected_centers = {
        norm
        for center in selected_centers
        for norm in [normalize_center_value(center)]
        if norm
    }

    # inbound_date가 없는 운송 중 재고만 추출
    arr_transport = moves_view[
        (moves_view["carrier_mode"] != "WIP")
        & (moves_view["to_center"].isin(normalized_selected_centers))
        & (moves_view["resource_code"].isin(selected_skus))
        & (moves_view["inbound_date"].isna())
    ].copy()

    # 표시 날짜: arrival_date → eta_date → onboard_date
    effective_display = arr_transport["arrival_date"].copy()
    if "eta_date" in arr_transport.columns:
        effective_display = effective_display.fillna(arr_transport["eta_date"])
    arr_transport["display_date"] = effective_display.fillna(
        arr_transport["onboard_date"]
    )
    arr_transport = arr_transport[arr_transport["display_date"].notna()]
    arr_transport = arr_transport[
        (arr_transport["display_date"] >= window_start)
        & (arr_transport["display_date"] <= window_end)
    ]

    # ========================================
    # 4단계: WIP 필터링 (생산 중)
    # ========================================
    # 태광KR만 WIP 표시 (센터명도 정규화 체크)
    arr_wip = pd.DataFrame()
    show_wip = any(
        normalize_center_value(center) == "태광KR" for center in selected_centers
    )
    if show_wip:
        arr_wip = moves_view[
            (moves_view["carrier_mode"] == "WIP")
            & (moves_view["to_center"] == "태광KR")
            & (moves_view["resource_code"].isin(selected_skus))
            & (moves_view["event_date"].notna())
            & (moves_view["event_date"] >= window_start)
            & (moves_view["event_date"] <= window_end)
        ].copy()
        arr_wip["display_date"] = arr_wip["event_date"]

    # ========================================
    # 5단계: 품명 매핑 추가
    # ========================================
    resource_name_map = build_resource_name_map(snapshot)

    confirmed_inbound = arr_transport.copy()
    if resource_name_map and not confirmed_inbound.empty:
        confirmed_inbound["resource_name"] = (
            confirmed_inbound["resource_code"].map(resource_name_map).fillna("")
        )

    # ========================================
    # 6단계: 입고 예정 현황 테이블 렌더링
    # ========================================
    st.markdown("#### ✅ 입고 예정 현황 (Confirmed / In-transit Inbound)")

    if confirmed_inbound.empty:
        st.caption(
            "선택한 조건에서 예정된 운송 입고가 없습니다. (오늘 이후 / 선택 기간)"
        )
    else:
        # 남은 일수 계산
        arrival_basis = confirmed_inbound.get("arrival_date")
        arrival_basis = safe_to_datetime(arrival_basis)
        if "eta_date" in confirmed_inbound.columns:
            eta_normalized = safe_to_datetime(confirmed_inbound.get("eta_date"))
            arrival_basis = arrival_basis.fillna(eta_normalized)

        days_arrival = (arrival_basis - today).dt.days.astype("Int64")
        days_to_arrival = days_arrival.astype(object)
        undefined_mask = arrival_basis.isna()
        if undefined_mask.any():
            days_to_arrival.loc[undefined_mask] = "not_defined"
        confirmed_inbound["days_to_arrival"] = days_to_arrival

        # pred_inbound_date 기반 days_to_inbound 계산 (미확정은 "not_defined")
        pred_inbound_normalized = confirmed_inbound["pred_inbound_date"].dt.normalize()
        days_inbound = (pred_inbound_normalized - today).dt.days.astype("Int64")
        days_to_inbound = days_inbound.astype(object)
        undefined_inbound_mask = pred_inbound_normalized.isna()
        if undefined_inbound_mask.any():
            days_to_inbound.loc[undefined_inbound_mask] = "not_defined"
        confirmed_inbound["days_to_inbound"] = days_to_inbound

        # pred_inbound_date 표시용 포맷 (NaT → "not_defined")
        pred_display = confirmed_inbound["pred_inbound_date"].apply(
            lambda x: (
                "not_defined"
                if pd.isna(x)
                else x.strftime("%Y-%m-%d") if hasattr(x, "strftime") else str(x)
            )
        )
        confirmed_inbound["pred_inbound_date"] = pred_display

        # 정렬
        confirmed_inbound = confirmed_inbound.sort_values(
            ["display_date", "to_center", "resource_code", "qty_ea"],
            ascending=[True, True, True, False],
        )

        # 표시할 컬럼 선택
        inbound_cols = [
            "display_date",
            "days_to_arrival",
            "to_center",
            "resource_code",
            "resource_name",
            "qty_ea",
            "carrier_mode",
            "onboard_date",
            "pred_inbound_date",
            "days_to_inbound",
            "lot",
        ]
        inbound_cols = [c for c in inbound_cols if c in confirmed_inbound.columns]

        st.dataframe(
            confirmed_inbound[inbound_cols].head(CONFIG.ui.max_table_rows),
            use_container_width=True,
            height=CONFIG.ui.table_height_inbound,
        )
        st.caption(
            "※ pred_inbound_date: 예상 입고일 (도착일 + 리드타임), days_to_inbound: 예상 입고까지 남은 일수"
        )

    # ========================================
    # 7단계: 생산 진행 현황 (WIP) 테이블 렌더링
    # ========================================
    st.markdown("#### 🛠 생산 진행 현황 (Manufacturing WIP Status)")

    if not arr_wip.empty:
        if resource_name_map:
            arr_wip["resource_name"] = (
                arr_wip["resource_code"].map(resource_name_map).fillna("")
            )

        # 정렬
        arr_wip = arr_wip.sort_values(
            ["display_date", "resource_code", "qty_ea"], ascending=[True, True, False]
        )

        # 완료까지 남은 일수
        arr_wip["days_to_completion"] = (
            arr_wip["display_date"].dt.normalize() - today
        ).dt.days

        # 표시할 컬럼 선택
        wip_cols = [
            "display_date",
            "days_to_completion",
            "resource_code",
            "resource_name",
            "qty_ea",
            "pred_inbound_date",
            "lot",
            "global_b2c",
            "global_b2b",
        ]
        wip_cols = [c for c in wip_cols if c in arr_wip.columns]

        st.dataframe(
            arr_wip[wip_cols].head(CONFIG.ui.max_table_rows),
            use_container_width=True,
            height=CONFIG.ui.table_height_wip,
        )
    else:
        st.caption("생산중(WIP) 데이터가 없습니다.")


def render_inventory_table(
    snapshot: pd.DataFrame,
    selected_centers: list[str],
    latest_dt: pd.Timestamp,
    resource_name_map: dict[str, str],
) -> pd.DataFrame:
    """
    선택 센터의 현재 재고 현황 테이블을 렌더링합니다.

    기능:
    - SKU 필터링 (검색)
    - 정렬 기준 선택
    - 총합=0 숨기기 옵션
    - 재고자산(제조원가) 표시 옵션
    - CSV 다운로드

    Args:
        snapshot: 스냅샷 데이터프레임
        selected_centers: 선택된 센터 목록
        latest_dt: 최신 스냅샷 날짜
        resource_name_map: SKU → 품명 매핑 딕셔너리

    Returns:
        표시된 데이터프레임 (로트 상세 렌더링에 사용)

    Notes:
        - 선택된 센터의 **전체 SKU** 표시 (필터와 무관)
        - 센터별 최신 스냅샷 사용
    """
    # ========================================
    # 1단계: 데이터 유효성 검증
    # ========================================
    if snapshot.empty or "date" not in snapshot.columns:
        st.info("스냅샷 데이터가 없습니다.")
        return pd.DataFrame()

    if pd.isna(latest_dt):
        st.info("스냅샷 데이터의 날짜 정보를 확인할 수 없습니다.")
        return pd.DataFrame()

    latest_dt_str = latest_dt.strftime("%Y-%m-%d")
    st.subheader(f"센터별 전체 재고 현황 (스냅샷 {latest_dt_str} / 전체 SKU)")

    # ========================================
    # 2단계: 센터별 최신 스냅샷 날짜 계산
    # ========================================
    center_latest_series = (
        snapshot[snapshot["center"].isin(selected_centers)]
        .groupby("center")["date"]
        .max()
    )
    center_latest_dates = {
        center: ts.normalize()
        for center, ts in center_latest_series.items()
        if pd.notna(ts)
    }

    if not center_latest_series.empty:
        caption_items = [
            f"{center}: {center_latest_dates[center].strftime('%Y-%m-%d')}"
            for center in selected_centers
            if center in center_latest_dates
        ]
        if caption_items:
            st.caption("센터별 최신 스냅샷: " + " / ".join(caption_items))
    else:
        center_latest_dates = {}

    # ========================================
    # 3단계: 최신 스냅샷 추출
    # ========================================
    sub = snapshot[
        (snapshot["date"] <= latest_dt) & (snapshot["center"].isin(selected_centers))
    ].copy()

    if not sub.empty:
        sub["center"] = sub["center"].astype(str).str.strip()
        # 각 센터/SKU의 최신 데이터만 추출
        sub = (
            sub.sort_values(["center", "resource_code", "date"])
            .groupby(["center", "resource_code"], as_index=False)
            .tail(1)
        )

    # ========================================
    # 4단계: Pivot 테이블 생성
    # ========================================
    pivot = (
        sub.groupby(["resource_code", "center"], as_index=False)["stock_qty"]
        .sum()
        .pivot(index="resource_code", columns="center", values="stock_qty")
        .reindex(columns=selected_centers)
        .fillna(0)
    )

    pivot = pivot.astype(int)
    pivot["총합"] = pivot.sum(axis=1)

    # ========================================
    # 5단계: 필터/정렬 UI
    # ========================================
    col_filter, col_sort = st.columns([2, 1])
    with col_filter:
        sku_query = st.text_input(
            "SKU 필터 — 품목번호 검색 시 해당 SKU의 센터별 제조번호(LOT) 확인",
            "",
            key="v9_sku_filter_text",
        )
    with col_sort:
        sort_candidates = ["총합"] + selected_centers
        sort_by = st.selectbox("정렬 기준", sort_candidates, index=0)

    col_zero, col_cost = st.columns(2)
    with col_zero:
        hide_zero = st.checkbox("총합=0 숨기기", value=True)
    with col_cost:
        show_cost = st.checkbox("재고자산(제조원가) 표시", value=False)

    # ========================================
    # 6단계: 필터/정렬 적용
    # ========================================
    view = pivot.copy()
    if sku_query.strip():
        view = view[
            view.index.astype(str).str.contains(
                sku_query.strip(), case=False, regex=False
            )
        ]
    if hide_zero and "총합" in view.columns:
        view = view[view["총합"] > 0]
    if sort_by in view.columns:
        view = view.sort_values(by=sort_by, ascending=False)

    display_df = view.reset_index().rename(columns={"resource_code": "SKU"})
    if resource_name_map:
        display_df.insert(
            1, "품명", display_df["SKU"].map(resource_name_map).fillna("")
        )

    # ========================================
    # 7단계: 재고자산 표시 (선택적)
    # ========================================
    if show_cost:
        snap_raw_df = load_snapshot_raw()
        cost_pivot = pivot_inventory_cost_from_raw(
            snap_raw_df, latest_dt, selected_centers, center_latest_dates
        )
        if cost_pivot.empty:
            st.warning(
                "재고자산 계산을 위한 'snapshot_raw' 데이터를 불러올 수 없어 수량만 표시합니다. (엑셀에 'snapshot_raw' 시트가 있으면 자동 사용됩니다)"
            )
            merged_df = display_df
            cost_columns = []
        else:
            merged_df = display_df.merge(
                cost_pivot.rename(columns={"resource_code": "SKU"}),
                on="SKU",
                how="left",
            )
            cost_columns = [c for c in merged_df.columns if c.endswith("_재고자산")]
            if "총 재고자산" in merged_df.columns:
                cost_columns.append("총 재고자산")
            if cost_columns:
                merged_df[cost_columns] = merged_df[cost_columns].fillna(0).astype(int)
                for col in cost_columns:
                    merged_df[col] = merged_df[col].apply(
                        lambda x: f"{x:,}원" if isinstance(x, (int, float)) else x
                    )

        quantity_columns = [
            c
            for c in merged_df.columns
            if c not in {"SKU", "품명", "총합", *cost_columns}
        ]
        ordered_columns = ["SKU"]
        if "품명" in merged_df.columns:
            ordered_columns.append("품명")
        ordered_columns.extend(
            [c for c in quantity_columns if not c.endswith("_재고자산")]
        )
        if "총합" in merged_df.columns:
            ordered_columns.append("총합")
        ordered_columns.extend(cost_columns)
        show_df = merged_df[ordered_columns]
    else:
        show_df = display_df
        cost_columns = []

    # ========================================
    # 8단계: 수량 포맷팅 (천단위 콤마)
    # ========================================
    qty_columns = [
        c
        for c in show_df.columns
        if c not in {"SKU", "품명"}
        and not c.endswith("_재고자산")
        and c != "총 재고자산"
    ]
    for column in qty_columns:
        show_df[column] = show_df[column].apply(
            lambda x: (
                f"{int(x):,}" if pd.notna(x) and isinstance(x, (int, float)) else x
            )
        )

    # ========================================
    # 9단계: 테이블 렌더링 + CSV 다운로드
    # ========================================
    st.dataframe(
        show_df, use_container_width=True, height=CONFIG.ui.table_height_inventory
    )

    csv_bytes = show_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "현재 표 CSV 다운로드",
        data=csv_bytes,
        file_name=f"centers_{'-'.join(selected_centers)}_snapshot_{latest_dt_str}.csv",
        mime="text/csv",
    )

    st.caption(
        "※ 이 표는 **선택된 센터 전체 SKU**의 최신 스냅샷 재고입니다. 상단 'SKU 선택'과 무관하게 모든 SKU가 포함됩니다."
    )

    return show_df


def render_lot_details(
    visible_skus: list[str],
    selected_centers: list[str],
    center_latest_dates: dict[str, pd.Timestamp],
    latest_dt: pd.Timestamp,
) -> None:
    """
    단일 SKU 선택 시 로트(제조번호) 상세 정보를 렌더링합니다.

    조건:
    - visible_skus가 정확히 1개일 때만 표시
    - snapshot_raw 데이터 필요

    Args:
        visible_skus: 현재 표시 중인 SKU 목록 (길이 1이어야 함)
        selected_centers: 선택된 센터 목록
        center_latest_dates: 센터별 최신 스냅샷 날짜 매핑
        latest_dt: 전체 최신 스냅샷 날짜

    Notes:
        - snapshot_raw에서 센터별 컬럼을 읽어 로트별 집계
        - CENTER_COL 매핑 사용 (scm_dashboard_v9.core.config)
    """
    # ========================================
    # 1단계: 단일 SKU 검증
    # ========================================
    if len(visible_skus) != 1:
        return  # 여러 SKU 또는 SKU 없음 → 로트 상세 표시 안 함

    lot_sku = visible_skus[0]
    latest_dt_str = latest_dt.strftime("%Y-%m-%d")

    # ========================================
    # 2단계: snapshot_raw 로드
    # ========================================
    snap_raw_df = load_snapshot_raw()
    if snap_raw_df is None or snap_raw_df.empty:
        st.markdown(
            f"### 로트 상세 (스냅샷 {latest_dt_str} / **{', '.join(selected_centers)}** / **{lot_sku}**)"
        )
        st.caption("해당 조건의 로트 상세가 없습니다. (snapshot_raw 없음)")
        return

    # ========================================
    # 3단계: 컬럼 매핑 (대소문자 무관)
    # ========================================
    raw_df = snap_raw_df.copy()
    cols_map = {str(col).strip().lower(): col for col in raw_df.columns}

    col_date = cols_map.get("snapshot_date") or cols_map.get("date")
    col_sku = (
        cols_map.get("resource_code") or cols_map.get("sku") or cols_map.get("상품코드")
    )
    col_lot = cols_map.get("lot")

    # 사용 가능한 센터 필터링 (CENTER_COL에 매핑된 센터만)
    used_centers = [
        ct for ct in selected_centers if CENTER_COL.get(ct) in raw_df.columns
    ]

    if not all([col_date, col_sku, col_lot]) or not used_centers:
        st.markdown(
            f"### 로트 상세 (스냅샷 {latest_dt_str} / **{', '.join(selected_centers)}** / **{lot_sku}**)"
        )
        st.caption("해당 조건의 로트 상세가 없습니다.")
        return

    # ========================================
    # 4단계: 날짜 정규화
    # ========================================
    raw_df[col_date] = pd.to_datetime(raw_df[col_date], errors="coerce").dt.normalize()

    # ========================================
    # 5단계: 센터별 스냅샷 날짜 매핑
    # ========================================
    lot_snapshot_dates = {
        center: center_latest_dates.get(center)
        for center in used_centers
        if center in center_latest_dates
    }

    lot_title_dates = (
        " / ".join(
            f"{ct}: {dt.strftime('%Y-%m-%d')}"
            for ct, dt in lot_snapshot_dates.items()
            if pd.notna(dt)
        )
        or latest_dt_str
    )

    st.markdown(
        f"### 로트 상세 (스냅샷 {lot_title_dates} / **{', '.join(selected_centers)}** / **{lot_sku}**)"
    )

    # ========================================
    # 6단계: 센터별 로트 집계
    # ========================================
    lot_tables = []
    for center in used_centers:
        src_col = CENTER_COL.get(center)
        if not src_col or src_col not in raw_df.columns:
            continue

        target_date = lot_snapshot_dates.get(center)
        if lot_snapshot_dates and pd.isna(target_date):
            continue

        # 해당 센터의 특정 날짜 데이터 필터링
        if lot_snapshot_dates:
            center_subset = raw_df[
                (raw_df[col_date] == target_date)
                & (raw_df[col_sku].astype(str) == str(lot_sku))
            ].copy()
        else:
            center_subset = raw_df[
                (raw_df[col_date] == latest_dt)
                & (raw_df[col_sku].astype(str) == str(lot_sku))
            ].copy()

        if center_subset.empty:
            continue

        # 수량 컬럼 정규화
        center_subset[src_col] = (
            pd.to_numeric(center_subset[src_col], errors="coerce")
            .fillna(0)
            .clip(lower=0)
        )

        # 로트별 집계
        center_table = (
            center_subset[[col_lot, src_col]]
            .groupby(col_lot, as_index=False)[src_col]
            .sum()
        )
        center_table = center_table.rename(columns={col_lot: "lot", src_col: center})
        lot_tables.append(center_table)

    # ========================================
    # 7단계: 센터별 테이블 병합
    # ========================================
    if not lot_tables:
        st.caption("해당 조건의 로트 상세가 없습니다.")
        return

    lot_table = lot_tables[0]
    for tbl in lot_tables[1:]:
        lot_table = lot_table.merge(tbl, on="lot", how="outer")

    # ========================================
    # 8단계: 로트명 정규화 및 빈 값 처리
    # ========================================
    lot_table["lot"] = (
        lot_table["lot"]
        .fillna("(no lot)")
        .astype(str)
        .str.strip()
        .replace({"": "(no lot)", "nan": "(no lot)"})
    )

    # 누락된 센터 컬럼 추가 (0으로 채움)
    for center in used_centers:
        if center not in lot_table.columns:
            lot_table[center] = 0

    # ========================================
    # 9단계: 수량 정규화 및 합계 계산
    # ========================================
    value_cols = [c for c in lot_table.columns if c != "lot"]
    lot_table[value_cols] = lot_table[value_cols].fillna(0)
    lot_table[value_cols] = lot_table[value_cols].applymap(lambda x: int(round(x)))

    lot_table["합계"] = lot_table[
        [c for c in used_centers if c in lot_table.columns]
    ].sum(axis=1)
    lot_table = lot_table[lot_table["합계"] > 0]

    # ========================================
    # 10단계: 테이블 렌더링
    # ========================================
    if lot_table.empty:
        st.caption("해당 조건의 로트 상세가 없습니다.")
    else:
        ordered_cols = ["lot"] + [c for c in used_centers if c in lot_table.columns]
        ordered_cols.append("합계")

        st.dataframe(
            lot_table[ordered_cols]
            .sort_values("합계", ascending=False)
            .reset_index(drop=True),
            use_container_width=True,
            height=CONFIG.ui.table_height_lot,
        )
