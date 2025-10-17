"""
인벤토리 섹션 유스케이스 (v6)

- 초기에는 v5 로직을 재사용하여 피벗/CSV 표시를 지원한다.
- 후속 단계에서 비용/LOT 상세 등 세부 표현을 이 모듈로 이전한다.
"""

from __future__ import annotations

from typing import Iterable, Optional, Callable

import pandas as pd
import streamlit as st
from scm_dashboard_v4.config import CENTER_COL
from scm_dashboard_v5.analytics.inventory import pivot_inventory_cost_from_raw


def render_inventory_pivot(
    *,
    snapshot: pd.DataFrame,
    centers: Iterable[str],
    latest_snapshot: pd.Timestamp,
    resource_name_map: Optional[dict[str, str]] = None,
    load_snapshot_raw_fn: Optional[Callable[[], pd.DataFrame]] = None,
    snapshot_raw: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """선택 센터의 최신 스냅샷 기준 피벗 테이블 + 검색/정렬 + LOT 상세를 렌더링한다."""

    if snapshot.empty or pd.isna(latest_snapshot):
        st.info("스냅샷 데이터가 없습니다.")
        return pd.DataFrame()

    selected_centers = [str(c) for c in centers]

    # 헤더: 스냅샷 날짜/센터별 최신 스냅샷
    latest_dt_str = pd.to_datetime(latest_snapshot).strftime("%Y-%m-%d")
    st.subheader(f"선택 센터 현재 재고 (스냅샷 {latest_dt_str} / 전체 SKU)")
    center_latest_series = (
        snapshot[snapshot["center"].isin(selected_centers)].groupby("center")["date"].max()
    )
    center_latest_dates = {
        center: ts.normalize() for center, ts in center_latest_series.items() if pd.notna(ts)
    }
    if not center_latest_series.empty:
        caption_items = [
            f"{center}: {center_latest_dates[center].strftime('%Y-%m-%d')}"
            for center in selected_centers
            if center in center_latest_dates
        ]
        if caption_items:
            st.caption("센터별 최신 스냅샷: " + " / ".join(caption_items))

    # 최신값 기준 피벗 생성
    sub = snapshot[(snapshot["date"] <= latest_snapshot) & (snapshot["center"].isin(selected_centers))].copy()
    if not sub.empty:
        sub["center"] = sub["center"].astype(str).str.strip()
        sub = (
            sub.sort_values(["center", "resource_code", "date"]).groupby(["center", "resource_code"], as_index=False).tail(1)
        )

    pivot = (
        sub.groupby(["resource_code", "center"], as_index=False)["stock_qty"].sum()
        .pivot(index="resource_code", columns="center", values="stock_qty")
        .reindex(columns=selected_centers)
        .fillna(0)
    )
    pivot = pivot.astype(int)
    pivot["총합"] = pivot.sum(axis=1)

    # 검색/정렬 UI
    col_filter, col_sort = st.columns([2, 1])
    with col_filter:
        sku_query = st.text_input(
            "SKU 필터 — 품목번호 검색 시 해당 SKU의 센터별 제조번호(LOT) 확인",
            "",
            key="v6_sku_filter_text",
        )
    with col_sort:
        sort_candidates = ["총합"] + selected_centers
        sort_by = st.selectbox("정렬 기준", sort_candidates, index=0)

    # 표시 옵션: 총합 0 숨기기 / 재고자산(제조원가)
    opt_col1, opt_col2 = st.columns([1, 1])
    with opt_col1:
        hide_zero = st.checkbox("총합=0 숨기기", value=True, key="v6_hide_zero_total")
    with opt_col2:
        show_cost = st.checkbox("재고자산(제조원가) 표시", value=False, key="v6_show_cost")

    view = pivot.copy()
    if sku_query.strip():
        view = view[view.index.astype(str).str.contains(sku_query.strip(), case=False, regex=False)]
    if sort_by in view.columns:
        view = view.sort_values(by=sort_by, ascending=False)
    if hide_zero and "총합" in view.columns:
        view = view[view["총합"] > 0]

    display_df = view.reset_index().rename(columns={"resource_code": "SKU"})
    if resource_name_map:
        display_df.insert(1, "품명", display_df["SKU"].map(resource_name_map).fillna(""))

    # 재고자산(제조원가) 계산/표시 (snapshot_raw 필요)
    if show_cost:
        raw_df = snapshot_raw if snapshot_raw is not None else (load_snapshot_raw_fn() if load_snapshot_raw_fn else None)
        if raw_df is None or raw_df.empty:
            st.caption("재고자산 계산을 위한 snapshot_raw 데이터가 없습니다.")
        else:
            # 센터별 최신 스냅샷 날짜 맵을 전달해 v5/v4 헬퍼로 비용 피벗 계산
            cost_pivot = pivot_inventory_cost_from_raw(
                raw_df,
                latest_snapshot,
                selected_centers,
                center_latest_dates=center_latest_dates,
            )
            if not cost_pivot.empty:
                cost_cols = [c for c in cost_pivot.columns if c.endswith("_재고자산")]
                cost_pivot["resource_code"] = cost_pivot["resource_code"].astype(str)
                merged = display_df.merge(
                    cost_pivot.rename(columns={"resource_code": "SKU"}),
                    on="SKU",
                    how="left",
                )
                if cost_cols:
                    merged[cost_cols] = merged[cost_cols].fillna(0).round().astype(int)
                    merged["총합_재고자산"] = merged[cost_cols].sum(axis=1)
                display_df = merged

    # 숫자 표기: 수량/총합은 천단위 구분. 비용은 '원' 단위 표기
    try:
        cost_cols = [c for c in display_df.columns if str(c).endswith("_재고자산") or c == "총 재고자산"]
        qty_cols = [
            c for c in display_df.columns
            if c not in {"SKU", "품명"} and c not in cost_cols
        ]
        df_fmt = display_df.copy()
        for col in qty_cols:
            if pd.api.types.is_numeric_dtype(df_fmt[col]):
                df_fmt[col] = df_fmt[col].apply(lambda x: f"{int(x):,}" if pd.notna(x) else x)
        for col in cost_cols:
            if pd.api.types.is_numeric_dtype(df_fmt[col]):
                df_fmt[col] = df_fmt[col].apply(lambda x: f"{int(x):,}원" if pd.notna(x) else x)
        st.dataframe(df_fmt, use_container_width=True, height=380)
    except Exception:
        st.dataframe(display_df, use_container_width=True, height=380)

    csv_bytes = display_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "현재 표 CSV 다운로드",
        data=csv_bytes,
        file_name=f"centers_{'-'.join(selected_centers)}_snapshot_{latest_dt_str}.csv",
        mime="text/csv",
    )

    # LOT 상세 (단일 SKU 선택 시)
    filtered_df = (
        display_df if "SKU" in display_df.columns else view.reset_index().rename(columns={"resource_code": "SKU"})
    )
    visible_skus = filtered_df.get("SKU", pd.Series(dtype=str)).dropna().astype(str).unique().tolist()
    if len(visible_skus) == 1:
        lot_sku = visible_skus[0]
        raw_df = snapshot_raw if snapshot_raw is not None else (load_snapshot_raw_fn() if load_snapshot_raw_fn else None)
        if raw_df is None or raw_df.empty:
            st.caption("해당 조건의 로트 상세가 없습니다. (snapshot_raw 없음)")
        else:
            raw = raw_df.copy()
            cols_map = {str(col).strip().lower(): col for col in raw.columns}
            col_date = cols_map.get("snapshot_date") or cols_map.get("date")
            col_sku = cols_map.get("resource_code") or cols_map.get("sku") or cols_map.get("상품코드")
            col_lot = cols_map.get("lot")
            used_centers = [ct for ct in selected_centers if CENTER_COL.get(ct) in raw.columns]
            st.markdown(
                f"### 로트 상세 (스냅샷 {latest_dt_str} / **{', '.join(selected_centers)}** / **{lot_sku}** )"
            )
            if not all([col_date, col_sku, col_lot]) or not used_centers:
                st.caption("해당 조건의 로트 상세가 없습니다.")
            else:
                raw[col_date] = pd.to_datetime(raw[col_date], errors="coerce").dt.normalize()
                lot_subset = raw[(raw[col_date] == latest_snapshot) & (raw[col_sku].astype(str) == str(lot_sku))].copy()
                if lot_subset.empty:
                    st.caption("해당 조건의 로트 상세가 없습니다.")
                else:
                    for center in used_centers:
                        src_col = CENTER_COL.get(center)
                        lot_subset[src_col] = pd.to_numeric(lot_subset[src_col], errors="coerce").fillna(0).clip(lower=0)
                    lot_table = pd.DataFrame({"lot": lot_subset[col_lot].astype(str).fillna("(no lot)")})
                    for center in used_centers:
                        src_col = CENTER_COL.get(center)
                        lot_table[center] = lot_subset.groupby(col_lot)[src_col].transform("sum")
                    lot_table = lot_table.drop_duplicates()
                    lot_table["합계"] = lot_table[used_centers].sum(axis=1)
                    lot_table = lot_table[lot_table["합계"] > 0]
                    if lot_table.empty:
                        st.caption("해당 조건의 로트 상세가 없습니다.")
                    else:
                        st.dataframe(
                            lot_table[["lot"] + used_centers + ["합계"]]
                            .sort_values("합계", ascending=False)
                            .reset_index(drop=True),
                            use_container_width=True,
                            height=320,
                        )

    return display_df


def _ensure_move_columns(moves: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "carrier_mode",
        "to_center",
        "resource_code",
        "qty_ea",
        "inbound_date",
        "arrival_date",
        "onboard_date",
        "event_date",
        "lot",
    ]
    mv = moves.copy()
    for col in cols:
        if col not in mv.columns:
            if "date" in col:
                mv[col] = pd.Series(pd.NaT, index=mv.index, dtype="datetime64[ns]")
            else:
                mv[col] = 0 if col == "qty_ea" else ""
        # 날짜형 컬럼은 문자열이어도 비교 가능하도록 datetime으로 강제 변환
        if "date" in col and col in mv.columns:
            mv[col] = pd.to_datetime(mv[col], errors="coerce")
    return mv


def _attach_pred_inbound(mv: pd.DataFrame, *, today: pd.Timestamp, lag_days: int) -> pd.DataFrame:
    mv = mv.copy()
    pred = pd.Series(pd.NaT, index=mv.index, dtype="datetime64[ns]")
    if "inbound_date" in mv.columns:
        mask_in = mv["inbound_date"].notna()
        pred.loc[mask_in] = mv.loc[mask_in, "inbound_date"]
    mask_in = mv["inbound_date"].notna() if "inbound_date" in mv.columns else pd.Series(False, index=mv.index)
    arr = mv.get("arrival_date")
    mask_arrival = (~mask_in) & arr.notna() if arr is not None else pd.Series(False, index=mv.index)
    if mask_arrival.any():
        past_arr = mask_arrival & (arr <= today)
        if past_arr.any():
            pred.loc[past_arr] = mv.loc[past_arr, "arrival_date"] + pd.Timedelta(days=int(lag_days))
        fut_arr = mask_arrival & (arr > today)
        if fut_arr.any():
            pred.loc[fut_arr] = mv.loc[fut_arr, "arrival_date"]
    mv["pred_inbound_date"] = pred
    return mv


def render_upcoming_inbound(
    *,
    moves: pd.DataFrame,
    centers: Iterable[str],
    skus: Iterable[str],
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    today: pd.Timestamp,
    lag_days: int,
    resource_name_map: Optional[dict[str, str]] = None,
) -> None:
    """확정 입고(Upcoming Inbound) 표를 렌더링한다 (v6)."""

    mv = _ensure_move_columns(moves)
    mv = _attach_pred_inbound(mv, today=today, lag_days=lag_days)
    arr_transport = mv[
        (mv["carrier_mode"] != "WIP")
        & (mv["to_center"].isin([str(c) for c in centers]))
        & (mv["resource_code"].isin([str(s) for s in skus]))
        & (mv["inbound_date"].isna())
    ].copy()
    arr_transport["display_date"] = arr_transport["arrival_date"].fillna(arr_transport["onboard_date"])
    arr_transport = arr_transport[arr_transport["display_date"].notna()]
    arr_transport = arr_transport[(arr_transport["display_date"] >= window_start) & (arr_transport["display_date"] <= window_end)]

    st.markdown("#### ✅ 확정 입고 (Upcoming Inbound)")
    if arr_transport.empty:
        st.caption("선택한 조건에서 예정된 운송 입고가 없습니다. (오늘 이후 / 선택 기간)")
        return

    if resource_name_map:
        arr_transport["resource_name"] = arr_transport["resource_code"].map(resource_name_map).fillna("")
    arr_transport["days_to_arrival"] = (arr_transport["display_date"].dt.normalize() - today).dt.days
    arr_transport["days_to_inbound"] = (arr_transport["pred_inbound_date"].dt.normalize() - today).dt.days
    arr_transport = arr_transport.sort_values(
        ["display_date", "to_center", "resource_code", "qty_ea"], ascending=[True, True, True, False]
    )
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
    inbound_cols = [c for c in inbound_cols if c in arr_transport.columns]
    st.dataframe(arr_transport[inbound_cols].head(1000), use_container_width=True, height=300)
    st.caption("※ pred_inbound_date: 예상 입고일 (도착일 + 리드타임), days_to_inbound: 예상 입고까지 남은 일수")


def render_wip_progress(
    *,
    moves: pd.DataFrame,
    centers: Iterable[str],
    skus: Iterable[str],
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    today: pd.Timestamp,
    resource_name_map: Optional[dict[str, str]] = None,
) -> None:
    """생산중(WIP) 진행 현황 표를 렌더링한다 (v6)."""

    mv = _ensure_move_columns(moves)
    arr_wip = mv[
        (mv["carrier_mode"] == "WIP")
        & (mv["to_center"].isin(["태광KR"]))
        & (mv["resource_code"].isin([str(s) for s in skus]))
        & (mv["event_date"].notna())
        & (mv["event_date"] >= window_start)
        & (mv["event_date"] <= window_end)
    ].copy()
    arr_wip["display_date"] = arr_wip["event_date"]

    st.markdown("#### 🛠 생산중 (WIP) 진행 현황")
    # v5 동작: 선택 센터에 태광KR 포함 시에만 의미 있는 WIP 표시
    if "태광KR" not in [str(c) for c in centers]:
        st.caption("선택 센터에 태광KR이 포함되어야 생산중(WIP) 표가 표시됩니다.")
        return
    if arr_wip.empty:
        st.caption("생산중(WIP) 데이터가 없습니다.")
        return

    if resource_name_map:
        arr_wip["resource_name"] = arr_wip["resource_code"].map(resource_name_map).fillna("")
    arr_wip = arr_wip.sort_values(["display_date", "resource_code", "qty_ea"], ascending=[True, True, False])
    arr_wip["days_to_completion"] = (arr_wip["display_date"].dt.normalize() - today).dt.days
    wip_cols = [
        "display_date",
        "days_to_completion",
        "resource_code",
        "resource_name",
        "qty_ea",
        "pred_inbound_date",
        "lot",
    ]
    wip_cols = [c for c in wip_cols if c in arr_wip.columns]
    st.dataframe(arr_wip[wip_cols].head(1000), use_container_width=True, height=260)

