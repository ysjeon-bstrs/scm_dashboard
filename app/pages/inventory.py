from __future__ import annotations

from typing import Dict, Sequence

import pandas as pd
import streamlit as st

from scm_dashboard_v4.config import CENTER_COL
from scm_dashboard_v4.inventory import pivot_inventory_cost_from_raw
from scm_dashboard_v4.loaders import load_snapshot_raw

from .filters import FilterControls
from .timeline import TimelineArtifacts


def _prepare_moves_view(moves: pd.DataFrame) -> pd.DataFrame:
    moves_view = moves.copy()
    for col in [
        "carrier_mode",
        "to_center",
        "resource_code",
        "inbound_date",
        "arrival_date",
        "onboard_date",
        "event_date",
        "lot",
    ]:
        if col not in moves_view.columns:
            if "date" in col:
                moves_view[col] = pd.Series(pd.NaT, index=moves_view.index, dtype="datetime64[ns]")
            else:
                moves_view[col] = pd.Series("", index=moves_view.index, dtype="object")

    if not moves_view.empty:
        pred_inbound = pd.Series(pd.NaT, index=moves_view.index, dtype="datetime64[ns]")

        if "inbound_date" in moves_view.columns:
            mask_inbound = moves_view["inbound_date"].notna()
            pred_inbound.loc[mask_inbound] = moves_view.loc[mask_inbound, "inbound_date"]
        else:
            mask_inbound = pd.Series(False, index=moves_view.index)

        arrival_series = moves_view.get("arrival_date")
        if arrival_series is not None:
            mask_arrival = (~mask_inbound) & arrival_series.notna()
        else:
            mask_arrival = pd.Series(False, index=moves_view.index)

        today = pd.Timestamp.today().normalize()
        if mask_arrival.any():
            past_arr = mask_arrival & (arrival_series <= today)
            if past_arr.any():
                pred_inbound.loc[past_arr] = moves_view.loc[past_arr, "arrival_date"] + pd.Timedelta(
                    days=int(st.session_state.get("lag_days_override", 0))
                )
            fut_arr = mask_arrival & (arrival_series > today)
            if fut_arr.any():
                pred_inbound.loc[fut_arr] = moves_view.loc[fut_arr, "arrival_date"]

        moves_view["pred_inbound_date"] = pred_inbound
    else:
        moves_view["pred_inbound_date"] = pd.Series(
            pd.NaT, index=moves_view.index, dtype="datetime64[ns]"
        )

    return moves_view


def render_inbound_and_wip(
    *,
    moves: pd.DataFrame,
    snapshot_df: pd.DataFrame,
    controls: FilterControls,
    artifacts: TimelineArtifacts,
) -> None:
    st.session_state["lag_days_override"] = int(controls.lag_days)

    moves_view = _prepare_moves_view(moves)

    today = pd.Timestamp.today().normalize()
    window_start = controls.start
    window_end = controls.end

    arr_transport = moves_view[
        (moves_view["carrier_mode"] != "WIP")
        & (moves_view["to_center"].isin(controls.centers))
        & (moves_view["resource_code"].isin(controls.skus))
        & (moves_view["inbound_date"].isna())
    ].copy()

    arr_transport["display_date"] = arr_transport["arrival_date"].fillna(arr_transport["onboard_date"])
    arr_transport = arr_transport[arr_transport["display_date"].notna()]
    arr_transport = arr_transport[
        (arr_transport["display_date"] >= window_start)
        & (arr_transport["display_date"] <= window_end)
    ]

    arr_wip = pd.DataFrame()
    if "태광KR" in controls.centers:
        arr_wip = moves_view[
            (moves_view["carrier_mode"] == "WIP")
            & (moves_view["to_center"] == "태광KR")
            & (moves_view["resource_code"].isin(controls.skus))
            & (moves_view["event_date"].notna())
            & (moves_view["event_date"] >= window_start)
            & (moves_view["event_date"] <= window_end)
        ].copy()
        arr_wip["display_date"] = arr_wip["event_date"]

    resource_name_map: Dict[str, str] = {}
    if "resource_name" in snapshot_df.columns:
        name_rows = snapshot_df.loc[
            snapshot_df["resource_name"].notna(), ["resource_code", "resource_name"]
        ].copy()
        name_rows["resource_code"] = name_rows["resource_code"].astype(str)
        name_rows["resource_name"] = name_rows["resource_name"].astype(str).str.strip()
        name_rows = name_rows[name_rows["resource_name"] != ""]
        if not name_rows.empty:
            resource_name_map = (
                name_rows.drop_duplicates("resource_code").set_index("resource_code")["resource_name"].to_dict()
            )

    confirmed_inbound = arr_transport.copy()
    if resource_name_map and not confirmed_inbound.empty:
        confirmed_inbound["resource_name"] = confirmed_inbound["resource_code"].map(resource_name_map).fillna("")

    st.markdown("#### ✅ 확정 입고 (Upcoming Inbound)")
    if confirmed_inbound.empty:
        st.caption("선택한 조건에서 예정된 운송 입고가 없습니다. (오늘 이후 / 선택 기간)")
    else:
        confirmed_inbound["days_to_arrival"] = (
            confirmed_inbound["display_date"].dt.normalize() - today
        ).dt.days
        confirmed_inbound["days_to_inbound"] = (
            confirmed_inbound["pred_inbound_date"].dt.normalize() - today
        ).dt.days
        confirmed_inbound = confirmed_inbound.sort_values(
            ["display_date", "to_center", "resource_code", "qty_ea"],
            ascending=[True, True, True, False],
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
        inbound_cols = [c for c in inbound_cols if c in confirmed_inbound.columns]
        st.dataframe(
            confirmed_inbound[inbound_cols].head(1000),
            use_container_width=True,
            height=300,
        )
        st.caption("※ pred_inbound_date: 예상 입고일 (도착일 + 리드타임), days_to_inbound: 예상 입고까지 남은 일수")

    st.markdown("#### 🛠 생산중 (WIP) 진행 현황")
    if not arr_wip.empty:
        if resource_name_map:
            arr_wip["resource_name"] = arr_wip["resource_code"].map(resource_name_map).fillna("")
        arr_wip = arr_wip.sort_values(
            ["display_date", "resource_code", "qty_ea"], ascending=[True, True, False]
        )
        arr_wip["days_to_completion"] = (
            arr_wip["display_date"].dt.normalize() - today
        ).dt.days
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
    else:
        st.caption("생산중(WIP) 데이터가 없습니다.")


def render_snapshot_tables(
    *,
    snapshot_df: pd.DataFrame,
    controls: FilterControls,
    artifacts: TimelineArtifacts,
) -> None:
    if snapshot_df.empty or "date" not in snapshot_df.columns:
        st.info("스냅샷 데이터가 없습니다.")
        return

    latest_dt_series = snapshot_df["date"].dropna()
    if latest_dt_series.empty:
        st.info("스냅샷 데이터의 날짜 정보를 확인할 수 없습니다.")
        return

    latest_dt = latest_dt_series.max()
    latest_dt_str = pd.to_datetime(latest_dt).strftime("%Y-%m-%d")
    st.subheader(f"선택 센터 현재 재고 (스냅샷 {latest_dt_str} / 전체 SKU)")

    center_latest_series = (
        snapshot_df[snapshot_df["center"].isin(controls.centers)]
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
            for center in controls.centers
            if center in center_latest_dates
        ]
        if caption_items:
            st.caption(
                "센터별 최신 스냅샷 기준일: " + ", ".join(caption_items)
            )

    pivot = (
        snapshot_df[snapshot_df["center"].isin(controls.centers)]
        .pivot_table(
            index="resource_code",
            columns="center",
            values="stock_qty",
            aggfunc="sum",
            fill_value=0,
        )
        .astype(int)
    )

    pivot["총합"] = pivot.sum(axis=1)

    st.text_input("SKU 필터", key="snapshot_sku_query", value="")
    sku_query = st.session_state.get("snapshot_sku_query", "")

    sort_candidates = ["총합"] + controls.centers
    sort_by = st.selectbox("정렬 기준", sort_candidates, index=0)

    col_zero, col_cost = st.columns(2)
    with col_zero:
        hide_zero = st.checkbox("총합=0 숨기기", value=True)
    with col_cost:
        show_cost = st.checkbox("재고자산(제조원가) 표시", value=False)

    view = pivot.copy()
    if sku_query.strip():
        view = view[view.index.astype(str).str.contains(sku_query.strip(), case=False, regex=False)]
    if hide_zero and "총합" in view.columns:
        view = view[view["총합"] > 0]
    if sort_by in view.columns:
        view = view.sort_values(by=sort_by, ascending=False)

    display_df = view.reset_index().rename(columns={"resource_code": "SKU"})

    resource_name_map: Dict[str, str] = {}
    if "resource_name" in snapshot_df.columns:
        name_rows = snapshot_df.loc[
            snapshot_df["resource_name"].notna(), ["resource_code", "resource_name"]
        ].copy()
        name_rows["resource_code"] = name_rows["resource_code"].astype(str)
        name_rows["resource_name"] = name_rows["resource_name"].astype(str).str.strip()
        name_rows = name_rows[name_rows["resource_name"] != ""]
        if not name_rows.empty:
            resource_name_map = (
                name_rows.drop_duplicates("resource_code").set_index("resource_code")["resource_name"].to_dict()
            )

    if resource_name_map:
        display_df.insert(1, "품명", display_df["SKU"].map(resource_name_map).fillna(""))

    cost_columns: Sequence[str] = []
    show_df = display_df
    if show_cost:
        snap_raw_df = load_snapshot_raw()
        cost_pivot = pivot_inventory_cost_from_raw(
            snap_raw_df, artifacts.latest_snapshot_dt, controls.centers, center_latest_dates
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
        ordered_columns.extend([c for c in quantity_columns if not c.endswith("_재고자산")])
        if "총합" in merged_df.columns:
            ordered_columns.append("총합")
        ordered_columns.extend(cost_columns)
        show_df = merged_df[ordered_columns]
    else:
        cost_columns = []

    qty_columns = [
        c
        for c in show_df.columns
        if c not in {"SKU", "품명"}
        and not c.endswith("_재고자산")
        and c != "총 재고자산"
    ]
    for column in qty_columns:
        show_df[column] = show_df[column].apply(
            lambda x: f"{int(x):,}" if pd.notna(x) and isinstance(x, (int, float)) else x
        )

    st.dataframe(show_df, use_container_width=True, height=380)

    csv_bytes = show_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "현재 표 CSV 다운로드",
        data=csv_bytes,
        file_name=f"centers_{'-'.join(controls.centers)}_snapshot_{latest_dt_str}.csv",
        mime="text/csv",
    )

    st.caption(
        "※ 이 표는 **선택된 센터 전체 SKU**의 최신 스냅샷 재고입니다. 상단 'SKU 선택'과 무관하게 모든 SKU가 포함됩니다."
    )

    filtered_df = (
        show_df if "SKU" in show_df.columns else view.reset_index().rename(columns={"resource_code": "SKU"})
    )
    visible_skus = filtered_df.get("SKU", pd.Series(dtype=str)).dropna().astype(str).unique().tolist()

    if len(visible_skus) == 1:
        lot_sku = visible_skus[0]
        snap_raw_df = load_snapshot_raw()
        if snap_raw_df is None or snap_raw_df.empty:
            st.markdown(
                f"### 로트 상세 (스냅샷 {latest_dt_str} / **{', '.join(controls.centers)}** / **{lot_sku}**)"
            )
            st.caption("해당 조건의 로트 상세가 없습니다. (snapshot_raw 없음)")
        else:
            raw_df = snap_raw_df.copy()
            cols_map = {str(col).strip().lower(): col for col in raw_df.columns}
            col_date = cols_map.get("snapshot_date") or cols_map.get("date")
            col_sku = (
                cols_map.get("resource_code")
                or cols_map.get("sku")
                or cols_map.get("상품코드")
            )
            col_lot = cols_map.get("lot")
            used_centers = [ct for ct in controls.centers if CENTER_COL.get(ct) in raw_df.columns]
            if not all([col_date, col_sku, col_lot]) or not used_centers:
                st.markdown(
                    f"### 로트 상세 (스냅샷 {latest_dt_str} / **{', '.join(controls.centers)}** / **{lot_sku}**)"
                )
                st.caption("해당 조건의 로트 상세가 없습니다.")
            else:
                raw_df[col_date] = pd.to_datetime(raw_df[col_date], errors="coerce").dt.normalize()
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
                    f"### 로트 상세 (스냅샷 {lot_title_dates} / **{', '.join(controls.centers)}** / **{lot_sku}**)"
                )

                lot_tables = []
                for center in used_centers:
                    src_col = CENTER_COL.get(center)
                    if not src_col or src_col not in raw_df.columns:
                        continue
                    target_date = lot_snapshot_dates.get(center)
                    if lot_snapshot_dates and pd.isna(target_date):
                        continue
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
                    center_subset[src_col] = (
                        pd.to_numeric(center_subset[src_col], errors="coerce").fillna(0).clip(lower=0)
                    )
                    center_table = (
                        center_subset[[col_lot, src_col]]
                        .groupby(col_lot, as_index=False)[src_col]
                        .sum()
                    )
                    center_table = center_table.rename(columns={col_lot: "lot", src_col: center})
                    lot_tables.append(center_table)

                if not lot_tables:
                    st.caption("해당 조건의 로트 상세가 없습니다.")
                else:
                    lot_table = lot_tables[0]
                    for tbl in lot_tables[1:]:
                        lot_table = lot_table.merge(tbl, on="lot", how="outer")
                    lot_table["lot"] = (
                        lot_table["lot"].fillna("(no lot)")
                        .astype(str)
                        .str.strip()
                        .replace({"": "(no lot)", "nan": "(no lot)"})
                    )
                    for center in used_centers:
                        if center not in lot_table.columns:
                            lot_table[center] = 0
                    value_cols = [c for c in lot_table.columns if c != "lot"]
                    lot_table[value_cols] = lot_table[value_cols].fillna(0)
                    lot_table[value_cols] = lot_table[value_cols].applymap(lambda x: int(round(x)))
                    lot_table["합계"] = lot_table[[c for c in used_centers if c in lot_table.columns]].sum(axis=1)
                    lot_table = lot_table[lot_table["합계"] > 0]
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
                            height=320,
                        )
