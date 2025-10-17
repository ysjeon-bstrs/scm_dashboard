"""
인벤토리 섹션 유스케이스 (v6)

- 초기에는 v5 로직을 재사용하여 피벗/CSV 표시를 지원한다.
- 후속 단계에서 비용/LOT 상세 등 세부 표현을 이 모듈로 이전한다.
"""

from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd
import streamlit as st


def render_inventory_pivot(
    *,
    snapshot: pd.DataFrame,
    centers: Iterable[str],
    latest_snapshot: pd.Timestamp,
    resource_name_map: Optional[dict[str, str]] = None,
) -> pd.DataFrame:
    """선택 센터의 최신 스냅샷 기준 피벗 테이블을 렌더링한다.

    반환값: 화면 표시용 데이터프레임(다운로드/후속 처리에 재사용 가능)
    """

    if snapshot.empty or pd.isna(latest_snapshot):
        st.info("스냅샷 데이터가 없습니다.")
        return pd.DataFrame()

    selected_centers = [str(c) for c in centers]

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

    display_df = pivot.reset_index().rename(columns={"resource_code": "SKU"})
    if resource_name_map:
        display_df.insert(1, "품명", display_df["SKU"].map(resource_name_map).fillna(""))

    st.dataframe(display_df, use_container_width=True, height=380)

    csv_bytes = display_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "현재 표 CSV 다운로드",
        data=csv_bytes,
        file_name=f"centers_{'-'.join(selected_centers)}_snapshot_{latest_snapshot.strftime('%Y-%m-%d')}.csv",
        mime="text/csv",
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

