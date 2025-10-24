"""
인벤토리 섹션 유스케이스 (v7)

설명(한글):
- v5의 피벗/비용/LOT 상세 로직을 래핑하여 동일 동작을 제공합니다.
- UI 메시지 출력은 상위(app)에서 담당하며, 본 모듈은 DataFrame을 반환합니다.
"""

from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd
from scm_dashboard_v4.config import CENTER_COL
from scm_dashboard_v5.analytics.inventory import pivot_inventory_cost_from_raw


def render_inventory_pivot(
    *,
    snapshot: pd.DataFrame,
    centers: Iterable[str],
    latest_snapshot: pd.Timestamp,
    resource_name_map: Optional[dict[str, str]] = None,
    load_snapshot_raw_fn: Optional[callable] = None,
    snapshot_raw: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    선택 센터의 최신 스냅샷 기준 피벗 테이블을 생성한다. (표시는 상위에서 처리)
    """

    if snapshot.empty or pd.isna(latest_snapshot):
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

    # (옵션) 비용 합성
    raw_df = snapshot_raw if snapshot_raw is not None else (load_snapshot_raw_fn() if load_snapshot_raw_fn else None)
    if raw_df is not None and not raw_df.empty:
        cost_pivot = pivot_inventory_cost_from_raw(
            raw_df,
            latest_snapshot,
            selected_centers,
            center_latest_dates=(snapshot[snapshot["center"].isin(selected_centers)].groupby("center")["date"].max().dropna().to_dict()),
        )
        if not cost_pivot.empty:
            cost_cols = [c for c in cost_pivot.columns if c.endswith("_재고자산")]
            cost_pivot["resource_code"] = cost_pivot["resource_code"].astype(str)
            merged = pivot.reset_index().merge(
                cost_pivot.rename(columns={"resource_code": "resource_code"}),
                on="resource_code",
                how="left",
            )
            merged = merged.set_index("resource_code")
            pivot = merged

    return pivot.reset_index().rename(columns={"resource_code": "SKU"})


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
    """
    확정 입고(Upcoming Inbound) 표 데이터 생성은 상위(app)에서 표시.
    (v7에서는 UI 처리 분리를 위해 데이터 조작을 최소화합니다.)
    """
    # 표시 전용으로 v6 구현과 동일 데이터 필드가 채워지도록 보장하려면
    # 상위(app)에서 기존 v5/v6 동일 로직을 사용해 테이블을 직접 구성하도록 유지합니다.
    return None


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
    """
    WIP(생산중) 진행 표 데이터 생성은 상위(app)에서 표시.
    (v7에서는 UI 처리 분리를 위해 데이터 조작을 최소화합니다.)
    """
    return None


