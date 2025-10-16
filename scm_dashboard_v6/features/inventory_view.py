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
