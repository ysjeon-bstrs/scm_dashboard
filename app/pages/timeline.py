from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
import streamlit as st

from scm_dashboard_v5.core import build_timeline as build_core_timeline
from scm_dashboard_v5.forecast import (
    apply_consumption_with_events,
    build_amazon_forecast_context,
)
from scm_dashboard_v5.ui import (
    render_amazon_sales_vs_inventory,
    render_sku_summary_cards,
)
from scm_dashboard_v5.ui.charts import _sku_color_map, _timeline_inventory_matrix

from scm_dashboard_v4.loaders import load_snapshot_raw

from .filters import FilterControls


@dataclass
class TimelineArtifacts:
    timeline_actual: pd.DataFrame
    timeline_forecast: pd.DataFrame
    timeline_for_chart: pd.DataFrame
    today: pd.Timestamp
    latest_snapshot_dt: Optional[pd.Timestamp]


def validate_timeline_inputs(
    snapshot: object,
    moves: object,
    start: object,
    end: object,
) -> bool:
    """Return True if the timeline inputs look structurally correct."""

    if not isinstance(snapshot, pd.DataFrame):
        st.error("스냅샷 데이터가 손상되었습니다. 엑셀/시트를 다시 불러와 주세요.")
        return False
    if not isinstance(moves, pd.DataFrame):
        st.error("이동 원장 데이터가 손상되었습니다. 엑셀/시트를 다시 불러와 주세요.")
        return False

    required_snapshot_cols = {"center", "resource_code", "stock_qty"}
    missing_snapshot = [col for col in required_snapshot_cols if col not in snapshot.columns]
    if missing_snapshot:
        st.error(
            "스냅샷 데이터에 필요한 컬럼이 없습니다: " + ", ".join(sorted(missing_snapshot))
        )
        return False

    required_move_cols = {"from_center", "to_center", "resource_code"}
    missing_moves = [col for col in required_move_cols if col not in moves.columns]
    if missing_moves:
        st.error(
            "이동 원장 데이터에 필요한 컬럼이 없습니다: " + ", ".join(sorted(missing_moves))
        )
        return False

    if not isinstance(start, pd.Timestamp) or not isinstance(end, pd.Timestamp):
        st.error("기간 정보가 손상되었습니다. 기간 슬라이더를 다시 설정해 주세요.")
        return False

    if end < start:
        st.error("기간의 종료일이 시작일보다 빠릅니다. 기간을 다시 선택하세요.")
        return False

    return True


def render_summary_and_timeline(
    *,
    moves: pd.DataFrame,
    snapshot_df: pd.DataFrame,
    controls: FilterControls,
    latest_snapshot_dt: Optional[pd.Timestamp],
) -> Optional[TimelineArtifacts]:
    st.subheader("요약 KPI")

    today_norm = pd.Timestamp.today().normalize()
    if latest_snapshot_dt is not None:
        proj_days_for_build = max(0, int((controls.end - latest_snapshot_dt).days))
    else:
        proj_days_for_build = max(0, int((controls.end - controls.start).days))

    render_sku_summary_cards(
        snapshot_df,
        moves,
        centers=controls.centers,
        skus=controls.skus,
        today=today_norm,
        latest_snapshot=latest_snapshot_dt,
        lag_days=int(controls.lag_days),
        start=controls.start,
        end=controls.end,
        lookback_days=controls.lookback_days,
        horizon_pad_days=60,
        events=controls.events,
    )

    st.divider()

    if not validate_timeline_inputs(snapshot_df, moves, controls.start, controls.end):
        return None

    timeline_actual = build_core_timeline(
        snapshot_df,
        moves,
        centers=controls.centers,
        skus=controls.skus,
        start=controls.start,
        end=controls.end,
        today=today_norm,
        lag_days=int(controls.lag_days),
        horizon_days=int(proj_days_for_build),
    )

    if timeline_actual is None or timeline_actual.empty:
        st.info("선택한 조건에 해당하는 타임라인 데이터가 없습니다.")
        return None

    cons_start = None
    if latest_snapshot_dt is not None:
        cons_start = (latest_snapshot_dt + pd.Timedelta(days=1)).normalize()

    timeline_forecast = apply_consumption_with_events(
        timeline_actual,
        snapshot_df,
        centers=controls.centers,
        skus=controls.skus,
        start=controls.start,
        end=controls.end,
        lookback_days=controls.lookback_days,
        events=controls.events,
        cons_start=cons_start,
    )

    if timeline_forecast is None or timeline_forecast.empty:
        timeline_forecast = timeline_actual.copy()

    timeline_for_chart = (
        timeline_forecast.copy() if controls.use_consumption_forecast else timeline_actual.copy()
    )

    render_step_chart(
        timeline_for_chart,
        start=controls.start,
        end=controls.end,
        centers=controls.centers,
        skus=controls.skus,
        show_production=controls.show_production,
        show_in_transit=controls.show_in_transit,
        today=today_norm,
    )

    return TimelineArtifacts(
        timeline_actual=timeline_actual,
        timeline_forecast=timeline_forecast,
        timeline_for_chart=timeline_for_chart,
        today=today_norm,
        latest_snapshot_dt=latest_snapshot_dt,
    )


def render_amazon_section(
    *,
    timeline_for_chart: pd.DataFrame,
    snapshot_df: pd.DataFrame,
    moves: pd.DataFrame,
    controls: FilterControls,
    today: pd.Timestamp,
) -> None:
    def _tidy_from_pivot(
        pivot: Optional[pd.DataFrame], mask: Optional[pd.Series]
    ) -> pd.DataFrame:
        if pivot is None or pivot.empty:
            return pd.DataFrame(columns=["date", "resource_code", "stock_qty"])
        subset = pivot if mask is None else pivot.loc[mask]
        if subset.empty:
            return pd.DataFrame(columns=["date", "resource_code", "stock_qty"])
        tidy = (
            subset.stack()
            .reset_index()
            .rename(columns={"level_0": "date", "level_1": "resource_code", 0: "stock_qty"})
        )
        tidy["date"] = pd.to_datetime(tidy["date"]).dt.normalize()
        tidy["stock_qty"] = pd.to_numeric(tidy["stock_qty"], errors="coerce").fillna(0)
        return tidy

    amazon_centers = [
        c
        for c in controls.centers
        if isinstance(c, str) and (c.upper().startswith("AMZ") or "AMAZON" in c.upper())
    ]
    if not amazon_centers and "AMZUS" in controls.centers:
        amazon_centers = ["AMZUS"]

    st.divider()
    st.subheader("Amazon US 일별 판매 vs. 재고")

    if not amazon_centers:
        st.info("Amazon 계열 센터가 선택되지 않았습니다.")
        return

    amz_inv_pivot = _timeline_inventory_matrix(
        timeline_for_chart,
        centers=amazon_centers,
        skus=controls.skus,
        start=controls.start,
        end=controls.end,
    )
    if amz_inv_pivot is not None:
        mask_actual = amz_inv_pivot.index <= today
        mask_forecast = amz_inv_pivot.index > today
    else:
        mask_actual = None
        mask_forecast = None

    inv_actual_from_step = _tidy_from_pivot(amz_inv_pivot, mask_actual)
    inv_forecast_from_step = _tidy_from_pivot(amz_inv_pivot, mask_forecast)
    sku_colors_map = _sku_color_map(controls.skus)

    snapshot_raw_df = load_snapshot_raw()
    amz_ctx = build_amazon_forecast_context(
        snap_long=snapshot_df,
        moves=moves,
        snapshot_raw=snapshot_raw_df,
        centers=amazon_centers,
        skus=controls.skus,
        start=controls.start,
        end=controls.end,
        today=today,
        lookback_days=int(controls.lookback_days),
        promotion_events=controls.events,
        use_consumption_forecast=controls.use_consumption_forecast,
    )
    render_amazon_sales_vs_inventory(
        amz_ctx,
        inv_actual=inv_actual_from_step,
        inv_forecast=inv_forecast_from_step,
        sku_colors=sku_colors_map,
        use_inventory_for_sales=True,
    )
