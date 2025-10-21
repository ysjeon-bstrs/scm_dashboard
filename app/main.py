from __future__ import annotations

from typing import Optional

import pandas as pd
import streamlit as st

from app.pages import data_sources, filters, inventory, timeline


def _normalize_snapshot_dates(snapshot_df: pd.DataFrame) -> pd.DataFrame:
    snapshot = snapshot_df.copy()
    if "date" in snapshot.columns:
        snapshot["date"] = pd.to_datetime(snapshot["date"], errors="coerce").dt.normalize()
    elif "snapshot_date" in snapshot.columns:
        snapshot["date"] = pd.to_datetime(snapshot["snapshot_date"], errors="coerce").dt.normalize()
    else:
        snapshot["date"] = pd.NaT
    return snapshot


def main() -> None:
    """Entrypoint for running the v5 dashboard in Streamlit."""

    st.set_page_config(page_title="SCM Dashboard v5", layout="wide")
    st.title("SCM Dashboard v5")
    st.caption("모듈화된 v5 파이프라인을 이용한 Streamlit 엔트리 포인트")

    outcome = data_sources.ensure_data()
    if outcome is None:
        st.info("데이터를 로드하면 차트와 테이블이 표시됩니다.")
        return

    data = outcome.data
    snapshot_df = _normalize_snapshot_dates(data.snapshot)

    centers, skus = filters.derive_center_and_sku_options(data.moves, snapshot_df)
    if not centers or not skus:
        st.warning("센터 또는 SKU 정보를 찾을 수 없습니다.")
        return

    today = pd.Timestamp.today().normalize()
    latest_dt_series = snapshot_df.get("date", pd.Series(dtype="datetime64[ns]"))
    snap_dates = latest_dt_series.dropna()
    latest_dt = snap_dates.max() if not snap_dates.empty else pd.NaT
    latest_snapshot_dt: Optional[pd.Timestamp]
    if pd.isna(latest_dt):
        latest_snapshot_dt = None
    else:
        latest_snapshot_dt = pd.to_datetime(latest_dt).normalize()

    base_past_days = 42
    base_future_days = 42
    bound_min, bound_max = filters.calculate_date_bounds(
        today=today,
        snapshot_df=snapshot_df,
        moves_df=data.moves,
        base_past_days=base_past_days,
        base_future_days=base_future_days,
    )

    controls = filters.render_sidebar_controls(
        centers=centers,
        skus=skus,
        bound_min=bound_min,
        bound_max=bound_max,
    )
    if controls is None:
        return

    artifacts = timeline.render_summary_and_timeline(
        moves=data.moves,
        snapshot_df=snapshot_df,
        controls=controls,
        latest_snapshot_dt=latest_snapshot_dt,
    )
    if artifacts is None:
        return

    timeline.render_amazon_section(
        timeline_for_chart=artifacts.timeline_for_chart,
        snapshot_df=snapshot_df,
        moves=data.moves,
        controls=controls,
        today=artifacts.today,
    )

    inventory.render_inbound_and_wip(
        moves=data.moves,
        snapshot_df=snapshot_df,
        controls=controls,
        artifacts=artifacts,
    )
    inventory.render_snapshot_tables(
        snapshot_df=snapshot_df,
        controls=controls,
        artifacts=artifacts,
    )


if __name__ == "__main__":
    main()
