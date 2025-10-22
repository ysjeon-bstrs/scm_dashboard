"""Amazon Forecast Context 빌더 헬퍼 함수 모듈.

build_amazon_forecast_context 함수를 작은 함수들로 분해한 헬퍼들입니다.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from center_alias import normalize_center_value


def calculate_promotion_multiplier(promotion_events: Optional[Iterable[dict]]) -> float:
    """프로모션 이벤트들로부터 전체 승수를 계산합니다.

    Args:
        promotion_events: 프로모션 이벤트 목록

    Returns:
        float: 프로모션 승수 (기본값 1.0)
    """
    promo_multiplier = 1.0
    if promotion_events:
        for event in promotion_events:
            try:
                uplift_val = float(event.get("uplift", 0.0))
            except (TypeError, ValueError):
                continue
            uplift_val = min(3.0, max(-1.0, uplift_val))
            promo_multiplier *= 1.0 + uplift_val
    if not np.isfinite(promo_multiplier) or promo_multiplier <= 0:
        promo_multiplier = 1.0
    return promo_multiplier


def normalize_inputs(
    centers: Sequence[str],
    skus: Sequence[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    today: pd.Timestamp,
) -> tuple[list[str], list[str], pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """입력 파라미터들을 정규화합니다.

    Returns:
        tuple: (center_list, sku_list, start_norm, end_norm, today_norm)
    """
    start_norm = pd.to_datetime(start).normalize()
    end_norm = pd.to_datetime(end).normalize()
    today_norm = pd.to_datetime(today).normalize()

    center_list: list[str] = []
    for raw_center in centers:
        normalized = normalize_center_value(raw_center)
        if not normalized:
            continue
        if normalized not in center_list:
            center_list.append(normalized)
    sku_list = [str(sku) for sku in skus if str(sku).strip()]

    return center_list, sku_list, start_norm, end_norm, today_norm


def calculate_latest_snapshot(snap_long: pd.DataFrame, today_norm: pd.Timestamp) -> pd.Timestamp:
    """스냅샷에서 최신 날짜를 계산합니다.

    Args:
        snap_long: 스냅샷 DataFrame
        today_norm: 현재 날짜 (fallback용)

    Returns:
        pd.Timestamp: 최신 스냅샷 날짜
    """
    snapshot_cols = {c.lower(): c for c in snap_long.columns}
    snap_date_col = snapshot_cols.get("snapshot_date") or snapshot_cols.get("date")

    if snap_date_col is not None:
        snap_dates = pd.to_datetime(snap_long[snap_date_col], errors="coerce")
        latest_snapshot = snap_dates.dropna().max()
    else:
        latest_snapshot = pd.NaT

    if pd.isna(latest_snapshot):
        latest_snapshot = today_norm
    else:
        latest_snapshot = pd.to_datetime(latest_snapshot).normalize()

    return latest_snapshot


def process_inventory_data(
    timeline: pd.DataFrame,
    today_norm: pd.Timestamp,
    center_list: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Timeline으로부터 inv_actual과 inv_forecast를 추출합니다.

    Args:
        timeline: build_timeline 결과
        today_norm: 현재 날짜
        center_list: 센터 목록

    Returns:
        tuple: (inv_actual, inv_forecast) - 정규화된 재고 DataFrames
    """
    timeline = timeline[timeline["center"].isin(center_list)].copy()
    timeline["date"] = pd.to_datetime(timeline["date"], errors="coerce").dt.normalize()
    timeline["stock_qty"] = pd.to_numeric(timeline.get("stock_qty"), errors="coerce").fillna(0)

    inv_actual = timeline[timeline["date"] <= today_norm].copy()

    if not inv_actual.empty:
        last_actual_dates = inv_actual.groupby(["center", "resource_code"])["date"].max()
        mask = inv_actual.set_index(["center", "resource_code", "date"]).index.isin(
            [(c, r, d) for (c, r), d in last_actual_dates.items()]
        )
        inv_actual = inv_actual[mask.values].copy()

    inv_actual["stock_qty"] = inv_actual["stock_qty"].round().clip(lower=0).astype(int)

    return inv_actual, timeline


def process_sales_history(
    sales_hist: pd.DataFrame,
    center_list: list[str],
    sku_list: list[str],
) -> pd.DataFrame:
    """판매 이력 데이터를 정규화하고 필터링합니다.

    Args:
        sales_hist: 원본 판매 이력
        center_list: 센터 목록
        sku_list: SKU 목록

    Returns:
        pd.DataFrame: 정규화된 판매 이력
    """
    sales_hist = sales_hist.copy()
    sales_hist["date"] = pd.to_datetime(sales_hist.get("date"), errors="coerce").dt.normalize()
    sales_hist["sales_ea"] = pd.to_numeric(sales_hist.get("sales_ea"), errors="coerce").fillna(0).astype(int)
    sales_hist = sales_hist.dropna(subset=["date"])
    sales_hist = sales_hist[
        (sales_hist["center"].isin(center_list)) & (sales_hist["resource_code"].isin(sku_list))
    ]

    return sales_hist
