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


def build_inbound_lookup(
    moves_df: pd.DataFrame,
    future_index: pd.DatetimeIndex,
    center_list: list[str],
    sku_list: list[str],
) -> dict[tuple[str, str], pd.Series]:
    """Moves 데이터로부터 입고 예정 lookup을 생성합니다.

    Args:
        moves_df: 이동 데이터
        future_index: 예측 기간 DatetimeIndex
        center_list: 센터 목록
        sku_list: SKU 목록

    Returns:
        dict: (center, sku) -> Series of inbound quantities by date
    """
    inbound_lookup: dict[tuple[str, str], pd.Series] = {}

    if future_index.empty or moves_df.empty:
        return inbound_lookup

    move_cols = {str(c).strip().lower(): c for c in moves_df.columns}
    event_col = move_cols.get("event_date")
    to_center_col = move_cols.get("to_center")
    sku_col = move_cols.get("resource_code") or move_cols.get("sku")
    qty_col = move_cols.get("qty_ea") or move_cols.get("qty")

    if not (event_col and to_center_col and sku_col and qty_col):
        return inbound_lookup

    inbound_norm = moves_df.rename(
        columns={
            event_col: "event_date",
            to_center_col: "to_center",
            sku_col: "resource_code",
            qty_col: "qty_ea",
        }
    ).copy()
    inbound_norm["event_date"] = pd.to_datetime(
        inbound_norm.get("event_date"), errors="coerce"
    ).dt.normalize()
    inbound_norm = inbound_norm.dropna(subset=["event_date"])
    inbound_norm["to_center"] = inbound_norm.get("to_center", "").apply(
        normalize_center_value
    )
    inbound_norm["resource_code"] = inbound_norm.get("resource_code", "").astype(str)
    inbound_norm["qty_ea"] = pd.to_numeric(
        inbound_norm.get("qty_ea"), errors="coerce"
    ).fillna(0.0)

    inbound_norm = inbound_norm[
        inbound_norm["to_center"].isin(center_list)
        & inbound_norm["resource_code"].isin(sku_list)
    ]

    if not inbound_norm.empty:
        start_future = future_index[0]
        end_future = future_index[-1]
        inbound_norm = inbound_norm[
            (inbound_norm["event_date"] >= start_future)
            & (inbound_norm["event_date"] <= end_future)
        ]

        if not inbound_norm.empty:
            inbound_grouped = (
                inbound_norm.groupby(
                    ["to_center", "resource_code", "event_date"],
                    as_index=False,
                )["qty_ea"].sum()
            )

            for (ct, sku), chunk in inbound_grouped.groupby(
                ["to_center", "resource_code"], dropna=True
            ):
                series = (
                    chunk.set_index("event_date")["qty_ea"]
                    .reindex(future_index, fill_value=0.0)
                )
                inbound_lookup[(ct, sku)] = series

    return inbound_lookup


def calculate_promotion_uplift(
    promotion_events: Optional[Iterable[dict]],
    future_index: pd.DatetimeIndex,
) -> pd.Series:
    """프로모션 이벤트로부터 일별 uplift Series를 계산합니다.

    Args:
        promotion_events: 프로모션 이벤트 목록
        future_index: 예측 기간 DatetimeIndex

    Returns:
        pd.Series: 일별 uplift multiplier (기본값 1.0)
    """
    uplift = pd.Series(1.0, index=future_index, dtype=float)

    if not promotion_events or uplift.empty:
        return uplift

    for event in promotion_events:
        start_evt = pd.to_datetime(event.get("start"), errors="coerce")
        end_evt = pd.to_datetime(event.get("end"), errors="coerce")
        uplift_val = float(event.get("uplift", 0.0))
        uplift_val = min(3.0, max(-1.0, uplift_val))
        if pd.notna(start_evt) and pd.notna(end_evt):
            s_norm = max(start_evt.normalize(), future_index[0])
            e_norm = min(end_evt.normalize(), future_index[-1])
            if s_norm <= e_norm:
                uplift.loc[s_norm:e_norm] = uplift.loc[s_norm:e_norm] * (1.0 + uplift_val)

    return uplift


def generate_sales_forecasts(
    sales_hist: pd.DataFrame,
    start_norm: pd.Timestamp,
    today_norm: pd.Timestamp,
    future_index: pd.DatetimeIndex,
    uplift: pd.Series,
    inbound_lookup: dict[tuple[str, str], pd.Series],
    latest_stock_lookup: dict[tuple[str, str], float],
    empty_sales: pd.DataFrame,
    make_forecast_sales_capped_fn,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """판매 이력으로부터 MA7과 예측을 생성합니다.

    Args:
        sales_hist: 판매 이력 데이터
        start_norm: 시작일
        today_norm: 오늘 날짜
        future_index: 예측 기간 DatetimeIndex
        uplift: 프로모션 uplift Series
        inbound_lookup: 입고 예정 lookup
        latest_stock_lookup: 최근 재고 lookup
        empty_sales: 빈 판매 DataFrame 템플릿
        make_forecast_sales_capped_fn: 재고 제약 적용 함수

    Returns:
        tuple: (sales_ma7, sales_forecast) DataFrames
    """
    sales_ma7_frames: list[pd.DataFrame] = []
    sales_fc_frames: list[pd.DataFrame] = []

    hist_index = pd.date_range(start_norm, today_norm, freq="D")

    for (center, sku), group in sales_hist.groupby(["center", "resource_code"], dropna=True):
        series = (
            group.sort_values("date")
            .set_index("date")["sales_ea"]
            .reindex(hist_index, fill_value=0.0)
        )
        ma7 = series.rolling(7, min_periods=1).mean()
        ma_df = (
            ma7.to_frame("sales_ea")
            .reset_index()
            .rename(columns={"index": "date"})
        )
        ma_df["center"] = center
        ma_df["resource_code"] = sku
        sales_ma7_frames.append(ma_df)

        if not future_index.empty:
            base_rate = float(ma7.iloc[-1]) if not ma7.empty else 0.0
            fc_values = pd.Series(base_rate, index=future_index, dtype=float)
            if not uplift.empty:
                fc_values = fc_values * uplift.reindex(future_index, fill_value=1.0)
            inbound_series = inbound_lookup.get((center, sku))
            latest_stock = latest_stock_lookup.get((center, sku), 0.0)
            fc_values = make_forecast_sales_capped_fn(
                base_daily_pred=fc_values,
                latest_stock=latest_stock,
                inbound_by_day=inbound_series,
            )
            fc_df = (
                fc_values.to_frame("sales_ea")
                .reset_index()
                .rename(columns={"index": "date"})
            )
            fc_df["center"] = center
            fc_df["resource_code"] = sku
            sales_fc_frames.append(fc_df)

    sales_ma7 = (
        pd.concat(sales_ma7_frames, ignore_index=True)
        if sales_ma7_frames
        else empty_sales.copy()
    )
    sales_ma7["date"] = pd.to_datetime(sales_ma7.get("date"), errors="coerce").dt.normalize()

    sales_forecast = (
        pd.concat(sales_fc_frames, ignore_index=True)
        if sales_fc_frames
        else empty_sales.copy()
    )
    sales_forecast["date"] = pd.to_datetime(sales_forecast.get("date"), errors="coerce").dt.normalize()

    return sales_ma7, sales_forecast


def apply_stock_depletion(
    sales_forecast: pd.DataFrame,
    latest_stock_lookup: dict[tuple[str, str], float],
    inbound_lookup: dict[tuple[str, str], pd.Series],
) -> pd.DataFrame:
    """재고 소진 날짜를 계산하고 판매 예측에 적용합니다.

    Args:
        sales_forecast: 판매 예측 DataFrame
        latest_stock_lookup: 최근 재고 lookup
        inbound_lookup: 입고 예정 lookup

    Returns:
        pd.DataFrame: 재고 소진이 적용된 판매 예측
    """
    if sales_forecast.empty:
        return sales_forecast

    depletion_dates: dict[tuple[str, str], pd.Timestamp] = {}
    tol = 1e-6

    for (center, sku), grp in sales_forecast.groupby([
        "center",
        "resource_code",
    ], dropna=True):
        fc_series = (
            grp.sort_values("date")
            .set_index("date")["sales_ea"]
            .astype(float)
        )
        if fc_series.empty:
            continue

        latest_stock = float(latest_stock_lookup.get((center, sku), 0.0))
        inbound_series = inbound_lookup.get((center, sku))
        if inbound_series is None:
            inbound_series = pd.Series(0.0, index=fc_series.index, dtype=float)
        else:
            inbound_series = inbound_series.reindex(fc_series.index, fill_value=0.0).astype(float)

        remain = max(latest_stock, 0.0)
        remain_before: list[float] = []
        sale_list: list[float] = []
        for day in fc_series.index:
            remain += float(inbound_series.loc[day])
            available = max(remain, 0.0)
            remain_before.append(available)
            want = float(fc_series.loc[day])
            sale = min(want, available)
            remain -= sale
            sale_list.append(sale)

        if sale_list:
            sale_array = np.asarray(sale_list, dtype=float)
            before_array = np.asarray(remain_before, dtype=float)
            if ((sale_array <= tol) & (before_array <= tol)).any():
                suffix_before = np.maximum.accumulate(before_array[::-1])[::-1]
                suffix_sale = np.maximum.accumulate(sale_array[::-1])[::-1]
                final_mask = (
                    (sale_array <= tol)
                    & (before_array <= tol)
                    & (suffix_before <= tol)
                    & (suffix_sale <= tol)
                )
                if final_mask.any():
                    depletion_dates[(center, sku)] = fc_series.index[int(np.argmax(final_mask))]

    if depletion_dates:
        mask_center = sales_forecast[["center", "resource_code"]].apply(tuple, axis=1)
        for key, dep_date in depletion_dates.items():
            clip_mask = (mask_center == key) & (sales_forecast["date"] > dep_date)
            if clip_mask.any():
                sales_forecast.loc[clip_mask, "sales_ea"] = 0.0

    return sales_forecast
