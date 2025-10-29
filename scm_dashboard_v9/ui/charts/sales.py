"""판매 데이터 계산 모듈.

스냅샷 기반 판매량 계산, 예측 생성 등의 로직을 제공합니다.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from center_alias import normalize_center_value
from scm_dashboard_v9.domain.filters import filter_by_centers, safe_to_datetime

from .data_utils import coerce_cols, empty_sales_frame


def sales_from_snapshot(
    snap_long: pd.DataFrame,
    centers: List[str],
    skus: List[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """
    스냅샷의 일간 차분으로 '판매(실측)'만 계산.
    - 증가분(+)은 입고로 보고 판매에서 제외
    - 감소분(-)만 판매로 본다
    반환: index=date, columns=sku, 값=EA/Day
    """
    c = coerce_cols(snap_long)
    s = snap_long.rename(
        columns={
            c["date"]: "date",
            c["center"]: "center",
            c["sku"]: "resource_code",
            c["qty"]: "stock_qty",
        }
    )[["date", "center", "resource_code", "stock_qty"]].copy()

    s["date"] = safe_to_datetime(s["date"])
    s = s[
        s["center"].astype(str).isin(centers)
        & s["resource_code"].astype(str).isin(skus)
    ]
    if s.empty:
        idx = pd.date_range(start, end, freq="D")
        return pd.DataFrame(0, index=idx, columns=skus)

    pv = (
        s.groupby(["date", "resource_code"])["stock_qty"]
        .sum()
        .unstack("resource_code")
        .reindex(columns=skus, fill_value=0)
        .sort_index()
    )
    pv = pv.asfreq("D").ffill()  # D 간격 보정
    d = pv.diff().fillna(0)
    sales = (-d).clip(lower=0)  # 감소분만 판매
    sales = sales.loc[(sales.index >= start) & (sales.index <= end)]
    return sales


def sales_forecast_ma(
    sales_daily: pd.DataFrame,
    *,
    sku: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    lookback_days: int = 28,
    promo_multiplier: float = 1.0,
    value_column: str = "qty_sold",
) -> pd.DataFrame:
    """Return a constant daily forecast using a guarded moving-average base.

    ``sales_daily`` is expected to contain at least ``date``, ``resource_code``
    and the ``value_column`` (defaults to ``qty_sold``).  The function mirrors
    the defensive logic described in the user guidance: the centre names must
    already be normalised and the forecast should survive even when only a few
    historical points are available.
    """

    if sales_daily is None or sales_daily.empty:
        return pd.DataFrame(columns=["date", "resource_code", "qty_pred"])

    start_norm = pd.to_datetime(start).normalize()
    end_norm = pd.to_datetime(end).normalize()
    if pd.isna(start_norm) or pd.isna(end_norm) or end_norm < start_norm:
        return pd.DataFrame(columns=["date", "resource_code", "qty_pred"])

    sku_str = str(sku)
    df = sales_daily.copy()
    df["resource_code"] = df.get("resource_code", "").astype(str)
    df = df[df["resource_code"] == sku_str]
    if df.empty:
        return pd.DataFrame(columns=["date", "resource_code", "qty_pred"])

    df["date"] = pd.to_datetime(df.get("date"), errors="coerce").dt.normalize()
    df = df.dropna(subset=["date"])
    if df.empty:
        return pd.DataFrame(columns=["date", "resource_code", "qty_pred"])

    values = pd.to_numeric(df.get(value_column), errors="coerce").fillna(0.0)
    df[value_column] = values

    last_hist_day = start_norm - pd.Timedelta(days=1)
    history = df[df["date"] <= last_hist_day]
    if history.empty:
        history = df

    lookback = max(1, int(lookback_days))
    history = history.sort_values("date").tail(lookback)
    if history.empty:
        return pd.DataFrame(columns=["date", "resource_code", "qty_pred"])

    series = history.set_index("date")[value_column].asfreq("D").fillna(0.0)
    if series.empty:
        return pd.DataFrame(columns=["date", "resource_code", "qty_pred"])

    ma7 = series.rolling(7, min_periods=1).mean()
    if not ma7.empty:
        base = float(ma7.iloc[-1])
    else:
        base = float(series.mean()) if not series.empty else 0.0

    if not np.isfinite(base):
        base = 0.0

    multiplier = float(promo_multiplier) if np.isfinite(promo_multiplier) else 1.0
    base = max(0.0, base * multiplier)
    has_positive_history = bool((series > 0).any())
    qty = int(round(base))
    if qty <= 0 and has_positive_history:
        qty = 1
    elif qty < 0:
        qty = 0

    index = pd.date_range(start_norm, end_norm, freq="D")
    if index.empty:
        return pd.DataFrame(columns=["date", "resource_code", "qty_pred"])

    return pd.DataFrame(
        {
            "date": index,
            "resource_code": sku_str,
            "qty_pred": qty,
        }
    )


def sales_from_snapshot_raw(
    snap_raw: Optional[pd.DataFrame],
    centers: Sequence[str],
    skus: Sequence[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    debug: Optional[dict[str, object]] = None,
) -> pd.DataFrame:
    """원시 스냅샷 데이터로부터 매출 데이터를 추출하고 변환합니다.

    이 함수는 다국어 컬럼명을 지원하는 강건한 데이터 파이프라인으로:
    1. 컬럼명 정규화 (한글/영문 다국어 지원)
    2. 센터명 정규화 (normalize_center_value 적용)
    3. 센터 및 SKU 필터링
    4. 기간 필터링 및 유효성 검증
    5. 일자별 매출 집계 (중복 제거)

    Args:
        snap_raw: 원시 스냅샷 DataFrame (다양한 컬럼명 형식 지원)
            - 필수 컬럼: date/snapshot_date/스냅샷일자 중 하나
            - 필수 컬럼: fba_output_stock/출고수량/fba출고 중 하나
            - 선택 컬럼: center/센터/warehouse, resource_code/sku/상품코드
        centers: 대상 센터 리스트 (정규화 전 값)
        skus: 대상 SKU 리스트
        start: 조회 시작 날짜
        end: 조회 종료 날짜
        debug: 디버그 정보 수집 딕셔너리 (선택)
            - 'rows_before_center_filter': 센터 필터 전 행 수
            - 'rows_after_center_filter': 센터 필터 후 행 수
            - 'snapshot_centers': 스냅샷에 포함된 센터 리스트
            - 'warning': 오류 메시지 (있을 경우)

    Returns:
        일자별 SKU 매출 DataFrame
        - 컬럼: date, resource_code, sales_ea
        - sales_ea: 0 이상 정수 (음수 제거됨)
        - 빈 결과 시: 동일 스키마의 빈 DataFrame

    Notes:
        - 센터 컬럼이 없으면 첫 번째 요청 센터를 기본값으로 사용 (또는 AMZUS)
        - 필수 컬럼 누락 시 경고와 함께 빈 DataFrame 반환
        - 중복 데이터는 합계로 집계됨

    Examples:
        >>> debug_info = {}
        >>> sales_df = sales_from_snapshot_raw(
        ...     snap_raw=raw_snapshot,
        ...     centers=["AMZUS", "AMZEU"],
        ...     skus=["SKU001", "SKU002"],
        ...     start=pd.Timestamp("2024-01-01"),
        ...     end=pd.Timestamp("2024-01-31"),
        ...     debug=debug_info,
        ... )
        >>> print(debug_info['snapshot_centers'])
        ['AMZEU', 'AMZUS']
    """
    if snap_raw is None or snap_raw.empty:
        if debug is not None:
            debug.clear()
            debug.update(
                {
                    "rows_before_center_filter": 0,
                    "rows_after_center_filter": 0,
                    "snapshot_centers": [],
                }
            )
        return empty_sales_frame()

    df = snap_raw.copy()

    rename_map: dict[str, str] = {}
    date_candidates = {"snapshot_date", "date", "스냅샷일자", "스냅샷 일자", "스냅샷일"}
    center_candidates = {"center", "센터", "창고", "창고명", "warehouse"}
    sku_candidates = {"resource_code", "sku", "상품코드", "product_code"}
    output_candidates = {
        "fba_output_stock",
        "fba출고",
        "출고수량",
        "출고",
        "fba_output",
        "출고 ea",
    }

    for col in df.columns:
        key = str(col).strip().lower()
        if key in date_candidates:
            rename_map[col] = "date"
        elif key in center_candidates:
            rename_map[col] = "center"
        elif key in sku_candidates:
            rename_map[col] = "resource_code"
        elif key in output_candidates or "fba_output_stock" in key:
            rename_map[col] = "fba_output_stock"

    df = df.rename(columns=rename_map)

    if "fba_output_stock" not in df.columns:
        if debug is not None:
            centers_for_debug: list[str] = []
            if "center" in df.columns:
                centers_for_debug = df["center"].dropna().astype(str).unique().tolist()
            debug.clear()
            debug.update(
                {
                    "rows_before_center_filter": len(df),
                    "rows_after_center_filter": len(df),
                    "snapshot_centers": sorted(centers_for_debug),
                    "warning": "missing fba_output_stock column",
                }
            )
        return empty_sales_frame()

    if "date" not in df.columns:
        centers_for_debug = []
        if "center" in df.columns:
            centers_for_debug = df["center"].dropna().astype(str).unique().tolist()
        if debug is not None:
            debug.clear()
            debug.update(
                {
                    "rows_before_center_filter": len(df),
                    "rows_after_center_filter": len(df),
                    "snapshot_centers": sorted(centers_for_debug),
                    "warning": "missing date column",
                }
            )
        return empty_sales_frame()

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["date"])
    df["fba_output_stock"] = pd.to_numeric(
        df.get("fba_output_stock"), errors="coerce"
    ).fillna(0)

    rows_before_center_filter = len(df)
    available_centers: list[str] = []

    if "center" in df.columns:
        df["center"] = df["center"].apply(normalize_center_value)
        df = df[df["center"].notna()]
        available_centers = (
            df["center"].dropna().astype(str).unique().tolist() if not df.empty else []
        )
    else:
        # If center is absent in snapshot_raw, infer a single Amazon centre to keep
        # the downstream filters working. Use the first requested centre, default AMZUS.
        inferred = None
        for ct in centers:
            norm = normalize_center_value(ct)
            if norm:
                inferred = norm
                break
        if inferred is None:
            inferred = "AMZUS"
        df["center"] = inferred
        available_centers = [inferred]

    if "center" in df.columns and centers:
        centers_norm = {
            normalized
            for c in centers
            for normalized in [normalize_center_value(c)]
            if normalized is not None
        }
        if centers_norm:
            df = df[df["center"].isin(centers_norm)]

    rows_after_center_filter = len(df)

    if debug is not None:
        debug.clear()
        debug.update(
            {
                "rows_before_center_filter": rows_before_center_filter,
                "rows_after_center_filter": rows_after_center_filter,
                "snapshot_centers": sorted(available_centers),
            }
        )

    if "resource_code" not in df.columns:
        return empty_sales_frame()

    if skus:
        sku_set = {str(s) for s in skus if str(s).strip()}
        if sku_set:
            df = df[df["resource_code"].astype(str).isin(sku_set)]

    if df.empty:
        return empty_sales_frame()

    start_norm = pd.to_datetime(start).normalize()
    end_norm = pd.to_datetime(end).normalize()

    df = df[(df["date"] >= start_norm) & (df["date"] <= end_norm)]
    if df.empty:
        return empty_sales_frame()

    grouped = (
        df.groupby(["date", "resource_code"], as_index=False)["fba_output_stock"]
        .sum()
        .rename(columns={"fba_output_stock": "sales_ea"})
    )

    grouped["sales_ea"] = (
        pd.to_numeric(grouped["sales_ea"], errors="coerce")
        .fillna(0)
        .clip(lower=0)
        .round()
        .astype(int)
    )
    return grouped


def sales_forecast_from_inventory_projection(
    inv_actual: pd.DataFrame,
    inv_forecast: pd.DataFrame,
    *,
    centers: Sequence[str],
    skus: Sequence[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    today: pd.Timestamp,
) -> pd.DataFrame:
    """Derive future sales from the projected inventory trajectory.

    ``apply_consumption_with_events`` already produces the most accurate stock
    projection for Amazon centres by blending actual inventory, inbound moves
    and the calibrated consumption trend.  Sales should therefore mirror the
    day-on-day decrease of that stock projection so that both visuals stay in
    sync even when promotions or inbound events shift the depletion curve.
    """

    today_norm = pd.to_datetime(today).normalize()

    frames: list[pd.DataFrame] = []
    for label, frame in (("actual", inv_actual), ("forecast", inv_forecast)):
        if frame is None or frame.empty:
            continue
        if not {"date", "center", "resource_code", "stock_qty"}.issubset(frame.columns):
            continue
        chunk = frame.copy()
        chunk["date"] = pd.to_datetime(
            chunk.get("date"), errors="coerce"
        ).dt.normalize()
        chunk = chunk.dropna(subset=["date"])
        if label == "actual":
            chunk = chunk[chunk["date"] <= today_norm]
        else:
            chunk = chunk[chunk["date"] > today_norm]
        if chunk.empty:
            continue
        chunk["center"] = chunk.get("center", "").apply(normalize_center_value)
        chunk = chunk[chunk["center"].notna()]
        chunk["resource_code"] = chunk.get("resource_code", "").astype(str).str.strip()
        chunk["stock_qty"] = pd.to_numeric(
            chunk.get("stock_qty"), errors="coerce"
        ).fillna(0.0)
        chunk["__source"] = label
        frames.append(chunk)

    if not frames:
        return pd.DataFrame(columns=["date", "resource_code", "sales_ea"])

    combined = pd.concat(frames, ignore_index=True)

    if "__source" in combined.columns:
        combined["__priority"] = (
            combined["__source"].map({"actual": 0, "forecast": 1}).fillna(1)
        )
        combined = (
            combined.sort_values(["date", "resource_code", "center", "__priority"])
            .drop_duplicates(subset=["date", "resource_code", "center"], keep="first")
            .drop(columns=["__source", "__priority"], errors="ignore")
        )

    centers_norm = [normalize_center_value(c) for c in centers]
    centers_norm = [c for c in centers_norm if c]
    skus_norm = [str(sku).strip() for sku in skus if str(sku).strip()]

    if centers_norm:
        combined = combined[combined["center"].isin(set(centers_norm))]
    if skus_norm:
        combined = combined[combined["resource_code"].isin(set(skus_norm))]

    if combined.empty:
        return pd.DataFrame(columns=["date", "resource_code", "sales_ea"])

    start_norm = pd.to_datetime(start).normalize()
    end_norm = pd.to_datetime(end).normalize()
    combined = combined[
        (combined["date"] >= start_norm) & (combined["date"] <= end_norm)
    ]
    if combined.empty:
        return pd.DataFrame(columns=["date", "resource_code", "sales_ea"])

    pivot = (
        combined.groupby(["date", "resource_code"])["stock_qty"]
        .sum()
        .unstack("resource_code")
    )
    if pivot.empty:
        return pd.DataFrame(columns=["date", "resource_code", "sales_ea"])

    pivot = pivot.reindex(columns=skus_norm, fill_value=0.0)
    full_index = pd.date_range(start_norm, end_norm, freq="D")
    pivot = pivot.reindex(full_index).sort_index()
    pivot = pivot.ffill().fillna(0.0)

    diff = pivot.diff()
    sales = (-diff).clip(lower=0.0)

    # 재고가 증가한 날(입고가 있는 날)에도 실제로는 판매가 발생합니다.
    # 하지만 재고 변화(diff)로는 정확히 계산할 수 없습니다:
    #   예: 재고 100 → 입고 50, 판매 30 → 재고 120
    #   diff = +20 → sales = 0 (하지만 실제로는 30 판매)
    # 해결: 입고 날짜를 제외한 평균 일 판매량을 입고 날짜에 적용
    inbound_mask = diff > 0

    # 입고 날짜를 제외하고 평균 계산 (입고 횟수에 비례하지 않도록)
    sales_no_inbound = sales.copy()
    for sku in sales.columns:
        if sku in inbound_mask.columns and inbound_mask[sku].any():
            sales_no_inbound.loc[inbound_mask[sku], sku] = np.nan

    avg_sales = sales_no_inbound.mean(skipna=True)
    avg_sales = avg_sales.where(np.isfinite(avg_sales), 0.0)

    for sku in sales.columns:
        if sku in inbound_mask.columns and inbound_mask[sku].any():
            sales.loc[inbound_mask[sku], sku] = avg_sales[sku]

    # 재고가 0인 날짜에만 판매를 0으로 설정.
    # 이전 로직은 첫 번째 0 이후 모든 판매를 차단했지만,
    # 재고가 다시 양수가 되면 판매도 재개되어야 합니다.
    for sku in sales.columns:
        stock_series = pivot[sku]
        # 미래 기간에서만 체크
        future_stock = stock_series.loc[stock_series.index > today_norm]
        zero_mask = future_stock <= 0

        if zero_mask.any():
            zero_dates = future_stock.index[zero_mask]
            # 재고가 0인 날짜에만 판매를 0으로 설정
            sales.loc[zero_dates, sku] = 0.0

    future = sales.loc[sales.index > today_norm]
    if future.empty:
        return pd.DataFrame(columns=["date", "resource_code", "sales_ea"])

    tidy = (
        future.round(0)
        .astype(int)
        .stack()
        .reset_index()
        .rename(columns={"level_0": "date", "level_1": "resource_code", 0: "sales_ea"})
    )
    tidy["date"] = pd.to_datetime(tidy["date"], errors="coerce").dt.normalize()
    tidy = tidy.dropna(subset=["date"])
    tidy["sales_ea"] = tidy["sales_ea"].clip(lower=0)
    return tidy.sort_values(["resource_code", "date"]).reset_index(drop=True)


def sales_forecast_from_actual_sales_with_stock_limit(
    sales_actual: pd.DataFrame,
    inv_forecast: pd.DataFrame,
    *,
    skus: Sequence[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    today: pd.Timestamp,
    lookback_days: int = 28,
) -> pd.DataFrame:
    """과거 실제 판매량(sales_qty)으로 평균을 계산하고 미래에 적용.
    재고가 0인 날짜는 판매도 0으로 설정하여 재고-판매 시각적 일치.

    Args:
        sales_actual: 과거 실제 판매 데이터 (columns: date, resource_code, sales_qty)
        inv_forecast: 미래 재고 예측 (columns: date, resource_code, stock_qty)
        skus: SKU 목록
        start: 예측 시작일
        end: 예측 종료일
        today: 오늘 날짜 (과거/미래 구분)
        lookback_days: 평균 계산 시 조회할 과거 일수

    Returns:
        DataFrame: 미래 판매 예측 (columns: date, resource_code, sales_qty)
    """
    # DEBUG: Streamlit 런타임 체크
    debug_enabled = False
    try:
        import streamlit as st
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        if get_script_run_ctx() is not None:
            debug_enabled = True
            st.write(
                "\n**🔍 [sales_forecast_from_actual_sales_with_stock_limit] 과거 판매 기반 예측:**"
            )
    except (ImportError, RuntimeError):
        pass

    today_norm = pd.to_datetime(today).normalize()
    start_norm = pd.to_datetime(start).normalize()
    end_norm = pd.to_datetime(end).normalize()

    # 1. 과거 sales_qty로 평균 계산 (정확한 데이터 사용)
    if sales_actual is None or sales_actual.empty:
        avg_sales_by_sku = {sku: 0.0 for sku in skus}
        if debug_enabled:
            st.warning("  ⚠️ sales_actual이 비어있음 → 평균 판매량 0으로 설정")
    else:
        lookback_start = today_norm - pd.Timedelta(days=lookback_days)
        recent_sales = sales_actual[
            (sales_actual["date"] >= lookback_start)
            & (sales_actual["date"] <= today_norm)
        ].copy()

        avg_sales_by_sku = {}
        for sku in skus:
            sku_sales = recent_sales[recent_sales["resource_code"] == sku]["sales_qty"]
            avg = float(sku_sales.mean()) if not sku_sales.empty else 0.0
            avg_sales_by_sku[sku] = max(0.0, avg)

        if debug_enabled:
            st.write(f"  - lookback 기간: {lookback_start} ~ {today_norm}")
            st.write(f"  - 과거 판매 데이터: {len(recent_sales)} 행")
            st.write(f"  - **평균 일 판매량 (SKU별):** {avg_sales_by_sku}")

    # 2. 미래 날짜에 평균 적용 (단순하고 명확)
    future_start = today_norm + pd.Timedelta(days=1)
    if future_start > end_norm:
        return pd.DataFrame(columns=["date", "resource_code", "sales_qty"])

    future_dates = pd.date_range(future_start, end_norm, freq="D")
    forecast_rows = []
    for sku in skus:
        for date in future_dates:
            forecast_rows.append(
                {
                    "date": date,
                    "resource_code": sku,
                    "sales_qty": avg_sales_by_sku[sku],
                }
            )

    sales_forecast = pd.DataFrame(forecast_rows)

    if debug_enabled:
        st.write(f"\n  - 미래 기간: {future_start} ~ {end_norm}")
        st.write(f"  - 초기 판매 예측: {len(sales_forecast)} 행")
        st.write(
            f"  - 판매 합계 (SKU별): {sales_forecast.groupby('resource_code')['sales_qty'].sum().to_dict()}"
        )

    if sales_forecast.empty:
        return pd.DataFrame(columns=["date", "resource_code", "sales_qty"])

    # 3. 재고가 0인 날짜는 판매도 0으로 설정 (시각적 일치)
    if inv_forecast is not None and not inv_forecast.empty:
        inv_forecast = inv_forecast.copy()
        inv_forecast["date"] = pd.to_datetime(inv_forecast["date"]).dt.normalize()

        if debug_enabled:
            st.write(f"\n  **재고 기반 판매 제약:**")

        for sku in skus:
            sku_inv = inv_forecast[inv_forecast["resource_code"] == sku]
            zero_dates = sku_inv[sku_inv["stock_qty"] <= 0]["date"].values

            if len(zero_dates) > 0:
                mask = (sales_forecast["resource_code"] == sku) & (
                    sales_forecast["date"].isin(zero_dates)
                )
                sales_forecast.loc[mask, "sales_qty"] = 0.0

                if debug_enabled:
                    st.write(
                        f"  - {sku}: 재고 0인 날짜 {len(zero_dates)}개 → 해당 날짜 판매량 0으로 설정"
                    )
            elif debug_enabled:
                st.write(f"  - {sku}: 재고 0 없음 → 평균 판매량 유지")

    if debug_enabled:
        st.write(f"\n  **최종 판매 예측:**")
        st.write(
            f"  - 판매 합계 (SKU별): {sales_forecast.groupby('resource_code')['sales_qty'].sum().to_dict()}"
        )
        st.write("  - 샘플 (마지막 10행):")
        st.dataframe(sales_forecast.tail(10))

    return sales_forecast


def sales_from_snapshot_decays(
    snap_like: Optional[pd.DataFrame],
    centers: Sequence[str],
    skus: Sequence[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    if snap_like is None or snap_like.empty:
        return empty_sales_frame()

    centers_list = [str(c) for c in centers if str(c).strip()]
    skus_list = [str(s) for s in skus if str(s).strip()]
    if not centers_list or not skus_list:
        return empty_sales_frame()

    matrix = sales_from_snapshot(
        snap_like,
        centers=centers_list,
        skus=skus_list,
        start=start,
        end=end,
    )

    if matrix is None or matrix.empty:
        return empty_sales_frame()

    tidy = matrix.reset_index().melt(
        id_vars="date", var_name="resource_code", value_name="sales_ea"
    )

    tidy["date"] = pd.to_datetime(tidy.get("date"), errors="coerce").dt.normalize()
    tidy = tidy.dropna(subset=["date"])
    tidy["resource_code"] = tidy["resource_code"].astype(str)
    tidy = tidy[tidy["resource_code"].isin(skus_list)]

    start_norm = pd.to_datetime(start).normalize()
    end_norm = pd.to_datetime(end).normalize()
    tidy = tidy[(tidy["date"] >= start_norm) & (tidy["date"] <= end_norm)]

    tidy["sales_ea"] = (
        pd.to_numeric(tidy.get("sales_ea"), errors="coerce")
        .fillna(0)
        .clip(lower=0)
        .round()
        .astype(int)
    )

    tidy = tidy[tidy["sales_ea"] >= 0]
    tidy = tidy.sort_values(["date", "resource_code"]).reset_index(drop=True)
    return tidy
