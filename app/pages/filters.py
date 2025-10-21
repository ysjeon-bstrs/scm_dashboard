from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import pandas as pd
import streamlit as st


def _norm_center(value: str) -> Optional[str]:
    from center_alias import normalize_center_value

    return normalize_center_value(value)


def derive_center_and_sku_options(
    moves: pd.DataFrame, snapshot: pd.DataFrame
) -> Tuple[list[str], list[str]]:
    """Derive selectable centers and SKUs from moves and snapshot frames."""

    snap_centers_src = snapshot.get("center")
    if snap_centers_src is None:
        snap_centers = pd.Series(dtype=str)
    else:
        snap_centers = snap_centers_src.dropna().astype(str).str.strip()
    move_centers = pd.concat(
        [
            moves.get("from_center", pd.Series(dtype=object)),
            moves.get("to_center", pd.Series(dtype=object)),
        ],
        ignore_index=True,
    ).dropna().astype(str).str.strip()

    all_candidates = pd.concat([snap_centers, move_centers], ignore_index=True).dropna()
    centers = sorted({c for c in (_norm_center(value) for value in all_candidates) if c})

    skus = sorted(snapshot.get("resource_code", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())
    if not skus:
        skus = sorted(moves.get("resource_code", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())

    return centers, skus


def _move_date_bounds(moves: pd.DataFrame) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    date_columns = [
        col
        for col in ("onboard_date", "arrival_date", "inbound_date", "event_date")
        if col in moves.columns
    ]
    if not date_columns:
        return None, None

    normalized_dates = []
    for col in date_columns:
        series = pd.to_datetime(moves[col], errors="coerce").dropna()
        if not series.empty:
            normalized_dates.append(series.dt.normalize())

    if not normalized_dates:
        return None, None

    combined = pd.concat(normalized_dates, ignore_index=True)
    if combined.empty:
        return None, None

    min_dt = combined.min()
    max_dt = combined.max()
    if pd.isna(min_dt) or pd.isna(max_dt):
        return None, None

    return pd.Timestamp(min_dt).normalize(), pd.Timestamp(max_dt).normalize()


def calculate_date_bounds(
    *,
    today: pd.Timestamp,
    snapshot_df: pd.DataFrame,
    moves_df: pd.DataFrame,
    base_past_days: int,
    base_future_days: int,
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Compute the selectable date range for the dashboard period slider."""

    normalized_today = pd.Timestamp(today).normalize()
    base_min = (normalized_today - pd.Timedelta(days=base_past_days)).normalize()
    base_max = (normalized_today + pd.Timedelta(days=base_future_days)).normalize()

    bound_min_candidates = [base_min]
    bound_max_candidates = [base_max]

    snap_dates = pd.to_datetime(snapshot_df.get("date"), errors="coerce").dropna()
    if not snap_dates.empty:
        bound_min_candidates.append(snap_dates.min().normalize())
        bound_max_candidates.append(snap_dates.max().normalize())

    move_min, move_max = _move_date_bounds(moves_df)
    if move_min is not None:
        bound_min_candidates.append(move_min)
    if move_max is not None:
        bound_max_candidates.append(move_max)

    dynamic_min = min(bound_min_candidates)
    dynamic_max = max(bound_max_candidates)

    bound_min = max(dynamic_min, base_min)
    bound_max = min(dynamic_max, base_max)

    if bound_min > bound_max:
        bound_min = bound_max

    return bound_min.normalize(), bound_max.normalize()


@dataclass
class FilterControls:
    centers: list[str]
    skus: list[str]
    start: pd.Timestamp
    end: pd.Timestamp
    show_production: bool
    show_in_transit: bool
    use_consumption_forecast: bool
    lookback_days: int
    events: list[dict[str, object]]
    lag_days: int


def _clamp_range(
    range_value: Tuple[pd.Timestamp, pd.Timestamp],
    *,
    bound_min: pd.Timestamp,
    bound_max: pd.Timestamp,
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    start_val, end_val = range_value
    start_val = pd.Timestamp(start_val).normalize()
    end_val = pd.Timestamp(end_val).normalize()
    start_val = max(min(start_val, bound_max), bound_min)
    end_val = max(min(end_val, bound_max), bound_min)
    if end_val < start_val:
        end_val = start_val
    return start_val, end_val


def _build_events() -> list[dict[str, object]]:
    lookback_days = int(st.session_state.get("trend_lookback_days", 28))
    promo_on = bool(st.session_state.get("promo_enabled", False))
    promo_start = st.session_state.get("promo_start")
    promo_end = st.session_state.get("promo_end")
    promo_uplift = float(st.session_state.get("promo_uplift_pct", 0.0)) / 100.0

    events: list[dict[str, object]] = []
    if promo_on and promo_start and promo_end and promo_uplift != 0.0:
        events.append(
            {
                "start": pd.to_datetime(promo_start),
                "end": pd.to_datetime(promo_end),
                "uplift": promo_uplift,
            }
        )
    return events


def render_sidebar_controls(
    *,
    centers: Sequence[str],
    skus: Sequence[str],
    bound_min: pd.Timestamp,
    bound_max: pd.Timestamp,
    default_centers: Optional[Iterable[str]] = None,
    default_skus: Optional[Iterable[str]] = None,
    default_past_days: int = 20,
    default_future_days: int = 30,
) -> Optional[FilterControls]:
    if not centers:
        st.warning("센터 또는 SKU 정보를 찾을 수 없습니다.")
        return None
    if not skus:
        st.warning("센터 또는 SKU 정보를 찾을 수 없습니다.")
        return None

    today = pd.Timestamp.today().normalize()

    default_centers = list(default_centers or [])
    default_skus = list(default_skus or [])

    preset_centers = ["태광KR", "AMZUS"]
    preset_skus = ["BA00021", "BA00022"]

    with st.sidebar:
        st.header("필터")
        st.caption("기본값: 센터 태광KR·AMZUS / SKU BA00021·BA00022 / 기간 오늘−20일 ~ +30일.")

        centers_default = [c for c in preset_centers if c in centers]
        if not centers_default:
            centers_default = list(default_centers) if default_centers else list(centers)
        selected_centers = st.multiselect("센터", list(centers), default=centers_default)

        skus_default = [s for s in preset_skus if s in skus]
        if not skus_default:
            if default_skus:
                skus_default = list(default_skus)
            else:
                skus_default = list(skus if len(skus) <= 10 else skus[:10])
        selected_skus = st.multiselect("SKU", list(skus), default=skus_default)

        st.subheader("기간 설정")

        if "date_range" not in st.session_state:
            default_start = max(today - pd.Timedelta(days=default_past_days), bound_min)
            default_end = min(today + pd.Timedelta(days=default_future_days), bound_max)
            if default_start > default_end:
                default_start = default_end
            st.session_state.date_range = (default_start, default_end)
        else:
            st.session_state.date_range = _clamp_range(
                tuple(st.session_state.date_range), bound_min=bound_min, bound_max=bound_max
            )

        date_range_value = st.slider(
            "기간",
            min_value=bound_min.to_pydatetime(),
            max_value=bound_max.to_pydatetime(),
            value=tuple(d.to_pydatetime() for d in st.session_state.date_range),
            format="YYYY-MM-DD",
        )
        start_ts = pd.Timestamp(date_range_value[0]).normalize()
        end_ts = pd.Timestamp(date_range_value[1]).normalize()
        st.session_state.date_range = (start_ts, end_ts)

        st.divider()
        st.header("표시 옵션")
        show_prod = st.checkbox("생산중 표시", value=False)
        show_transit = False
        st.caption("체크 시 계단식 차트에 생산중 라인이 표시됩니다.")
        use_cons_forecast = st.checkbox("추세 기반 재고 예측", value=True)

        st.subheader("추세 계산 설정")
        lookback_days = int(
            st.number_input(
                "추세 계산 기간(일)",
                min_value=7,
                max_value=56,
                value=28,
                step=7,
                key="trend_lookback_days",
            )
        )
        with st.expander("프로모션 가중치(+%)", expanded=False):
            st.checkbox("가중치 적용", value=False, key="promo_enabled")
            st.date_input("시작일", key="promo_start")
            st.date_input("종료일", key="promo_end")
            st.number_input(
                "가중치(%)",
                min_value=-100.0,
                max_value=300.0,
                value=30.0,
                step=5.0,
                key="promo_uplift_pct",
            )

        st.divider()
        st.header("입고 반영 가정")
        lag_days = int(
            st.number_input(
                "입고 반영 리드타임(일) – inbound 미기록 시 arrival+N",
                min_value=0,
                max_value=21,
                value=5,
                step=1,
            )
        )

    selected_centers = [str(center) for center in selected_centers if str(center).strip()]
    selected_skus = [str(sku) for sku in selected_skus if str(sku).strip()]

    if not selected_centers:
        st.warning("최소 한 개의 센터를 선택하세요.")
        return None
    if not selected_skus:
        st.warning("최소 한 개의 SKU를 선택하세요.")
        return None

    events = _build_events()

    return FilterControls(
        centers=selected_centers,
        skus=selected_skus,
        start=start_ts,
        end=end_ts,
        show_production=show_prod,
        show_in_transit=show_transit,
        use_consumption_forecast=use_cons_forecast,
        lookback_days=int(lookback_days),
        events=events,
        lag_days=int(lag_days),
    )
