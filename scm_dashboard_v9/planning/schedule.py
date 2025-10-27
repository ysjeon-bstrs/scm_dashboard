"""Scheduling helpers for inbound move projections."""

from __future__ import annotations

import pandas as pd

from ..domain.normalization import normalize_dates


# ============================================================
# 상수 정의 (매직 넘버 제거)
# ============================================================

# 과거 도착건 처리 시 적용할 버퍼 일수
PAST_ARRIVAL_BUFFER_DAYS = 3

# 기본 폴백 일수 (날짜 정보가 없을 때)
DEFAULT_FALLBACK_DAYS = 1


# ============================================================
# 공통 헬퍼 함수
# ============================================================


def calculate_predicted_inbound_date(
    moves: pd.DataFrame,
    *,
    today: pd.Timestamp,
    lag_days: int,
    past_arrival_buffer_days: int = PAST_ARRIVAL_BUFFER_DAYS,
) -> pd.DataFrame:
    """
    이동 원장에 예상 입고일(pred_inbound_date)을 계산하여 추가합니다.

    이 함수는 v9_app.py와 ui/tables.py에서 공통으로 사용되며,
    UI 표시용 pred_inbound_date 계산 로직을 제공합니다.

    계산 규칙:
    1. inbound_date가 있으면 그대로 사용 (최우선)
    2. WIP: event_date를 그대로 사용 (리드타임 추가 안 함)
    3. In-Transit:
       - 과거/오늘 도착: today + past_arrival_buffer_days
       - 미래 도착: arrival_date/eta_date + lag_days

    Args:
        moves: 이동 원장 DataFrame
        today: 오늘 날짜 (기준일)
        lag_days: 입고 반영 리드타임 (일)
        past_arrival_buffer_days: 과거 도착건 처리 버퍼 (기본 3일)

    Returns:
        pred_inbound_date 컬럼이 추가된 DataFrame

    Examples:
        >>> moves_with_pred = calculate_predicted_inbound_date(
        ...     moves,
        ...     today=pd.Timestamp("2024-01-15"),
        ...     lag_days=5
        ... )
        >>> moves_with_pred["pred_inbound_date"]
    """
    if moves.empty:
        moves_out = moves.copy()
        moves_out["pred_inbound_date"] = pd.Series(
            pd.NaT, index=moves_out.index, dtype="datetime64[ns]"
        )
        return moves_out

    moves_out = moves.copy()
    today_norm = pd.to_datetime(today).normalize()

    # ========================================
    # 1단계: 필수 컬럼 보완
    # ========================================
    for col in [
        "carrier_mode",
        "inbound_date",
        "arrival_date",
        "eta_date",
        "event_date",
    ]:
        if col not in moves_out.columns:
            if "date" in col:
                moves_out[col] = pd.Series(
                    pd.NaT, index=moves_out.index, dtype="datetime64[ns]"
                )
            else:
                moves_out[col] = pd.Series("", index=moves_out.index, dtype="object")

    # ========================================
    # 2단계: 날짜 컬럼 정규화
    # ========================================
    for col in ["arrival_date", "eta_date", "inbound_date", "event_date"]:
        if col in moves_out.columns:
            moves_out[col] = pd.to_datetime(
                moves_out[col], errors="coerce"
            ).dt.normalize()

    # ========================================
    # 3단계: carrier_mode 확인
    # ========================================
    carrier_mode = moves_out.get("carrier_mode", pd.Series("", index=moves_out.index))
    is_wip = carrier_mode.astype(str).str.upper() == "WIP"

    # ========================================
    # 4단계: pred_inbound_date 계산
    # ========================================
    pred_inbound = pd.Series(pd.NaT, index=moves_out.index, dtype="datetime64[ns]")

    # 4-1. inbound_date가 있으면 우선 사용
    mask_inbound = moves_out["inbound_date"].notna()
    if mask_inbound.any():
        pred_inbound.loc[mask_inbound] = moves_out.loc[mask_inbound, "inbound_date"]

    # 4-2. WIP: event_date 그대로 사용
    wip_mask = is_wip & (~mask_inbound)
    if wip_mask.any():
        event_series = moves_out.get("event_date")
        if isinstance(event_series, pd.Series):
            event_normalized = pd.to_datetime(
                event_series, errors="coerce"
            ).dt.normalize()
            wip_with_event = wip_mask & event_normalized.notna()
            if wip_with_event.any():
                pred_inbound.loc[wip_with_event] = event_normalized.loc[wip_with_event]

    # 4-3. In-Transit: arrival/eta + 리드타임
    intransit_mask = (~is_wip) & (~mask_inbound)
    arrival_series = moves_out.get("arrival_date")
    eta_series = moves_out.get("eta_date")

    if isinstance(arrival_series, pd.Series) and isinstance(eta_series, pd.Series):
        arrival_normalized = pd.to_datetime(
            arrival_series, errors="coerce"
        ).dt.normalize()
        eta_normalized = pd.to_datetime(eta_series, errors="coerce").dt.normalize()
        effective_arrival = arrival_normalized.fillna(eta_normalized)
    elif isinstance(arrival_series, pd.Series):
        effective_arrival = pd.to_datetime(
            arrival_series, errors="coerce"
        ).dt.normalize()
    elif isinstance(eta_series, pd.Series):
        effective_arrival = pd.to_datetime(eta_series, errors="coerce").dt.normalize()
    else:
        effective_arrival = pd.Series(
            pd.NaT, index=moves_out.index, dtype="datetime64[ns]"
        )

    mask_eta = intransit_mask & effective_arrival.notna()

    if mask_eta.any():
        # 과거/오늘 도착: today + buffer_days
        past_eta = mask_eta & (effective_arrival <= today_norm)
        if past_eta.any():
            pred_inbound.loc[past_eta] = today_norm + pd.Timedelta(
                days=int(past_arrival_buffer_days)
            )

        # 미래 도착: arrival + lag_days
        future_eta = mask_eta & (effective_arrival > today_norm)
        if future_eta.any():
            pred_inbound.loc[future_eta] = effective_arrival.loc[
                future_eta
            ] + pd.Timedelta(days=int(lag_days))

    moves_out["pred_inbound_date"] = pd.to_datetime(pred_inbound).dt.normalize()
    return moves_out


def annotate_move_schedule(
    moves: pd.DataFrame,
    *,
    today: pd.Timestamp,
    lag_days: int,
    horizon_end: pd.Timestamp,
    fallback_days: int = 1,
) -> pd.DataFrame:
    """Attach predicted inbound dates aligned with the centre inventory policy."""

    today_norm = pd.to_datetime(today).normalize()
    horizon_end = pd.to_datetime(horizon_end).normalize()
    fallback_date = min(
        today_norm + pd.Timedelta(days=int(fallback_days)),
        horizon_end + pd.Timedelta(days=1),
    )

    out = normalize_dates(moves)
    out["carrier_mode"] = out.get("carrier_mode", "").astype(str).str.upper()
    is_wip = out["carrier_mode"] == "WIP"
    actual_onboard = pd.to_datetime(
        out.get("onboard_date"), errors="coerce"
    ).dt.normalize()
    out["_onboard_date_actual"] = actual_onboard

    pred = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns]")
    inbound_col = out.get("inbound_date")
    has_inbound = (
        inbound_col.notna()
        if inbound_col is not None
        else pd.Series(False, index=out.index)
    )
    if has_inbound.any():
        pred.loc[has_inbound] = inbound_col.loc[has_inbound]

    # WIP: event_date (intended_push_date) 그대로 사용
    wip_mask = is_wip & (~has_inbound)
    if wip_mask.any():
        event_raw = out.get("event_date")
        if isinstance(event_raw, pd.Series):
            event_series = pd.to_datetime(event_raw, errors="coerce").dt.normalize()
            wip_with_event = wip_mask & event_series.notna()
            if wip_with_event.any():
                pred.loc[wip_with_event] = event_series.loc[wip_with_event]

    # In-Transit: arrival/eta + 리드타임
    arrival_raw = out.get("arrival_date")
    if isinstance(arrival_raw, pd.Series):
        arrival_series = pd.to_datetime(arrival_raw, errors="coerce").dt.normalize()
    else:
        arrival_series = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns]")

    eta_raw = out.get("eta_date")
    if isinstance(eta_raw, pd.Series):
        eta_series = pd.to_datetime(eta_raw, errors="coerce").dt.normalize()
    else:
        eta_series = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns]")
    effective_arrival = arrival_series.fillna(eta_series)
    intransit_mask = (~is_wip) & (~has_inbound)
    has_arrival = intransit_mask & effective_arrival.notna()
    if has_arrival.any():
        arr_dates = effective_arrival
        # 과거/오늘 arrival: today + 3일 (수정: 기존 lag_days에서 3일로)
        past_arrival = has_arrival & (arr_dates <= today_norm)
        if past_arrival.any():
            pred.loc[past_arrival] = today_norm + pd.Timedelta(days=3)

        # 미래 arrival: arrival + lag_days
        future_arrival = has_arrival & (arr_dates > today_norm)
        if future_arrival.any():
            pred.loc[future_arrival] = arr_dates.loc[future_arrival] + pd.Timedelta(
                days=int(lag_days)
            )

    has_signal = has_inbound | (wip_mask & pred.notna()) | has_arrival
    need_fallback = has_signal & pred.isna()
    if need_fallback.any():
        pred.loc[need_fallback] = fallback_date

    pred = pred.where(has_signal, pd.NaT)
    out["pred_inbound_date"] = pd.to_datetime(pred).dt.normalize()
    out["pred_inbound_date"] = out["pred_inbound_date"].clip(
        upper=horizon_end + pd.Timedelta(days=1)
    )
    out["in_transit_end_date"] = out["pred_inbound_date"]
    return out
