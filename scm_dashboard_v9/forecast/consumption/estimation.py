"""일평균 소비량 추정 및 이벤트 적용.

과거 판매 데이터로부터 소비율을 추정하고, 프로모션 이벤트를 적용합니다.
"""

from __future__ import annotations

from typing import Optional, Sequence
import numpy as np

import pandas as pd

from ...core.config import CONFIG


def estimate_daily_consumption(
    snap_long: pd.DataFrame,
    centers_sel: list[str],
    skus_sel: list[str],
    asof_dt: pd.Timestamp,
    lookback_days: int = 28,
) -> dict[tuple[str, str], float]:
    """과거 재고 스냅샷으로부터 일평균 소비율을 추정합니다.

    Args:
        snap_long: 장기 스냅샷 DataFrame (snapshot_date 또는 date 컬럼 필요)
        centers_sel: 대상 센터 리스트
        skus_sel: 대상 SKU 리스트
        asof_dt: 기준 날짜 (이 날짜까지의 데이터 사용)
        lookback_days: 과거 조회 기간 (기본 28일)

    Returns:
        (center, sku) -> 일평균 소비율 매핑 딕셔너리
    """
    snap = snap_long.rename(columns={"snapshot_date": "date"}).copy()
    snap["date"] = pd.to_datetime(snap["date"], errors="coerce").dt.normalize()
    start = (asof_dt - pd.Timedelta(days=int(lookback_days) - 1)).normalize()

    hist = snap[
        (snap["date"] >= start)
        & (snap["date"] <= asof_dt)
        & (snap["center"].isin(centers_sel))
        & (snap["resource_code"].isin(skus_sel))
    ]

    rates: dict[tuple[str, str], float] = {}
    if hist.empty:
        return rates

    for (ct, sku), g in hist.groupby(["center", "resource_code"]):
        series = (
            g.dropna(subset=["date"])  # drop rows without a usable date
            .sort_values("date")
            .groupby("date", as_index=False)["stock_qty"]
            .last()
        )
        if series.empty:
            continue

        ts = series.set_index("date")["stock_qty"].astype(float).asfreq("D").ffill()
        if ts.dropna().shape[0] < max(7, lookback_days // 2):
            continue
        x = np.arange(len(ts), dtype=float)
        y = ts.values.astype(float)
        rate_reg = max(0.0, -np.polyfit(x, y, 1)[0])
        dec = (-np.diff(y)).clip(min=0)
        rate_dec = dec.mean() if len(dec) else 0.0
        rate = max(rate_reg, rate_dec)
        if rate > 0:
            rates[(ct, sku)] = float(rate)
    return rates


def apply_consumption_with_events(
    timeline: pd.DataFrame,
    snapshot: pd.DataFrame,
    *,
    centers: Sequence[str],
    skus: Sequence[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    lookback_days: int = 28,
    events: Optional[Iterable[dict]] = None,
    cons_start: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """재고 타임라인에 일평균 소비율과 프로모션 이벤트를 적용하여 미래 재고를 시뮬레이션합니다.

    이 함수는 다음 단계로 재고 시뮬레이션을 수행합니다:
    1. 과거 스냅샷 데이터로부터 일평균 소비율 추정
    2. 프로모션 이벤트에 따른 소비량 변화(uplift) 계산
    3. 각 센터/SKU별로 일자별 재고 감소 시뮬레이션
    4. 최종 재고량 계산 (음수 방지, 정수 변환)

    Args:
        timeline: 시뮬레이션 대상 타임라인 DataFrame (center, resource_code, date, stock_qty 필요)
        snapshot: 과거 재고 스냅샷 DataFrame (소비율 추정용)
        centers: 대상 센터 리스트
        skus: 대상 SKU (resource_code) 리스트
        start: 시뮬레이션 시작 날짜
        end: 시뮬레이션 종료 날짜
        lookback_days: 소비율 추정 시 과거 조회 기간 (기본 28일)
        events: 프로모션 이벤트 리스트 (각 이벤트: {start, end, uplift})
            - uplift: 소비 증감률 (-1.0 ~ 3.0, 예: 0.5 = 50% 증가)
        cons_start: 소비 시작 날짜 (None이면 latest_snap + 1일로 자동 설정)

    Returns:
        재고 감소가 적용된 타임라인 DataFrame
        - In-Transit, WIP 센터는 소비 적용 안 함
        - 재고량은 0 이상 정수로 제한됨

    Examples:
        >>> events = [
        ...     {"start": "2024-01-10", "end": "2024-01-15", "uplift": 0.5},  # 50% 증가
        ... ]
        >>> result = apply_consumption_with_events(
        ...     timeline=timeline_df,
        ...     snapshot=snapshot_df,
        ...     centers=["center1", "center2"],
        ...     skus=["SKU001", "SKU002"],
        ...     start=pd.Timestamp("2024-01-01"),
        ...     end=pd.Timestamp("2024-01-31"),
        ...     events=events,
        ... )
    """
    centers_list = list(centers)
    skus_list = list(skus)
    start_norm = pd.to_datetime(start).normalize()
    end_norm = pd.to_datetime(end).normalize()
    events_list = list(events) if events else None

    out = timeline.copy()
    if out.empty:
        return out

    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()

    snap_cols = {c.lower(): c for c in snapshot.columns}
    date_col = snap_cols.get("date") or snap_cols.get("snapshot_date")
    if date_col is None:
        raise KeyError("snapshot must include a 'date' or 'snapshot_date' column")

    latest_vals = pd.to_datetime(snapshot[date_col], errors="coerce").dropna()
    latest_snap = latest_vals.max().normalize() if not latest_vals.empty else pd.NaT

    if cons_start is not None:
        cons_start_norm = pd.to_datetime(cons_start).normalize()
        cons_start_norm = max(cons_start_norm, start_norm)
        if not pd.isna(latest_snap):
            cons_start_norm = max(cons_start_norm, latest_snap + pd.Timedelta(days=1))
    elif pd.isna(latest_snap):
        cons_start_norm = start_norm
    else:
        cons_start_norm = max(latest_snap + pd.Timedelta(days=1), start_norm)

    if cons_start_norm > end_norm:
        return out

    idx = pd.date_range(cons_start_norm, end_norm, freq="D")
    uplift = pd.Series(1.0, index=idx)
    if events_list:
        for event in events_list:
            s = pd.to_datetime(event.get("start"), errors="coerce")
            t = pd.to_datetime(event.get("end"), errors="coerce")
            u = min(
                CONFIG.consumption.max_promo_uplift,
                max(
                    CONFIG.consumption.min_promo_uplift, float(event.get("uplift", 0.0))
                ),
            )
            if pd.notna(s) and pd.notna(t):
                s = s.normalize()
                t = t.normalize()
                s = max(s, idx[0])
                t = min(t, idx[-1])
                if s <= t:
                    uplift.loc[s:t] = uplift.loc[s:t] * (1.0 + u)

    rates = {}
    if not pd.isna(latest_snap):
        rates = estimate_daily_consumption(
            snapshot,
            centers_list,
            skus_list,
            latest_snap,
            int(lookback_days),
        )

    chunks: list[pd.DataFrame] = []
    for (ct, sku), grp in out.groupby(["center", "resource_code"]):
        g = grp.sort_values("date").copy()
        g["stock_qty"] = pd.to_numeric(g.get("stock_qty"), errors="coerce")

        if ct in ("In-Transit", "WIP"):
            chunks.append(g)
            continue

        g["stock_qty"] = g["stock_qty"].ffill()

        rate = float(rates.get((ct, sku), 0.0)) if rates else 0.0
        if rate > 0:
            mask = g["date"] >= cons_start_norm
            if mask.any():
                daily = g.loc[mask, "date"].map(uplift).fillna(1.0).values * rate
                stk = g.loc[mask, "stock_qty"].astype(float).values
                for i in range(len(stk)):
                    dec = daily[i]
                    stk[i:] = np.maximum(0.0, stk[i:] - dec)
                g.loc[mask, "stock_qty"] = stk

        chunks.append(g)

    if not chunks:
        return out

    combined = pd.concat(chunks, ignore_index=True)
    combined = combined.sort_values(["center", "resource_code", "date"])
    combined["stock_qty"] = pd.to_numeric(combined["stock_qty"], errors="coerce")
    ffill_mask = ~combined["center"].isin(["In-Transit", "WIP"])
    combined.loc[ffill_mask, "stock_qty"] = (
        combined.loc[ffill_mask]
        .groupby(["center", "resource_code"])["stock_qty"]
        .ffill()
    )
    combined["stock_qty"] = combined["stock_qty"].fillna(0)
    combined["stock_qty"] = combined["stock_qty"].replace([np.inf, -np.inf], 0)
    combined["stock_qty"] = combined["stock_qty"].round().clip(lower=0).astype(int)

    return combined
