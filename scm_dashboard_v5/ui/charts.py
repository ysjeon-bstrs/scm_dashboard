# scm_dashboard_v5/ui/charts.py
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import streamlit as st

try:  # Plotly는 선택적 의존성이므로 임포트 실패를 허용한다.
    import plotly.express as px  # type: ignore
    import plotly.graph_objects as go  # type: ignore
except ImportError as _plotly_err:  # pragma: no cover - 의존성 결손 환경만 재현 가능
    px = None  # type: ignore[assignment]
    go = None  # type: ignore[assignment]
    _PLOTLY_IMPORT_ERROR = _plotly_err
else:
    _PLOTLY_IMPORT_ERROR = None

from scm_dashboard_v5.ui.kpi import render_sku_summary_cards as _render_sku_summary_cards
from scm_dashboard_v5.forecast import apply_consumption_with_events

# ---------------- Palette (SKU -> Color) ----------------
# 계단식 차트와 최대한 비슷한 톤(20+색)
_AMAZON_PALETTE = [
    "#4E79A7","#F28E2B","#E15759","#76B7B2","#59A14F",
    "#EDC948","#B07AA1","#FF9DA7","#9C755F","#BAB0AC",
    "#1F77B4","#FF7F0E","#2CA02C","#D62728","#9467BD",
    "#8C564B","#E377C2","#7F7F7F","#BCBD22","#17BECF",
]

_PLOTLY_WARNING_EMITTED = False


def _safe_dataframe(
    df: Optional[pd.DataFrame],
    *,
    index: Optional[Sequence[pd.Timestamp]] = None,
    columns: Optional[Sequence[str]] = None,
    fill_value: float = 0.0,
    dtype: type = float,
) -> pd.DataFrame:
    """Return a DataFrame that is always usable for downstream math/plotting."""

    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(dtype=dtype)

    result = df.copy()

    if columns is not None:
        result = result.reindex(columns=list(columns), fill_value=fill_value)

    if index is not None:
        if not isinstance(index, pd.Index):
            index = pd.Index(index)
        result = result.reindex(index, fill_value=fill_value)

    if result.empty and columns is not None:
        result = pd.DataFrame([], columns=list(columns), dtype=dtype)

    return result


def _safe_series(obj: Optional[pd.Series | Iterable[float]], *, length_hint: int | None = None) -> pd.Series:
    """Coerce arbitrary iterables into a Pandas Series for plotting."""

    if isinstance(obj, pd.Series):
        return obj

    if obj is None:
        data: List[float] = []
    elif isinstance(obj, (pd.Index, np.ndarray)):
        data = list(obj)
    elif hasattr(obj, "tolist"):
        data = list(obj.tolist())  # type: ignore[arg-type]
    elif isinstance(obj, Iterable) and not isinstance(obj, (str, bytes)):
        data = list(obj)
    else:
        data = [obj] if obj is not None else []

    if length_hint is not None and len(data) > length_hint:
        data = data[:length_hint]

    return pd.Series(data)


def _to_plot_list(values: Optional[Iterable]) -> List:
    """Convert arbitrary iterable values into a clean list for Plotly APIs."""

    if values is None:
        return []

    if isinstance(values, (pd.Index, pd.Series)):
        cleaned = values.dropna().tolist()
    elif isinstance(values, np.ndarray):
        cleaned = [v for v in values.tolist() if not pd.isna(v)]
    elif isinstance(values, Iterable) and not isinstance(values, (str, bytes)):
        cleaned = [v for v in values if not pd.isna(v)]
    else:
        cleaned = [] if pd.isna(values) else [values]

    return cleaned


def _safe_add_bar(
    fig: "go.Figure",
    *,
    x: Optional[Iterable],
    y: Optional[Iterable],
    name: str,
    marker_color: Optional[str],
    **kwargs: object,
) -> None:
    """Add a bar trace only when data is valid, preventing runtime TypeErrors."""

    xs = _to_plot_list(x)
    ys = _to_plot_list(y)

    if not xs or not ys:
        return

    if marker_color is None:
        return

    if len(xs) != len(ys):
        limit = min(len(xs), len(ys))
        xs = xs[:limit]
        ys = ys[:limit]

    try:
        fig.add_bar(x=xs, y=ys, name=name, marker_color=marker_color, **kwargs)
    except Exception:  # pragma: no cover - plotting defensive guard
        return


def _safe_add_scatter(
    fig: "go.Figure",
    *,
    x: Optional[Iterable],
    y: Optional[Iterable],
    name: str,
    line: Optional[Dict[str, object]] = None,
    yaxis: str = "y",
    mode: str = "lines",
    **kwargs: object,
) -> None:
    """Add a scatter trace only when data is valid."""

    xs = _to_plot_list(x)
    ys = _to_plot_list(y)

    if not xs or not ys:
        return

    try:
        fig.add_trace(
            go.Scatter(x=xs, y=ys, name=name, mode=mode, line=line, yaxis=yaxis, **kwargs)
        )
    except Exception:  # pragma: no cover - plotting defensive guard
        return


def _as_naive_timestamp(value: pd.Timestamp | str | None) -> pd.Timestamp:
    """Return a timezone-naive timestamp for consistent comparisons."""

    ts = pd.Timestamp(value) if value is not None else pd.Timestamp.today()
    try:
        return ts.tz_localize(None)  # handles tz-aware values
    except TypeError:
        return ts  # already naive


def _ensure_naive_index(idx: pd.Index) -> pd.DatetimeIndex:
    """Coerce an index to a timezone-naive DatetimeIndex."""

    dt_idx = pd.to_datetime(idx, errors="coerce")
    if not isinstance(dt_idx, pd.DatetimeIndex):
        dt_idx = pd.DatetimeIndex(dt_idx)
    if dt_idx.tz is not None:
        dt_idx = dt_idx.tz_localize(None)
    return dt_idx


def _ensure_plotly_available() -> bool:
    """Plotly 미설치 환경에서도 ImportError 없이 경고만 띄우도록 보조."""

    global _PLOTLY_WARNING_EMITTED
    if _PLOTLY_IMPORT_ERROR is None:
        return True
    if not _PLOTLY_WARNING_EMITTED:
        st.warning(
            "Plotly가 설치되어 있지 않아 차트를 렌더링할 수 없습니다. "
            "관리자에게 Plotly 설치를 요청하거나 requirements를 확인하세요.\n"
            f"원인: {_PLOTLY_IMPORT_ERROR}"
        )
        _PLOTLY_WARNING_EMITTED = True
    return False

def _sku_colors(skus: Sequence[str], base: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """SKU 별 고정 색상 매핑을 만든다. (기존 매핑을 넘기면 그대로 존중)"""
    cmap = {} if base is None else dict(base)
    i = 0
    for s in skus:
        if s not in cmap:
            cmap[s] = _AMAZON_PALETTE[i % len(_AMAZON_PALETTE)]
            i += 1
    return cmap


def _is_wip_center_name(value: object) -> bool:
    """Return True when the provided center name represents a WIP/production bucket."""

    text = str(value or "")
    upper = text.upper()
    if "WIP" in upper or "PRODUCTION" in upper:
        return True
    return "생산" in text


def _drop_wip_centers(df: pd.DataFrame, *, center_col: str = "center") -> pd.DataFrame:
    """Remove rows that represent WIP/production centers from the given DataFrame."""

    if not isinstance(df, pd.DataFrame) or center_col not in df.columns:
        return df

    centers = df[center_col].astype(str)
    mask = centers.map(_is_wip_center_name)
    if not mask.any():
        return df.copy()

    return df.loc[~mask].copy()

def _pick_amazon_centers(all_centers: Iterable[str]) -> List[str]:
    """선택 센터 중 Amazon 계열만 추출 (없으면 자동 감지에 사용)"""
    out = []
    for c in all_centers:
        if not c:
            continue
        c_up = str(c).upper()
        if c_up.startswith("AMZ") or "AMAZON" in c_up:
            out.append(str(c))
    return out


def _contains_wip_center(centers: Sequence[str]) -> bool:
    """Return True if the selection looks like it includes a WIP/태광 center."""

    for center in centers:
        norm = str(center).replace(" ", "").lower()
        if not norm:
            continue
        if norm == "wip":
            return True
        if "태광" in norm or "taekwang" in norm or "tae-kwang" in norm:
            return True
    return False

# ---------------- Core helpers ----------------
def _coerce_cols(df: pd.DataFrame) -> Dict[str, str]:
    cols = {c.lower(): c for c in df.columns}
    date_col   = cols.get("snapshot_date") or cols.get("date")
    center_col = cols.get("center")
    sku_col    = cols.get("resource_code") or cols.get("sku")
    qty_col    = cols.get("stock_qty") or cols.get("qty") or cols.get("quantity")
    return {"date": date_col, "center": center_col, "sku": sku_col, "qty": qty_col}

def _sales_from_snapshot(
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
    c = _coerce_cols(snap_long)
    s = snap_long.rename(
        columns={c["date"]: "date", c["center"]: "center", c["sku"]: "resource_code", c["qty"]: "stock_qty"}
    )[["date", "center", "resource_code", "stock_qty"]].copy()

    s["date"] = pd.to_datetime(s["date"], errors="coerce").dt.normalize()
    s = s[s["center"].astype(str).isin(centers) & s["resource_code"].astype(str).isin(skus)]
    if s.empty:
        idx = pd.date_range(start, end, freq="D")
        return pd.DataFrame(0, index=idx, columns=skus)

    pv = (s.groupby(["date", "resource_code"])["stock_qty"].sum()
            .unstack("resource_code")
            .reindex(columns=skus, fill_value=0)
            .sort_index())
    pv = pv.asfreq("D").ffill()  # D 간격 보정
    d  = pv.diff().fillna(0)
    sales = (-d).clip(lower=0)  # 감소분만 판매
    sales = sales.loc[(sales.index >= start) & (sales.index <= end)]
    return sales

def _inventory_matrix(
    snap_long: pd.DataFrame,
    centers: List[str],
    skus: List[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """선택 센터×SKU의 재고(실측) 시계열 매트릭스. index=date, columns=sku"""
    c = _coerce_cols(snap_long)
    s = snap_long.rename(
        columns={c["date"]: "date", c["center"]: "center", c["sku"]: "resource_code", c["qty"]: "stock_qty"}
    )[["date", "center", "resource_code", "stock_qty"]].copy()

    s["date"] = pd.to_datetime(s["date"], errors="coerce").dt.normalize()
    s = s[s["center"].astype(str).isin(centers) & s["resource_code"].astype(str).isin(skus)]
    if s.empty:
        idx = pd.date_range(start, end, freq="D")
        return pd.DataFrame(0, index=idx, columns=skus)

    pv = (s.groupby(["date", "resource_code"])["stock_qty"].sum()
            .unstack("resource_code")
            .reindex(columns=skus, fill_value=0)
            .sort_index())
    pv = pv.asfreq("D").ffill()
    pv = pv.loc[(pv.index >= start) & (pv.index <= end)]
    return pv


def _timeline_inventory_matrix(
    timeline: Optional[pd.DataFrame],
    centers: Sequence[str],
    skus: Sequence[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> Optional[pd.DataFrame]:
    """Pivot the step-chart timeline into a date×SKU inventory matrix."""

    if timeline is None or timeline.empty:
        return None

    required_cols = {"date", "center", "resource_code", "stock_qty"}
    if not required_cols.issubset(timeline.columns):
        return None

    df = timeline.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df[df["date"].notna()]
    df = df[df["center"].astype(str).isin(centers)]
    df = df[df["resource_code"].astype(str).isin(skus)]

    if df.empty:
        return None

    pivot = (
        df.groupby(["date", "resource_code"])["stock_qty"].sum()
        .unstack("resource_code")
        .reindex(columns=list(skus), fill_value=0.0)
        .sort_index()
    )

    pivot = pivot.loc[(pivot.index >= start) & (pivot.index <= end)]
    return pivot

# ---------------- Public renderer ----------------
def render_amazon_sales_vs_inventory(
    snap_long: pd.DataFrame,
    centers: Sequence[str],
    skus: Sequence[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    today: pd.Timestamp,
    *,
    color_map: Optional[Dict[str, str]] = None,
    show_ma7: bool = True,
    show_inventory_forecast: bool = True,
    use_consumption_forecast: bool = False,
    lookback_days: int = 28,
    events: Optional[Sequence[Dict[str, object]]] = None,
    timeline: Optional[pd.DataFrame] = None,
    show_wip: Optional[bool] = None,
) -> None:
    """
    Amazon US 일별 판매(누적 막대) vs. 재고(라인) 패널.
    - 판매(실측): 스냅샷 감소분만 사용
    - 판매(예측): 최근 7일(옵션) 평균을 오늘 다음 날부터 적용 (막대 색상 동일)
    - 재고(실측): 실선, 재고(예측): 오늘 이후 점선
    - SKU별 색상 고정, 계단형 라인(hv), 오늘 세로 기준선 추가
    - 생산중(WIP) 수량은 Amazon 판매/재고 패널에서 더 이상 노출되지 않음
    """
    show_wip_flag = False if show_wip is None else bool(show_wip)

    if not _ensure_plotly_available():
        return

    centers_list = [str(c) for c in centers]
    if not show_wip_flag:
        centers_list = [c for c in centers_list if not _is_wip_center_name(c)]

    snap_df = snap_long.copy()
    if not show_wip_flag:
        snap_df = _drop_wip_centers(snap_df, center_col="center")

    timeline_df: Optional[pd.DataFrame] = None
    timeline_forecast: Optional[pd.DataFrame] = None
    if isinstance(timeline, pd.DataFrame):
        timeline_df = timeline.copy()
        if not show_wip_flag:
            timeline_df = _drop_wip_centers(timeline_df, center_col="center")

    if timeline_df is not None and use_consumption_forecast:
        timeline_forecast = apply_consumption_with_events(
            timeline,
            snap_long,
            centers=list(centers_list),
            skus=list(skus),
            start=start,
            end=end,
            lookback_days=int(lookback_days),
            events=list(events) if events else None,
        )
        if not show_wip_flag and isinstance(timeline_forecast, pd.DataFrame):
            timeline_forecast = _drop_wip_centers(timeline_forecast, center_col="center")
    else:
        timeline_forecast = timeline_df

    start = _as_naive_timestamp(start)
    end = _as_naive_timestamp(end)
    today = _as_naive_timestamp(today)

    skus = [str(s) for s in skus]
    if not skus:
        st.info("선택된 SKU가 없습니다.")
        return

    amz_centers = _pick_amazon_centers(centers_list)
    if not show_wip_flag:
        amz_centers = [c for c in amz_centers if not _is_wip_center_name(c)]
    if not amz_centers:
        # 그래도 AMZ 관련 센터가 스냅샷에 있다면 자동 감지
        amz_centers = _pick_amazon_centers(
            snap_df.get("center", pd.Series()).dropna().unique()
        )
        if not show_wip_flag:
            amz_centers = [c for c in amz_centers if not _is_wip_center_name(c)]

    if not amz_centers:
        st.info("Amazon/AMZ 계열 센터가 보이지 않습니다.")
        return

    # 색상 고정
    cmap = _sku_colors(skus, base=color_map)

    # 판매(실측) 및 이동평균 (필터 lookback_days 반영)
    sales = _safe_dataframe(
        _sales_from_snapshot(snap_df, amz_centers, skus, start, end),
        columns=skus,
    )
    if not sales.empty:
        sales.index = _ensure_naive_index(sales.index)
    ma_window = max(1, int(lookback_days))
    ma7 = sales.rolling(ma_window, min_periods=1).mean() if show_ma7 else None
    if ma7 is not None:
        ma7 = _safe_dataframe(ma7, columns=skus)
        if not ma7.empty:
            ma7.index = _ensure_naive_index(ma7.index)
        else:
            ma7 = None

    # 재고(실측)
    inv = _safe_dataframe(
        _inventory_matrix(snap_df, amz_centers, skus, start, end), columns=skus
    )
    if not inv.empty:
        inv.index = _ensure_naive_index(inv.index)

    future_idx = pd.date_range(today + pd.Timedelta(days=1), end, freq="D")
    if len(future_idx) > 0:
        future_idx = _ensure_naive_index(future_idx)

    inv_future: Optional[pd.DataFrame] = None
    future_sales_from_inventory: Optional[pd.DataFrame] = None
    timeline_source = timeline_forecast if timeline_forecast is not None else timeline_df
    timeline_pivot = _timeline_inventory_matrix(timeline_source, amz_centers, skus, start, end)
    if timeline_pivot is not None and not timeline_pivot.empty:
        timeline_pivot.index = _ensure_naive_index(timeline_pivot.index)
        timeline_pivot = _safe_dataframe(timeline_pivot, columns=skus)
    anchor_date: Optional[pd.Timestamp] = None
    start_vector: Optional[pd.DataFrame] = None

    wip_pivot: Optional[pd.DataFrame] = None
    wip_requested = _contains_wip_center(centers)
    if show_wip and wip_requested:
        wip_source = timeline_forecast if timeline_forecast is not None else timeline
        wip_pivot = _timeline_inventory_matrix(wip_source, ["WIP"], skus, start, end)
        if wip_pivot is not None and not wip_pivot.empty:
            wip_pivot.index = _ensure_naive_index(wip_pivot.index)
            wip_pivot = _safe_dataframe(wip_pivot, columns=skus)
        else:
            wip_pivot = None
    else:
        wip_pivot = None

    if show_inventory_forecast:
        if use_consumption_forecast and timeline_pivot is not None and not timeline_pivot.empty:
            anchor_date = None
            if not inv.empty:
                past_idx = inv.index[inv.index <= today]
                if len(past_idx) > 0:
                    anchor_date = past_idx.max()
            if anchor_date is None:
                timeline_past = timeline_pivot.index[timeline_pivot.index <= today]
                if len(timeline_past) > 0:
                    anchor_date = timeline_past.max()

            if anchor_date is not None:
                anchor_date = pd.Timestamp(anchor_date).normalize()
                trimmed = timeline_pivot.loc[timeline_pivot.index >= anchor_date]
                if anchor_date not in trimmed.index:
                    prev = timeline_pivot.loc[timeline_pivot.index <= anchor_date]
                    if not prev.empty:
                        prev_row = prev.tail(1)
                        trimmed = pd.concat([prev_row, trimmed])

                if not trimmed.empty:
                    inv_future = _safe_dataframe(trimmed, columns=skus)
        if inv_future is None:
            # 오늘을 포함한 마지막 실측 값을 기준으로 이후 구간을 잇는다.
            if (inv.index <= today).any():
                start_vector = inv.loc[inv.index <= today].iloc[[-1]].copy()
                start_vector = _safe_dataframe(start_vector, columns=skus)
                anchor_date = pd.Timestamp(start_vector.index[-1]).normalize()
                start_vector.index = pd.DatetimeIndex([anchor_date])
        else:
            if anchor_date is None:
                anchor_date = pd.Timestamp(today).normalize()
            if start_vector is None and inv_future is not None and not inv_future.empty:
                inv_future = _safe_dataframe(inv_future.copy(), columns=skus)
                inv_future.index = pd.to_datetime(inv_future.index).normalize()
                if anchor_date in inv_future.index:
                    start_vector = inv_future.loc[[anchor_date]].copy()
                else:
                    first_row = inv_future.iloc[[0]].copy()
                    first_row = _safe_dataframe(first_row, columns=skus)
                    anchor_date = pd.Timestamp(first_row.index[0]).normalize()
                    first_row.index = pd.DatetimeIndex([anchor_date])
                    start_vector = first_row
            if start_vector is None and anchor_date is not None:
                start_vector = pd.DataFrame(
                    [np.zeros(len(skus))],
                    columns=skus,
                    index=pd.DatetimeIndex([anchor_date]),
                )
            elif start_vector is None:
                start_vector = _safe_dataframe(None, columns=skus)

        if anchor_date is not None and start_vector is not None:
            # 1) 데이터에 이미 미래 재고가 존재한다면 그대로 사용
            future_actual = inv.loc[inv.index > anchor_date]
            if not future_actual.empty:
                inv_future = _safe_dataframe(
                    pd.concat([start_vector, future_actual]).loc[anchor_date:], columns=skus
                )
            else:
                # 2) 미래 데이터가 없으면 MA7 소비량을 기반으로 점선 예측을 생성
                forecast_idx = pd.date_range(anchor_date + pd.Timedelta(days=1), end, freq="D")
                if len(forecast_idx) > 0:
                    if ma7 is None or ma7.empty:
                        daily = pd.DataFrame(0.0, index=forecast_idx, columns=skus)
                    else:
                        extended = (
                            ma7.reindex(ma7.index.union(forecast_idx))
                            .sort_index()
                            .ffill()
                        )
                        daily = extended.reindex(forecast_idx).fillna(0.0)

                    cur = start_vector.iloc[0].astype(float).values
                    vals = []
                    for d in forecast_idx:
                        forecast_step = daily.loc[d].reindex(skus, fill_value=0.0).values
                        cur = np.maximum(0.0, cur - forecast_step)
                        vals.append(cur.copy())
                    forecast_df = pd.DataFrame(vals, index=forecast_idx, columns=skus)
                    inv_future = _safe_dataframe(
                        pd.concat([start_vector, forecast_df]), columns=skus
                    )
        else:
            inv_future = None

    if isinstance(inv_future, pd.DataFrame):
        inv_future = _safe_dataframe(inv_future, columns=skus)

    if inv_future is not None and not inv_future.empty:
        inv_future.index = _ensure_naive_index(inv_future.index)
        inv_future = inv_future.reindex(columns=skus, fill_value=0.0)
        inv_future = inv_future.sort_index()
        future_sales_from_inventory = (-inv_future.diff()).iloc[1:]
        if future_sales_from_inventory is not None:
            future_sales_from_inventory = _safe_dataframe(
                future_sales_from_inventory, columns=skus, index=future_idx
            )
            if not future_sales_from_inventory.empty:
                future_sales_from_inventory.index = _ensure_naive_index(
                    future_sales_from_inventory.index
                )
                future_sales_from_inventory = future_sales_from_inventory.clip(lower=0.0)
            else:
                future_sales_from_inventory = None

        inv_future_plot = _safe_dataframe(inv_future.round(0), columns=skus)
        if not inv_future_plot.empty:
            inv_future_plot.index = _ensure_naive_index(inv_future_plot.index)
        else:
            inv_future_plot = None
    else:
        inv_future_plot = None

    # ---------- Figure ----------
    fig = go.Figure()

    # (1) 판매: SKU별로 누적(bar)
    past_sales = _safe_dataframe(sales.loc[sales.index <= today], columns=skus)
    if not past_sales.empty:
        past_sales.index = _ensure_naive_index(past_sales.index)

    # 실측 막대
    for sku in skus:
        series = past_sales.get(sku)
        color = cmap.get(sku)
        _safe_add_bar(
            fig,
            name=f"{sku} 판매",
            x=past_sales.index if not past_sales.empty else None,
            y=series,
            marker_color=color,
            opacity=0.95,
            hovertemplate="날짜 %{x|%Y-%m-%d}<br>%{fullData.name}: %{y:,} EA<extra></extra>",
            yaxis="y",
        )
    # 예측 막대(색상 동일, 투명도만 낮춤)
    if len(future_idx) > 0:
        future_sales: Optional[pd.DataFrame]
        if future_sales_from_inventory is not None:
            future_sales = _safe_dataframe(
                future_sales_from_inventory, columns=skus, index=future_idx
            )
        elif ma7 is not None:
            future_sales = _safe_dataframe(
                ma7.reindex(future_idx).ffill().fillna(0.0), columns=skus, index=future_idx
            )
        else:
            future_sales = pd.DataFrame(0.0, index=future_idx, columns=skus)

        future_sales = _safe_dataframe(future_sales, columns=skus, index=future_idx)
        if future_sales.empty:
            future_sales = None
        else:
            future_sales.index = _ensure_naive_index(future_sales.index)

        if future_sales is not None:
            for sku in skus:
                series = future_sales.get(sku)
                color = cmap.get(sku)
                _safe_add_bar(
                    fig,
                    name=f"{sku} 판매(예측)",
                    x=future_sales.index,
                    y=series,
                    marker_color=color,
                    opacity=0.25,
                    hovertemplate="날짜 %{x|%Y-%m-%d}<br>%{fullData.name}: %{y:,} EA<extra></extra>",
                    yaxis="y",
                )

    # (2) 재고: SKU별 선(실선), 오늘 이후는 점선
    inv_past = _safe_dataframe(inv.loc[inv.index <= today], columns=skus)
    inv_past_plot = _safe_dataframe(inv_past.round(0), columns=skus)
    if inv_past_plot.empty:
        inv_past_plot = None
    else:
        inv_past_plot.index = _ensure_naive_index(inv_past_plot.index)

    for sku in skus:
        color = cmap.get(sku)
        series = inv_past_plot.get(sku) if inv_past_plot is not None else None
        _safe_add_scatter(
            fig,
            x=inv_past_plot.index if inv_past_plot is not None else None,
            y=series,
            name=f"{sku} 재고(실측)",
            line=dict(color=color, width=2, shape="hv") if color else None,
            yaxis="y2",
            hovertemplate="날짜 %{x|%Y-%m-%d}<br>%{fullData.name}: %{y:,.0f} EA<extra></extra>",
        )

    if inv_future_plot is not None:
        for sku in skus:
            color = cmap.get(sku)
            series = inv_future_plot.get(sku) if sku in inv_future_plot.columns else None
            _safe_add_scatter(
                fig,
                x=inv_future_plot.index,
                y=series,
                name=f"{sku} 재고(예측)",
                line=dict(color=color, width=2, dash="dash", shape="hv") if color else None,
                yaxis="y2",
                hovertemplate="날짜 %{x|%Y-%m-%d}<br>%{fullData.name}: %{y:,.0f} EA<extra></extra>",
            )

    # (3) 오늘 기준선
    fig.add_vline(x=today, line_color="red", line_dash="dot", line_width=2)

    # 레이아웃 (제목 겹침 제거용 여백 포함)
    fig.update_layout(
        barmode="stack",
        margin=dict(l=16, r=16, t=40, b=40),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(title="Date"),
        yaxis=dict(title="판매량 (EA/Day)"),
        yaxis2=dict(title="재고 (EA)", overlaying="y", side="right", tickformat=",d"),
    )

    # 안내문(상단 설명을 그림 안에 넣지 않아 제목 겹침 방지)
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})


def render_amazon_panel(*args: object, **kwargs: object) -> None:
    """Backward-compatible alias for :func:`render_amazon_sales_vs_inventory`."""

    show_wip = kwargs.pop("show_wip", None)
    show_production = kwargs.pop("show_production", None)

    if show_wip is None:
        if show_production is not None:
            show_wip = bool(show_production)
        else:
            show_wip = False
    else:
        show_wip = bool(show_wip)

    render_amazon_sales_vs_inventory(*args, show_wip=show_wip, **kwargs)

# --- STEP CHART (v5_main이 호출하는 공개 API) ---------------------------------
# 충분히 긴 팔레트 (SKU별 고정색)
_STEP_PALETTE = [
    "#4E79A7","#F28E2B","#E15759","#76B7B2","#59A14F",
    "#EDC948","#B07AA1","#FF9DA7","#9C755F","#BAB0AC",
    "#1F77B4","#FF7F0E","#2CA02C","#D62728","#9467BD",
    "#8C564B","#E377C2","#7F7F7F","#BCBD22","#17BECF",
    "#8DD3C7","#FFFFB3","#BEBADA","#FB8072","#80B1D3",
    "#FDB462","#B3DE69","#FCCDE5","#D9D9D9","#BC80BD",
    "#CCEBC5","#FFED6F"
]

def _sku_color_map(labels: list[str]) -> dict[str, str]:
    """'SKU @ Center' 라벨들에서 SKU만 뽑아 SKU별 색을 고정 매핑."""
    m, i = {}, 0
    for lb in labels:
        sku = lb.split(" @ ", 1)[0] if " @ " in lb else lb
        if sku not in m:
            m[sku] = _STEP_PALETTE[i % len(_STEP_PALETTE)]
            i += 1
    return m

def render_step_chart(
    timeline: pd.DataFrame,
    *,
    centers: list[str] | None = None,
    skus: list[str] | None = None,
    start: pd.Timestamp,
    end: pd.Timestamp,
    today: pd.Timestamp | None = None,
    horizon_days: int = 0,
    show_in_transit: bool = True,
    show_wip: bool | None = None,
    title: str = "선택한 SKU × 센터(및 In‑Transit/WIP) 계단식 재고 흐름",
    **kwargs
) -> None:
    """
    v5_main에서 그대로 호출하는 공개 API.
    timeline: columns=[date, center, resource_code, stock_qty] (apply_consumption_with_events 반영 가능)
    """
    if not _ensure_plotly_available():
        return

    if timeline is None or timeline.empty:
        st.info("타임라인 데이터가 없습니다.")
        return

    df = timeline.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    # "WIP" 센터를 태광KR 소속 생산중 상태로 통합한다.
    df["center"] = df["center"].astype(str)
    df["resource_code"] = df["resource_code"].astype(str)
    wip_mask = df["center"] == "WIP"
    df["is_wip"] = wip_mask
    df.loc[wip_mask, "center"] = "태광KR"

    show_production = kwargs.pop("show_production", None)
    if show_wip is None:
        show_wip = True if show_production is None else bool(show_production)
    else:
        show_wip = bool(show_wip)

    # 기간 슬라이스
    df = df[(df["date"] >= pd.to_datetime(start).normalize()) &
            (df["date"] <= pd.to_datetime(end).normalize())]

    # In‑Transit / WIP 노출 옵션
    if not show_in_transit:
        df = df[df["center"] != "In-Transit"]

    if centers:
        normalized_centers = [
            "태광KR" if str(center) == "WIP" else str(center)
            for center in centers
        ]
        df = df[df["center"].isin(normalized_centers)]
    if skus:
        df = df[df["resource_code"].isin([str(sku) for sku in skus])]

    wip_source = pd.DataFrame()
    if show_wip:
        wip_source = df[df["is_wip"]].copy()

    base_df = df[~df["is_wip"]].copy()

    if base_df.empty and (not show_wip or wip_source.empty):
        st.info("선택 조건에 해당하는 라인이 없습니다.")
        return

    # 라벨 생성: SKU @ Center
    plot_df = base_df.copy()
    if not plot_df.empty:
        plot_df["label"] = plot_df["resource_code"] + " @ " + plot_df["center"].astype(str)
    else:
        plot_df = pd.DataFrame(columns=["date", "stock_qty", "label"])

    # 기본 step line
    if plot_df.empty:
        fig = go.Figure()
        fig.update_layout(title=title)
    else:
        fig = px.line(
            plot_df.sort_values(["label", "date"]),
            x="date",
            y="stock_qty",
            color="label",
            line_shape="hv",
            render_mode="svg",
            title=title,
        )
        fig.update_traces(
            mode="lines",
            hovertemplate="날짜: %{x|%Y-%m-%d}<br>재고: %{y:,} EA<br>%{fullData.name}<extra></extra>",
        )

    # SKU별 고정 색, 상태(In‑Transit/WIP) 별 스타일
    color_labels: list[str] = []
    if not plot_df.empty:
        color_labels.extend(plot_df["label"].unique().tolist())
    if show_wip and not wip_source.empty:
        wip_skus_for_colors = sorted({str(v) for v in wip_source["resource_code"].unique()})
        color_labels.extend([f"{sku} @ 태광KR" for sku in wip_skus_for_colors])

    sku_colors = _sku_color_map(color_labels)
    for tr in fig.data:
        name = tr.name or ""
        sku, center = (name.split(" @ ", 1) + [""])[:2]
        color = sku_colors.get(sku, _STEP_PALETTE[0])

        # 상태/센터별 선 스타일
        if center == "In-Transit":
            tr.update(line=dict(color=color, dash="dot", width=2.0), opacity=0.85)
        else:
            tr.update(line=dict(color=color, dash="solid", width=2.2), opacity=0.95)

    wip_plot: Optional[pd.DataFrame] = None
    if show_wip and not wip_source.empty:
        wip_skus = list(dict.fromkeys([str(v) for v in (skus or [])]))
        if not wip_skus:
            wip_skus = sorted({str(v) for v in wip_source["resource_code"].unique()})
        wip_pivot = (
            wip_source.groupby(["date", "resource_code"])["stock_qty"].sum().unstack("resource_code")
        )
        wip_pivot = wip_pivot.reindex(columns=wip_skus, fill_value=0.0).sort_index()
        if not wip_pivot.empty:
            wip_plot = _safe_dataframe(wip_pivot.round(0), columns=wip_skus)
            if not wip_plot.empty:
                wip_plot.index = _ensure_naive_index(wip_plot.index)

    if wip_plot is not None and not wip_plot.empty:
        for sku in wip_plot.columns:
            series = wip_plot.get(sku)
            if series is None:
                continue
            if isinstance(series, pd.DataFrame):
                if series.empty:
                    continue
                series = series.iloc[:, 0]
            if not isinstance(series, pd.Series):
                series = pd.Series(series, index=wip_plot.index)
            numeric = pd.to_numeric(series, errors="coerce").dropna()
            if numeric.empty:
                continue
            color = sku_colors.get(sku, _STEP_PALETTE[0])
            fig.add_scatter(
                x=numeric.index,
                y=numeric,
                mode="lines",
                name=f"{sku} 태광KR 생산중",
                line=dict(color=color, dash="dot", width=2.0),
                hovertemplate="날짜: %{x|%Y-%m-%d}<br>재고: %{y:,} EA<br>%{fullData.name}<extra></extra>",
            )

    # 오늘 세로선
    if today is not None:
        t = pd.to_datetime(today).normalize()
        # Plotly의 add_vline은 Pandas Timestamp를 직접 사용할 경우
        # annotation 위치 계산 과정에서 Timestamp 덧셈이 발생해
        # TypeError가 발생할 수 있다. 이를 방지하기 위해
        # Python datetime 객체로 변환해 전달한다.
        t_dt = t.to_pydatetime()
        fig.add_shape(
            type="line",
            x0=t_dt,
            x1=t_dt,
            xref="x",
            y0=0,
            y1=1,
            yref="paper",
            line=dict(color="crimson", dash="dot", width=1.5),
        )
        fig.add_annotation(
            x=t_dt,
            xref="x",
            y=1.0,
            yref="paper",
            yshift=8,
            text="오늘",
            showarrow=False,
            font=dict(color="crimson"),
        )

    fig.update_layout(
        hovermode="x unified",
        xaxis_title="날짜",
        yaxis_title="재고 (EA)",
        legend_title_text="SKU @ Center / 생산중",
        margin=dict(l=10, r=10, t=60, b=10),
        height=520,
    )

    # 라벨 겹침 완화: 상단 캡션으로 설명 이동 (Streamlit UI에서 처리)
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})


def render_sku_summary_cards(*args: object, **kwargs: object):
    """Expose the KPI card renderer via this module for compatibility."""

    return _render_sku_summary_cards(*args, **kwargs)

