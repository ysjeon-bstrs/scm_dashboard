# scm_dashboard_v5/ui/charts.py
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd
import streamlit as st

from center_alias import normalize_center_value

try:  # Plotly는 선택적 의존성이므로 임포트 실패를 허용한다.
    import plotly.express as px  # type: ignore
    import plotly.graph_objects as go  # type: ignore
    from plotly.subplots import make_subplots  # type: ignore
except ImportError as _plotly_err:  # pragma: no cover - 의존성 결손 환경만 재현 가능
    px = None  # type: ignore[assignment]
    go = None  # type: ignore[assignment]
    make_subplots = None  # type: ignore[assignment]
    _PLOTLY_IMPORT_ERROR = _plotly_err
else:
    _PLOTLY_IMPORT_ERROR = None

from scm_dashboard_v5.ui.kpi import render_sku_summary_cards as _render_sku_summary_cards

if TYPE_CHECKING:
    from scm_dashboard_v5.forecast import AmazonForecastContext

# ---------------- Palette (SKU -> Color) ----------------
# 계단식 차트와 최대한 비슷한 톤(20+색)
PALETTE = [
    "#4E79A7","#F28E2B","#E15759","#76B7B2","#59A14F",
    "#EDC948","#B07AA1","#FF9DA7","#9C755F","#BAB0AC",
    "#1F77B4","#FF7F0E","#2CA02C","#D62728","#9467BD",
    "#8C564B","#E377C2","#7F7F7F","#BCBD22","#17BECF",
]


def _hex_to_rgb(hx: str) -> Tuple[int, int, int]:
    hx = hx.lstrip("#")
    if len(hx) == 3:
        hx = "".join(ch * 2 for ch in hx)
    return tuple(int(hx[i : i + 2], 16) for i in (0, 2, 4))


def _rgb_to_hex(rgb: Tuple[float, float, float]) -> str:
    r, g, b = [max(0, min(255, int(round(v)))) for v in rgb]
    return f"#{r:02X}{g:02X}{b:02X}"


def _tint(hex_color: str, factor: float) -> str:
    r, g, b = _hex_to_rgb(hex_color)
    if factor >= 1.0:
        r = r + (255 - r) * (factor - 1.0)
        g = g + (255 - g) * (factor - 1.0)
        b = b + (255 - b) * (factor - 1.0)
    else:
        r = r * factor
        g = g * factor
        b = b * factor
    return _rgb_to_hex((r, g, b))


CENTER_SHADE: Dict[str, float] = {
    "태광KR": 0.85,
    "AMZUS": 1.00,
    "CJ서부US": 1.15,
    "품고KR": 0.90,
    "AcrossBUS": 1.10,
    "SBSPH": 1.05,
    "SBSSG": 1.05,
    "SBSMY": 1.05,
}
DEFAULT_SHADE_STEP = 0.10


def _shade_for(center: str, index: int) -> float:
    if center in CENTER_SHADE:
        return CENTER_SHADE[center]
    if index <= 0:
        return 1.0
    step = ((index + 1) // 2) * DEFAULT_SHADE_STEP
    if index % 2 == 1:
        return 1.0 + step
    return max(0.4, 1.0 - step)

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


def _normalize_inventory_frame(
    df: Optional[pd.DataFrame], *, default_center: str | None = None
) -> pd.DataFrame:
    cols = ["date", "center", "resource_code", "stock_qty"]
    if df is None or df.empty:
        return pd.DataFrame(columns=cols)

    frame = df.copy()
    rename_map = {str(col).lower(): col for col in frame.columns}

    if "date" not in frame.columns and "snapshot_date" in rename_map:
        frame = frame.rename(columns={rename_map["snapshot_date"]: "date"})
    if "center" not in frame.columns and "center" in rename_map:
        frame = frame.rename(columns={rename_map["center"]: "center"})
    if "resource_code" not in frame.columns and "resource_code" in rename_map:
        frame = frame.rename(columns={rename_map["resource_code"]: "resource_code"})
    if "stock_qty" not in frame.columns and "stock" in rename_map:
        frame = frame.rename(columns={rename_map["stock"]: "stock_qty"})

    frame["date"] = pd.to_datetime(frame.get("date"), errors="coerce").dt.normalize()
    if "center" in frame.columns:
        frame["center"] = frame["center"].astype(str)
    elif default_center is not None:
        frame["center"] = default_center
    else:
        frame["center"] = ""

    if "resource_code" in frame.columns:
        frame["resource_code"] = frame["resource_code"].astype(str)
    else:
        frame["resource_code"] = ""

    frame["stock_qty"] = pd.to_numeric(frame.get("stock_qty"), errors="coerce").fillna(0)

    return frame[cols]


def _normalize_sales_frame(
    df: Optional[pd.DataFrame], *, default_center: str | None = None
) -> pd.DataFrame:
    cols = ["date", "center", "resource_code", "sales_ea"]
    if df is None or df.empty:
        return pd.DataFrame(columns=cols)

    frame = df.copy()
    rename_map = {str(col).lower(): col for col in frame.columns}

    if "date" not in frame.columns and "snapshot_date" in rename_map:
        frame = frame.rename(columns={rename_map["snapshot_date"]: "date"})
    if "center" not in frame.columns and "center" in rename_map:
        frame = frame.rename(columns={rename_map["center"]: "center"})
    if "resource_code" not in frame.columns and "resource_code" in rename_map:
        frame = frame.rename(columns={rename_map["resource_code"]: "resource_code"})
    if "sales_ea" not in frame.columns:
        candidate = None
        for col in frame.columns:
            if str(col).lower().endswith("sales_ea") or "sales" in str(col).lower():
                candidate = col
                break
        if candidate is not None:
            frame = frame.rename(columns={candidate: "sales_ea"})

    frame["date"] = pd.to_datetime(frame.get("date"), errors="coerce").dt.normalize()
    if "center" in frame.columns:
        frame["center"] = frame["center"].astype(str)
    elif default_center is not None:
        frame["center"] = default_center
    else:
        frame["center"] = ""

    if "resource_code" in frame.columns:
        frame["resource_code"] = frame["resource_code"].astype(str)
    else:
        frame["resource_code"] = ""

    frame["sales_ea"] = pd.to_numeric(frame.get("sales_ea"), errors="coerce").fillna(0)
    frame = frame.dropna(subset=["date"])

    return frame[cols]


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
            cmap[s] = PALETTE[i % len(PALETTE)]
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


def _empty_sales_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["date", "resource_code", "sales_ea"])


def _sales_forecast_ma(
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

    series = (
        history.set_index("date")[value_column]
        .asfreq("D")
        .fillna(0.0)
    )
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


def _sales_from_snapshot_raw(
    snap_raw: Optional[pd.DataFrame],
    centers: Sequence[str],
    skus: Sequence[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    debug: Optional[dict[str, object]] = None,
) -> pd.DataFrame:
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
        return _empty_sales_frame()

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
                centers_for_debug = (
                    df["center"].dropna().astype(str).unique().tolist()
                )
            debug.clear()
            debug.update(
                {
                    "rows_before_center_filter": len(df),
                    "rows_after_center_filter": len(df),
                    "snapshot_centers": sorted(centers_for_debug),
                    "warning": "missing fba_output_stock column",
                }
            )
        return _empty_sales_frame()

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
        return _empty_sales_frame()

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
            df["center"].dropna().astype(str).unique().tolist()
            if not df.empty
            else []
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
        return _empty_sales_frame()

    if skus:
        sku_set = {str(s) for s in skus if str(s).strip()}
        if sku_set:
            df = df[df["resource_code"].astype(str).isin(sku_set)]

    if df.empty:
        return _empty_sales_frame()

    start_norm = pd.to_datetime(start).normalize()
    end_norm = pd.to_datetime(end).normalize()

    df = df[(df["date"] >= start_norm) & (df["date"] <= end_norm)]
    if df.empty:
        return _empty_sales_frame()

    grouped = (
        df.groupby(["date", "resource_code"], as_index=False)["fba_output_stock"].sum()
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


def _sales_forecast_from_inventory_projection(
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
        chunk["date"] = pd.to_datetime(chunk.get("date"), errors="coerce").dt.normalize()
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
        chunk["stock_qty"] = pd.to_numeric(chunk.get("stock_qty"), errors="coerce").fillna(0.0)
        chunk["__source"] = label
        frames.append(chunk)

    if not frames:
        return pd.DataFrame(columns=["date", "resource_code", "sales_ea"])

    combined = pd.concat(frames, ignore_index=True)

    if "__source" in combined.columns:
        combined["__priority"] = combined["__source"].map({"actual": 0, "forecast": 1}).fillna(1)
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
    combined = combined[(combined["date"] >= start_norm) & (combined["date"] <= end_norm)]
    if combined.empty:
        return pd.DataFrame(columns=["date", "resource_code", "sales_ea"])

    pivot = (
        combined.groupby(["date", "resource_code"])["stock_qty"].sum().unstack("resource_code")
    )
    if pivot.empty:
        return pd.DataFrame(columns=["date", "resource_code", "sales_ea"])

    pivot = pivot.reindex(columns=skus_norm, fill_value=0.0)
    full_index = pd.date_range(start_norm, end_norm, freq="D")
    pivot = pivot.reindex(full_index).sort_index()
    pivot = pivot.ffill().fillna(0.0)

    diff = pivot.diff()
    sales = (-diff).clip(lower=0.0)

    # Once the stock reaches zero we clamp subsequent sales to zero.  This
    # prevents tiny negative diffs introduced by floating point noise from
    # leaking into the forecast bars after depletion.
    for sku in sales.columns:
        stock_series = pivot[sku]
        zero_dates = stock_series.index[stock_series <= 0]
        if len(zero_dates) > 0:
            first_zero = zero_dates[0]
            sales.loc[sales.index >= first_zero, sku] = 0.0

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


def _sales_from_snapshot_decays(
    snap_like: Optional[pd.DataFrame],
    centers: Sequence[str],
    skus: Sequence[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    if snap_like is None or snap_like.empty:
        return _empty_sales_frame()

    centers_list = [str(c) for c in centers if str(c).strip()]
    skus_list = [str(s) for s in skus if str(s).strip()]
    if not centers_list or not skus_list:
        return _empty_sales_frame()

    matrix = _sales_from_snapshot(
        snap_like,
        centers=centers_list,
        skus=skus_list,
        start=start,
        end=end,
    )

    if matrix is None or matrix.empty:
        return _empty_sales_frame()

    tidy = (
        matrix.reset_index()
        .melt(id_vars="date", var_name="resource_code", value_name="sales_ea")
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


def _total_inventory_series(
    inv_actual: pd.DataFrame,
    inv_forecast: pd.DataFrame,
    *,
    sku: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.Series:
    """Aggregate actual/forecast inventory for a SKU across centres."""

    start_norm = pd.to_datetime(start).normalize()
    end_norm = pd.to_datetime(end).normalize()
    if pd.isna(start_norm) or pd.isna(end_norm) or end_norm < start_norm:
        return pd.Series(dtype=float)

    sku_str = str(sku)
    frames: list[pd.DataFrame] = []
    for frame in (inv_actual, inv_forecast):
        if frame is None or frame.empty:
            continue
        if "resource_code" not in frame.columns or "date" not in frame.columns:
            continue
        chunk = frame.copy()
        chunk["resource_code"] = chunk.get("resource_code", "").astype(str)
        chunk = chunk[chunk["resource_code"] == sku_str]
        if chunk.empty:
            continue
        chunk["date"] = pd.to_datetime(chunk.get("date"), errors="coerce").dt.normalize()
        chunk = chunk.dropna(subset=["date"])
        if chunk.empty:
            continue
        qty = pd.to_numeric(chunk.get("stock_qty"), errors="coerce").fillna(0.0)
        chunk = chunk.assign(stock_qty=qty)
        frames.append(chunk[["date", "stock_qty"]])

    if not frames:
        # When no inventory data exists for the SKU we should not fabricate a
        # zero-filled timeline. Returning an empty, date-indexed series allows
        # downstream callers to treat the situation as "no data" instead of
        # "out of stock", so forecasts remain visible until real inventory
        # reaches zero.
        empty_index = pd.DatetimeIndex([], name="date")
        return pd.Series(dtype=float, index=empty_index)

    combined = pd.concat(frames, ignore_index=True)
    combined = (
        combined.groupby("date")["stock_qty"].sum().sort_index()
    )

    index = pd.date_range(start_norm, end_norm, freq="D")
    if index.empty:
        return pd.Series(dtype=float)

    combined = combined.reindex(index)
    if combined.notna().any():
        combined = combined.ffill()
        combined = combined.bfill()
    combined = combined.fillna(0.0)
    combined.index.name = "date"
    return combined.astype(float)


def _trim_sales_forecast_to_inventory(
    forecast_df: pd.DataFrame,
    inv_actual: pd.DataFrame,
    inv_forecast: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    forecast_start: pd.Timestamp,
) -> pd.DataFrame:
    """Remove forecast rows that extend beyond the first stock-out date."""

    if forecast_df is None or forecast_df.empty:
        return forecast_df

    forecast = forecast_df.copy()
    forecast["date"] = pd.to_datetime(forecast.get("date"), errors="coerce").dt.normalize()
    forecast = forecast.dropna(subset=["date"])
    if forecast.empty:
        return forecast

    trimmed_frames: list[pd.DataFrame] = []
    for sku, group in forecast.groupby("resource_code"):
        inv_series = _total_inventory_series(
            inv_actual,
            inv_forecast,
            sku=sku,
            start=start,
            end=end,
        )
        if inv_series.empty:
            trimmed_frames.append(group)
            continue

        zero_candidates = inv_series.loc[inv_series.index >= forecast_start]
        zero_dates = zero_candidates[zero_candidates <= 0]
        if zero_dates.empty:
            trimmed_frames.append(group)
            continue

        cutoff = zero_dates.index[0]
        trimmed = group[group["date"] <= cutoff]
        trimmed_frames.append(trimmed)

    if not trimmed_frames:
        return forecast.iloc[0:0]

    result = pd.concat(trimmed_frames, ignore_index=True)
    result = result.sort_values(["resource_code", "date"]).reset_index(drop=True)
    return result

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

def _sku_color_map(skus):
    m = {}
    for i, s in enumerate(skus):
        m[s] = PALETTE[i % len(PALETTE)]
    return m




def _clamped_forecast_series(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    base_stock: float,
    inbound_by_day: dict[pd.Timestamp, float],
    daily_demand: float,
) -> tuple[pd.Series, pd.Series]:
    """Return paired (sales, inventory) series respecting remaining stock."""

    if pd.isna(start_date) or pd.isna(end_date) or end_date < start_date:
        empty_index = pd.DatetimeIndex([], dtype="datetime64[ns]")
        return pd.Series(dtype=float, index=empty_index), pd.Series(dtype=float, index=empty_index)

    idx = pd.date_range(start_date, end_date, freq="D")
    fcst_sales = pd.Series(0.0, index=idx, dtype=float)
    inv = pd.Series(np.nan, index=idx, dtype=float)

    remain = float(base_stock)
    demand = max(0.0, float(daily_demand))

    for d in idx:
        inbound_qty = float(inbound_by_day.get(pd.to_datetime(d), 0.0))
        remain += inbound_qty

        sell = min(demand, max(remain, 0.0))
        fcst_sales.loc[d] = sell
        remain -= sell

        inv.loc[d] = max(remain, 0.0)

        if remain <= 0:
            if d != idx[-1]:
                remainder = slice(d + pd.Timedelta(days=1), idx[-1])
                fcst_sales.loc[remainder] = 0.0
                inv.loc[remainder] = 0.0
            break

    return fcst_sales, inv


def render_amazon_sales_vs_inventory(ctx: "AmazonForecastContext") -> None:
    """Draw the Amazon US panel using actual sales and clamped forecasts."""

    if not _ensure_plotly_available() or go is None:
        return

    if ctx is None:
        st.info("아마존 데이터가 없습니다.")
        return

    skus = [str(sku) for sku in getattr(ctx, "skus", []) if str(sku).strip()]
    if not skus:
        st.info("SKU를 선택하세요.")
        return

    target_centers = [normalize_center_value(c) for c in getattr(ctx, "centers", [])]
    target_centers = [c for c in target_centers if c]
    if not target_centers:
        target_centers = ["AMZUS"]

    snap_long = getattr(ctx, "snapshot_long", pd.DataFrame()).copy()
    if snap_long.empty:
        st.info("AMZUS 데이터가 없습니다.")
        return

    cols_lower = {str(c).strip().lower(): c for c in snap_long.columns}
    date_col = cols_lower.get("date") or cols_lower.get("snapshot_date")
    center_col = cols_lower.get("center")
    sku_col = cols_lower.get("resource_code") or cols_lower.get("sku")
    stock_col = cols_lower.get("stock_qty") or cols_lower.get("qty")
    sales_col = cols_lower.get("sales_qty") or cols_lower.get("sale_qty")

    if not all([date_col, center_col, sku_col, stock_col]):
        st.warning("정제 스냅샷 형식이 예상과 다릅니다.")
        return

    rename_map = {
        date_col: "date",
        center_col: "center",
        sku_col: "resource_code",
        stock_col: "stock_qty",
    }
    if sales_col:
        rename_map[sales_col] = "sales_qty"

    df = snap_long.rename(columns=rename_map).copy()
    if "sales_qty" not in df.columns:
        df["sales_qty"] = 0

    df["date"] = pd.to_datetime(df.get("date"), errors="coerce").dt.normalize()
    df = df.dropna(subset=["date"])
    df["center"] = df.get("center", "").astype(str)
    df["resource_code"] = df.get("resource_code", "").astype(str)
    df["stock_qty"] = pd.to_numeric(df.get("stock_qty"), errors="coerce").fillna(0)
    df["sales_qty"] = pd.to_numeric(df.get("sales_qty"), errors="coerce").fillna(0)

    df = df[
        df["center"].isin(target_centers)
        & df["resource_code"].isin(skus)
    ].copy()

    if df.empty:
        st.info("AMZUS 데이터가 없습니다.")
        return

    start = pd.to_datetime(getattr(ctx, "start", df["date"].min())).normalize()
    end = pd.to_datetime(getattr(ctx, "end", df["date"].max())).normalize()
    today = pd.to_datetime(getattr(ctx, "today", pd.Timestamp.today())).normalize()
    lookback_days = int(getattr(ctx, "lookback_days", 28) or 28)
    lookback_days = max(1, lookback_days)
    promo_multiplier = float(getattr(ctx, "promotion_multiplier", 1.0) or 1.0)
    if not np.isfinite(promo_multiplier) or promo_multiplier <= 0:
        promo_multiplier = 1.0

    df = df[
        (df["date"] >= start - pd.Timedelta(days=lookback_days + 2))
        & (df["date"] <= end)
    ]

    df["kind"] = np.where(df["date"] <= today, "actual", "future")

    inv_actual = (
        df[df["kind"] == "actual"]
        .groupby(["date", "resource_code"], as_index=False)["stock_qty"]
        .sum()
    )

    sales_actual = (
        df[df["kind"] == "actual"]
        .groupby(["date", "resource_code"], as_index=False)["sales_qty"]
        .sum()
    )

    avg_demand_by_sku: dict[str, float] = {}
    last_stock_by_sku: dict[str, float] = {}
    for sku, group in df.groupby("resource_code"):
        history = group[group["date"] <= today].sort_values("date")
        tail = history.tail(lookback_days)
        avg = float(tail["sales_qty"].mean() or 0.0)
        avg_demand_by_sku[sku] = max(0.0, avg)

        if not history.empty:
            last_stock_by_sku[sku] = float(history.iloc[-1]["stock_qty"])
        else:
            last_stock_by_sku[sku] = 0.0

    moves_df = getattr(ctx, "moves", pd.DataFrame()).copy()
    if not moves_df.empty:
        mv_cols = {str(c).lower(): c for c in moves_df.columns}
        rename_moves = {mv_cols.get("event_date", "event_date"): "event_date"}
        for name in ["to_center", "resource_code", "qty_ea"]:
            if name in mv_cols:
                rename_moves[mv_cols[name]] = name
        moves_df = moves_df.rename(columns=rename_moves)
        moves_df["event_date"] = pd.to_datetime(
            moves_df.get("event_date"), errors="coerce"
        ).dt.normalize()
        moves_df = moves_df.dropna(subset=["event_date"])
        moves_df["to_center"] = moves_df.get("to_center", "").astype(str)
        moves_df["resource_code"] = moves_df.get("resource_code", "").astype(str)
        moves_df["qty_ea"] = pd.to_numeric(moves_df.get("qty_ea"), errors="coerce").fillna(0)
        moves_df = moves_df[
            moves_df["to_center"].isin(target_centers)
            & moves_df["resource_code"].isin(skus)
            & (moves_df["event_date"] >= today + pd.Timedelta(days=1))
            & (moves_df["event_date"] <= end)
        ]
    else:
        moves_df = pd.DataFrame(columns=["event_date", "resource_code", "qty_ea"])

    inbound = (
        moves_df.groupby(["resource_code", "event_date"], as_index=False)["qty_ea"].sum()
        if not moves_df.empty
        else pd.DataFrame(columns=["resource_code", "event_date", "qty_ea"])
    )

    fcst_start = max(today + pd.Timedelta(days=1), start)
    fcst_sales_rows: list[pd.DataFrame] = []
    fcst_inv_rows: list[pd.DataFrame] = []

    if fcst_start <= end:
        for sku in skus:
            base_stock = float(last_stock_by_sku.get(sku, 0.0))
            inbound_map = {
                pd.to_datetime(day): float(qty)
                for day, qty in inbound[inbound["resource_code"] == sku][["event_date", "qty_ea"]]
                .itertuples(index=False, name=None)
            }

            daily_demand = avg_demand_by_sku.get(sku, 0.0) * promo_multiplier
            fcst_sales, inv_series = _clamped_forecast_series(
                start_date=fcst_start,
                end_date=end,
                base_stock=base_stock,
                inbound_by_day=inbound_map,
                daily_demand=daily_demand,
            )

            if not fcst_sales.empty:
                fcst_sales_rows.append(
                    pd.DataFrame(
                        {
                            "date": fcst_sales.index,
                            "resource_code": sku,
                            "sales_qty": fcst_sales.values,
                        }
                    )
                )
            if not inv_series.empty:
                fcst_inv_rows.append(
                    pd.DataFrame(
                        {
                            "date": inv_series.index,
                            "resource_code": sku,
                            "stock_qty": inv_series.values,
                        }
                    )
                )

    sales_forecast = (
        pd.concat(fcst_sales_rows, ignore_index=True)
        if fcst_sales_rows
        else pd.DataFrame(columns=["date", "resource_code", "sales_qty"])
    )

    inv_forecast = (
        pd.concat(fcst_inv_rows, ignore_index=True)
        if fcst_inv_rows
        else pd.DataFrame(columns=["date", "resource_code", "stock_qty"])
    )

    show_ma7 = bool(getattr(ctx, "show_ma7", True))
    if show_ma7 and not sales_actual.empty:
        ma = (
            sales_actual.set_index("date")
            .groupby("resource_code")["sales_qty"]
            .apply(lambda s: s.rolling(7, min_periods=1).mean())
            .reset_index()
            .rename(columns={"sales_qty": "sales_ma7"})
        )
    else:
        ma = pd.DataFrame(columns=["date", "resource_code", "sales_ma7"])

    fig = go.Figure()
    colors = _sku_colors(skus)

    if not sales_actual.empty:
        for sku, group in sales_actual.groupby("resource_code"):
            fig.add_bar(
                x=group["date"],
                y=group["sales_qty"],
                name=f"{sku} 판매(실측)",
                marker_color=colors.get(sku, "#6BA3FF"),
                opacity=0.95,
            )

    if not sales_forecast.empty:
        for sku, group in sales_forecast.groupby("resource_code"):
            fig.add_bar(
                x=group["date"],
                y=group["sales_qty"],
                name=f"{sku} 판매(예측)",
                marker=dict(color=colors.get(sku, "#6BA3FF"), pattern=dict(shape="/")),
                opacity=0.6,
            )

    if not inv_actual.empty:
        for sku, group in inv_actual.groupby("resource_code"):
            fig.add_trace(
                go.Scatter(
                    x=group["date"],
                    y=group["stock_qty"],
                    mode="lines",
                    name=f"{sku} 재고(실측)",
                    line=dict(color="#F28E2B", width=2),
                    yaxis="y2",
                )
            )

    if not inv_forecast.empty:
        for sku, group in inv_forecast.groupby("resource_code"):
            fig.add_trace(
                go.Scatter(
                    x=group["date"],
                    y=group["stock_qty"],
                    mode="lines",
                    name=f"{sku} 재고(예측)",
                    line=dict(color="#F28E2B", width=2, dash="dot"),
                    yaxis="y2",
                )
            )

    if show_ma7 and not ma.empty:
        for sku, group in ma.groupby("resource_code"):
            fig.add_trace(
                go.Scatter(
                    x=group["date"],
                    y=group["sales_ma7"],
                    mode="lines",
                    name=f"{sku} 판매 7일 평균",
                    line=dict(color=colors.get(sku, "#6BA3FF"), dash="dash"),
                )
            )

    fig.add_vline(x=today, line_color="crimson", line_dash="dash", line_width=2)

    fig.update_layout(
        title="Amazon US 일별 판매 vs. 재고",
        barmode="stack",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=30, r=20, t=40, b=20),
        hovermode="x unified",
        xaxis=dict(title="Date"),
        yaxis=dict(title="판매량 (EA/Day)"),
        yaxis2=dict(
            title="재고 (EA)",
            overlaying="y",
            side="right",
            showgrid=False,
            zeroline=False,
            showline=False,
            tickfont=dict(color="#666"),
        ),
    )

    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})


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


def _step_sku_color_map(labels: list[str]) -> dict[str, str]:
    """'SKU @ Center' 라벨들에서 SKU만 뽑아 SKU별 색을 고정 매핑."""

    m: dict[str, str] = {}
    i = 0
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
    **kwargs,
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

    sku_colors = _step_sku_color_map(color_labels)
    sku_center_seen: Dict[str, Dict[str, int]] = {}
    for tr in fig.data:
        name = tr.name or ""
        if " @ " not in name:
            continue
        sku, center = name.split(" @ ", 1)
        base_color = sku_colors.get(sku, _STEP_PALETTE[0])

        if center == "In-Transit":
            tr.update(line=dict(color=base_color, dash="dot", width=2.0), opacity=0.85)
            continue

        centers_for_sku = sku_center_seen.setdefault(sku, {})
        if center not in centers_for_sku:
            centers_for_sku[center] = len(centers_for_sku)
        center_index = centers_for_sku[center]
        shade = _shade_for(center, center_index)
        color = _tint(base_color, shade)
        tr.update(line=dict(color=color, dash="solid", width=2.4), opacity=0.95)

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

