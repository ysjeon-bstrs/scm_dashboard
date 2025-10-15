# scm_dashboard_v5/ui/charts.py
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd
import streamlit as st

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


def render_amazon_sales_vs_inventory(ctx: "AmazonForecastContext") -> None:
    """Draw the Amazon sales and inventory panel from a shared forecast context."""

    if not _ensure_plotly_available() or go is None or make_subplots is None:
        return

    if ctx is None:
        st.info("아마존 데이터가 없습니다.")
        return

    frames = [
        getattr(ctx, "inv_actual", pd.DataFrame()),
        getattr(ctx, "inv_forecast", pd.DataFrame()),
        getattr(ctx, "sales_hist", pd.DataFrame()),
        getattr(ctx, "sales_forecast", pd.DataFrame()),
        getattr(ctx, "sales_ma7", pd.DataFrame()),
    ]

    if all(frame is None or frame.empty for frame in frames):
        st.info("아마존 데이터가 없습니다.")
        return

    colors = _sku_colors(ctx.skus)
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    actual = ctx.inv_actual.copy()
    actual["date"] = pd.to_datetime(actual.get("date"), errors="coerce")
    actual = actual.dropna(subset=["date"])
    actual["stock_qty"] = pd.to_numeric(actual.get("stock_qty"), errors="coerce").fillna(0)

    forecast = ctx.inv_forecast.copy()
    forecast["date"] = pd.to_datetime(forecast.get("date"), errors="coerce")
    forecast = forecast.dropna(subset=["date"])
    forecast["stock_qty"] = pd.to_numeric(forecast.get("stock_qty"), errors="coerce").fillna(0)

    for (center, sku), group in actual.groupby(["center", "resource_code"], dropna=True):
        group = group.sort_values("date")
        if group.empty:
            continue
        color = colors.get(sku, PALETTE[0])
        fig.add_trace(
            go.Scatter(
                x=group["date"],
                y=group["stock_qty"],
                name=f"{sku} 재고(실측)",
                mode="lines",
                line=dict(color=color, width=2.4),
                legendgroup=f"{sku}@{center}",
                hovertemplate=(
                    f"센터: {center}<br>날짜 %{{x|%Y-%m-%d}}<br>재고 %{{y:,}} EA<extra></extra>"
                ),
            ),
            secondary_y=True,
        )

        future = forecast[
            (forecast["center"] == center)
            & (forecast["resource_code"] == sku)
        ].sort_values("date")

        if not future.empty:
            past = group[group["date"] <= ctx.today]
            if not past.empty:
                last_actual = past.iloc[-1]
                if future["date"].min() > ctx.today:
                    bridge = pd.DataFrame(
                        {
                            "date": [ctx.today],
                            "center": [center],
                            "resource_code": [sku],
                            "stock_qty": [last_actual["stock_qty"]],
                        }
                    )
                    future = pd.concat([bridge, future], ignore_index=True)

            fig.add_trace(
                go.Scatter(
                    x=future["date"],
                    y=future["stock_qty"],
                    name=f"{sku} 재고(예측)",
                    mode="lines",
                    line=dict(color=color, width=2.0, dash="dot"),
                    legendgroup=f"{sku}@{center}",
                    hovertemplate=(
                        f"센터: {center}<br>날짜 %{{x|%Y-%m-%d}}<br>재고 %{{y:,}} EA<extra></extra>"
                    ),
                ),
                secondary_y=True,
            )

    hist = ctx.sales_hist.copy()
    hist["date"] = pd.to_datetime(hist.get("date"), errors="coerce").dt.normalize()
    hist = hist.dropna(subset=["date"])
    hist["sales_ea"] = pd.to_numeric(hist.get("sales_ea"), errors="coerce").fillna(0)
    hist = hist[(hist["date"] >= ctx.start) & (hist["date"] <= ctx.today)]
    hist_grouped = (
        hist.groupby(["date", "resource_code"], as_index=False)["sales_ea"].sum()
        if not hist.empty
        else pd.DataFrame(columns=["date", "resource_code", "sales_ea"])
    )

    future_sales = ctx.sales_forecast.copy()
    future_sales["date"] = pd.to_datetime(future_sales.get("date"), errors="coerce").dt.normalize()
    future_sales = future_sales.dropna(subset=["date"])
    future_sales["sales_ea"] = pd.to_numeric(
        future_sales.get("sales_ea"), errors="coerce"
    ).fillna(0)
    future_sales = future_sales[
        (future_sales["date"] > ctx.today) & (future_sales["date"] <= ctx.end)
    ]
    future_grouped = (
        future_sales.groupby(["date", "resource_code"], as_index=False)["sales_ea"].sum()
        if not future_sales.empty
        else pd.DataFrame(columns=["date", "resource_code", "sales_ea"])
    )

    for sku, group in hist_grouped.groupby("resource_code"):
        group = group.sort_values("date")
        if group.empty:
            continue
        fig.add_trace(
            go.Bar(
                x=group["date"],
                y=group["sales_ea"],
                name=f"{sku} 판매(실측)",
                marker_color=colors.get(sku, PALETTE[0]),
                opacity=0.85,
                hovertemplate="날짜 %{x|%Y-%m-%d}<br>판매 %{y:,} EA<extra></extra>",
            ),
            secondary_y=False,
        )

    for sku, group in future_grouped.groupby("resource_code"):
        group = group.sort_values("date")
        if group.empty:
            continue
        fig.add_trace(
            go.Bar(
                x=group["date"],
                y=group["sales_ea"],
                name=f"{sku} 판매(예측)",
                marker_color=colors.get(sku, PALETTE[0]),
                opacity=0.35,
                hovertemplate="날짜 %{x|%Y-%m-%d}<br>예상 판매 %{y:,} EA<extra></extra>",
            ),
            secondary_y=False,
        )

    ma7 = ctx.sales_ma7.copy()
    ma7["date"] = pd.to_datetime(ma7.get("date"), errors="coerce").dt.normalize()
    ma7 = ma7.dropna(subset=["date"])
    ma7 = ma7[(ma7["date"] >= ctx.start) & (ma7["date"] <= ctx.today)]
    ma7_grouped = (
        ma7.groupby(["date", "resource_code"], as_index=False)["sales_ea"].sum()
        if not ma7.empty
        else pd.DataFrame(columns=["date", "resource_code", "sales_ea"])
    )

    for sku, group in ma7_grouped.groupby("resource_code"):
        group = group.sort_values("date")
        if group.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=group["date"],
                y=group["sales_ea"],
                name=f"{sku} 판매 7일 평균",
                mode="lines",
                line=dict(color=colors.get(sku, PALETTE[0]), dash="dash"),
                hovertemplate="날짜 %{x|%Y-%m-%d}<br>판매 7일 평균 %{y:.1f} EA<extra></extra>",
            ),
            secondary_y=False,
        )

    fig.add_vline(x=ctx.today, line_color="red", line_dash="dash", opacity=0.85)

    fig.update_layout(
        barmode="stack",
        hovermode="x unified",
        margin=dict(l=10, r=10, t=40, b=20),
        legend=dict(orientation="h"),
        xaxis=dict(title="Date", range=[ctx.start, ctx.end]),
    )
    fig.update_yaxes(
        title_text="판매량 (EA/Day)",
        rangemode="tozero",
        secondary_y=False,
        gridcolor="rgba(0,0,0,0.08)",
    )
    fig.update_yaxes(
        title_text="재고 (EA)",
        rangemode="tozero",
        secondary_y=True,
        gridcolor="rgba(0,0,0,0.08)",
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

