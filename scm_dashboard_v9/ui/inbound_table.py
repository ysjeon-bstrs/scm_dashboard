"""
입고 예정 테이블 모듈 (Inbound Table Module)

스프레드시트 기반 입고 예정/운송 데이터를 읽기 쉬운 시각화 테이블로 렌더링합니다.

주요 기능:
- 인보이스별 SKU 그룹핑 및 요약
- ETA 색상 코딩 (빨강=지남, 초록=5일 이내, 회색=그 외, 주황=미확인)
- SKU 색상 팔레트 적용
- 운송 경로 시각화

산출물:
- build_inbound_table: 원본 데이터를 테이블 포맷으로 변환
- render_inbound_table: Streamlit 테이블 렌더링
"""

from __future__ import annotations

from datetime import date
from typing import Dict, Optional

import pandas as pd
import streamlit as st


def build_inbound_table(
    inbound_raw: pd.DataFrame, sku_color_map: Dict[str, str] = None
) -> pd.DataFrame:
    """
    입고 예정 원본 데이터를 읽기 쉬운 테이블 형식으로 변환합니다.

    Args:
        inbound_raw: 원본 입고 예정 데이터프레임
            필수 컬럼:
            - invoice_no: 주문번호/송장번호
            - from_country: 출발 국가
            - to_country: 도착 국가
            - to_center: 도착 센터
            - resource_code: SKU 코드
            - resource_name: 품명
            - qty_ea: 수량
            - carrier_mode: 운송모드
            - onboard_date: 출발일
            - pred_inbound_date: 예상 입고일 (ETA)
            - expected_inbound_date: 평균 리드타임 기반 예상 입고일

        sku_color_map: (사용 안 함, 하위 호환성 유지용)

    Returns:
        변환된 테이블 데이터프레임
            컬럼:
            - invoice_no: 주문번호
            - route: 경로 (예: "KR → US")
            - carrier_mode: 운송모드
            - sku_summary: SKU 요약 (예: "BA00021: 30,132ea 외 2종")
            - onboard_date: 출발일 (YYYY-MM-DD)
            - eta_text: ETA 표시 텍스트 (YYYY-MM-DD 또는 "미확인")
            - eta_color: ETA 색상 코드 ("red"/"green"/"gray"/"orange")
            - expected_inbound_date: 평균 리드타임 기반 예상 입고일 (YYYY-MM-DD)
            - sku_summary_html: SKU 요약 HTML (대표 SKU 강조)
            - _rep_sku: 대표 SKU 코드 (내부용)

    Examples:
        >>> import pandas as pd
        >>> raw_data = pd.DataFrame({
        ...     "invoice_no": ["INV001", "INV001", "INV002"],
        ...     "from_country": ["KR", "KR", "US"],
        ...     "to_country": ["US", "US", "KR"],
        ...     "resource_code": ["BA00021", "BA00022", "BA00023"],
        ...     "resource_name": ["제품A", "제품B", "제품C"],
        ...     "qty_ea": [1000, 500, 200],
        ...     "carrier_mode": ["특송", "특송", "해운"],
        ...     "onboard_date": ["2025-01-15", "2025-01-15", "2025-01-20"],
        ...     "pred_inbound_date": ["2025-01-20", "2025-01-20", "not_defined"]
        ... })
        >>> sku_map = {"BA00021": "#4E79A7", "BA00022": "#F28E2B", "BA00023": "#E15759"}
        >>> result = build_inbound_table(raw_data, sku_map)
        >>> result.loc[0, "invoice_no"]
        'INV001'
        >>> result.loc[0, "route"]
        'KR → US'

    Notes:
        - 인보이스별로 그룹핑하여 한 행으로 집계
        - 대표 SKU는 수량이 가장 많은 SKU (동률 시 코드 사전순)
        - ETA 색상 규칙:
          * 미확인/결측: orange
          * 과거: red
          * 오늘~5일 이내: green
          * 6일 이후: gray
    """
    # ========================================
    # 1단계: 데이터 복사 및 기본 검증
    # ========================================
    if inbound_raw.empty:
        return pd.DataFrame()

    df = inbound_raw.copy()

    if sku_color_map is None:
        sku_color_map = {}

    # 필수 컬럼 확인 (없으면 빈 컬럼 추가)
    required_cols = [
        "invoice_no",
        "from_country",
        "to_country",
        "to_center",
        "resource_code",
        "resource_name",
        "qty_ea",
        "carrier_mode",
        "onboard_date",
        "pred_inbound_date",
    ]

    for col in required_cols:
        if col not in df.columns:
            if col == "qty_ea":
                df[col] = 0
            else:
                df[col] = ""

    if "expected_inbound_date" not in df.columns:
        df["expected_inbound_date"] = pd.NaT

    # ========================================
    # 2단계: 날짜 및 수량 정규화
    # ========================================
    # onboard_date: 출발일
    df["onboard_date"] = pd.to_datetime(df["onboard_date"], errors="coerce")

    # pred_inbound_date: 예상 입고일 ("not_defined" → NaT)
    df["pred_inbound_date"] = df["pred_inbound_date"].replace("not_defined", pd.NaT)
    df["pred_inbound_date"] = pd.to_datetime(df["pred_inbound_date"], errors="coerce")

    # expected_inbound_date: 리드타임 기반 예상 입고일
    df["expected_inbound_date"] = pd.to_datetime(
        df.get("expected_inbound_date"), errors="coerce"
    )

    # qty_ea: 수량
    df["qty_ea"] = pd.to_numeric(df["qty_ea"], errors="coerce").fillna(0).astype(int)

    # ========================================
    # 3단계: 인보이스별 그룹핑 및 집계
    # ========================================
    if df.empty:
        return pd.DataFrame()

    today = pd.Timestamp.today().normalize()
    grouped = df.groupby("invoice_no", sort=False)

    agg_base = grouped.agg(
        from_country=("from_country", "first"),
        to_country=("to_country", "first"),
        carrier_mode=("carrier_mode", "first"),
        to_center=("to_center", "first"),
        onboard_min=("onboard_date", "min"),
        pred_min=("pred_inbound_date", "min"),
        expected_min=("expected_inbound_date", "min"),
    )

    totals = grouped["qty_ea"].sum().rename("_total_qty")
    sku_counts = grouped["resource_code"].nunique().rename("_sku_count")

    df_sorted = df.sort_values(
        ["invoice_no", "qty_ea", "resource_code"],
        ascending=[True, False, True],
    )
    top_rows = (
        df_sorted.drop_duplicates("invoice_no", keep="first")
        .set_index("invoice_no")
        .reindex(agg_base.index)
    )

    agg_base = (
        agg_base.join(totals)
        .join(sku_counts)
        .join(
            top_rows[["resource_code", "qty_ea"]].rename(
                columns={"resource_code": "_rep_sku", "qty_ea": "_top_qty"}
            )
        )
    )

    agg_base["from_country"] = agg_base["from_country"].fillna("").astype(str)
    agg_base["to_country"] = agg_base["to_country"].fillna("").astype(str)
    agg_base["carrier_mode"] = agg_base["carrier_mode"].fillna("").astype(str)
    agg_base["to_center"] = agg_base["to_center"].fillna("").astype(str)
    agg_base["_rep_sku"] = agg_base["_rep_sku"].fillna("").astype(str)
    agg_base["_top_qty"] = agg_base["_top_qty"].fillna(0).astype(int)
    agg_base["_total_qty"] = agg_base["_total_qty"].fillna(0).astype(int)
    agg_base["_others"] = (
        agg_base["_sku_count"].fillna(0).astype(int) - 1
    ).clip(lower=0)

    qty_text = agg_base["_top_qty"].apply(lambda x: f"{x:,}ea")
    agg_base["sku_summary"] = agg_base["_rep_sku"] + ": " + qty_text
    mask_others = agg_base["_others"] > 0
    agg_base.loc[mask_others, "sku_summary"] += (
        " 외 "
        + agg_base.loc[mask_others, "_others"].astype(int).astype(str)
        + "종"
    )

    agg_base["route"] = (
        agg_base["from_country"] + " → " + agg_base["to_country"]
    )

    onboard_str = agg_base["onboard_min"].apply(
        lambda ts: ts.strftime("%Y-%m-%d") if pd.notna(ts) else ""
    )
    expected_str = agg_base["expected_min"].apply(
        lambda ts: ts.strftime("%Y-%m-%d") if pd.notna(ts) else ""
    )

    eta_dates = agg_base["pred_min"]
    eta_text = eta_dates.dt.strftime("%Y-%m-%d")
    eta_text = eta_text.where(eta_dates.notna(), "미확인")

    diff_days = (eta_dates.dt.normalize() - today).dt.days
    eta_color = pd.Series("gray", index=agg_base.index)
    eta_color.loc[eta_dates.isna()] = "orange"
    eta_color.loc[(eta_dates.notna()) & (diff_days < 0)] = "red"
    eta_color.loc[(eta_dates.notna()) & (diff_days.between(0, 5))] = "green"

    out = pd.DataFrame(
        {
            "invoice_no": agg_base.index.astype(str),
            "route": agg_base["route"],
            "carrier_mode": agg_base["carrier_mode"],
            "sku_summary": agg_base["sku_summary"],
            "onboard_date": onboard_str,
            "eta_text": eta_text,
            "eta_color": eta_color,
            "expected_inbound_date": expected_str,
            "_rep_sku": agg_base["_rep_sku"],
            "_to_center": agg_base["to_center"],
            "_total_qty": agg_base["_total_qty"],
            "_pred_inbound_date": agg_base["pred_min"],
            "_expected_inbound_date": agg_base["expected_min"],
        }
    )

    summary = out.get("sku_summary", pd.Series(index=out.index, dtype=object)).fillna("")
    split_parts = summary.str.split(":", n=1, expand=True)
    left = split_parts[0].fillna("")
    if 1 in split_parts.columns:
        right = split_parts[1].fillna("")
    else:
        right = pd.Series("", index=out.index, dtype=object)

    colors = out.get("_rep_sku", pd.Series(index=out.index, dtype=object)).map(sku_color_map)
    colors = colors.fillna("#6b7280").astype(str)

    base_html = (
        '<span style="color:'
        + colors
        + ';font-weight:600">'
        + left.astype(str)
        + "</span>"
    )

    right = right.astype(str)
    right_mask = right != ""
    colon_right = right.copy()
    colon_right.loc[right_mask] = ":" + colon_right.loc[right_mask]
    colon_right.loc[~right_mask] = ""

    hidden = summary.astype(str)
    hidden_mask = hidden != ""
    hidden.loc[hidden_mask] = (
        '<span style="display:none">' + hidden.loc[hidden_mask] + "</span>"
    )
    hidden.loc[~hidden_mask] = ""

    out["sku_summary_html"] = base_html + colon_right + hidden

    # ========================================
    # 4단계: 정렬 (출발일 내림차순 - 최신 우선)
    # ========================================
    # onboard_date를 날짜로 변환하여 정렬
    out["_onboard_sort"] = pd.to_datetime(out["onboard_date"], errors="coerce")
    out = out.sort_values("_onboard_sort", ascending=False, na_position="last")
    out = out.drop(columns=["_onboard_sort"]).reset_index(drop=True)

    return out


def render_inbound_table(
    df: pd.DataFrame,
    title: str = "📦 입고 예정 현황 (Inbound Schedule)",
    height: int = 520,
    sku_color_map: dict = None,
) -> None:
    """
    입고 예정 테이블을 Streamlit dataframe으로 렌더링합니다 (개선된 UI).

    Args:
        df: build_inbound_table()의 출력 데이터프레임
        title: 테이블 제목 (기본: "📦 입고 예정 현황")
        height: 테이블 높이 (픽셀, 기본: 520)
        sku_color_map: SKU 색상 매핑 딕셔너리 (선택사항)

    Notes:
        - 행 높이 44-48px, compact design
        - 주문번호·경로만 볼드
        - ETA 색상만 상태별 변경 (빨강/초록/주황/회색)
        - SKU 요약의 대표 SKU만 팔레트 색으로 강조
    """
    # ========================================
    # 1단계: 데이터 유효성 검증
    # ========================================
    if df.empty:
        st.info("📭 입고 예정 데이터가 없습니다.")
        return

    if title:
        st.markdown(f"### {title}")

    if sku_color_map is None:
        sku_color_map = {}

    # ========================================
    # 2단계: ETA 색상 팔레트
    # ========================================
    PALETTE = {
        "red": "#ef4444",  # 빨강 (지연)
        "green": "#22c55e",  # 초록 (곧 도착)
        "gray": "#9ca3af",  # 회색 (6일 이후)
        "orange": "#f59e0b",  # 주황 (미확인)
    }

    def _eta_color(c):
        return PALETTE.get(c, "#374151")

    # ========================================
    # 3단계: SKU 요약 HTML 생성 (대표 SKU만 색상)
    # ========================================
    def sku_html(row):
        """대표 SKU만 색 강조, 나머지 텍스트는 기본색"""
        sku = row.get("_rep_sku", "")
        hexc = sku_color_map.get(sku, "#0ea5e9")  # 기본 파랑톤
        txt = str(row.get("sku_summary", ""))

        # "BA00021: 30,132ea 외 2종" → 대표 SKU 부분만 색상
        parts = txt.split(":", 1)
        if len(parts) == 2:
            sku_part = parts[0]
            rest = ":" + parts[1]
            return f'<span style="color:{hexc};font-weight:600">{sku_part}</span>{rest}'
        return txt

    # ========================================
    # 4단계: 데이터 준비
    # ========================================
    view = df.copy()
    view["SKU 요약"] = view.apply(sku_html, axis=1)

    # 컬럼 선택 및 이름 변경
    view = view.rename(
        columns={
            "invoice_no": "주문번호",
            "route": "경로",
            "carrier_mode": "운송모드",
            "onboard_date": "발송일",
            "eta_text": "예상 도착일",
            "expected_inbound_date": "예상 입고일",
        }
    )

    display_cols = [
        "주문번호",
        "경로",
        "운송모드",
        "SKU 요약",
        "발송일",
        "예상 도착일",
        "예상 입고일",
    ]
    view = view[[col for col in display_cols if col in view.columns]]

    # eta_color를 별도로 보관
    eta_colors = df["eta_color"].tolist()

    # ========================================
    # 5단계: Styler 적용
    # ========================================
    def apply_styles(row):
        """행별 스타일 적용"""
        styles = [""] * len(row)
        idx = row.name

        if idx >= len(eta_colors):
            return styles

        # ETA 색상만 적용
        if "예상 도착일" in view.columns:
            eta_idx = view.columns.get_loc("예상 도착일")
            color_hex = _eta_color(eta_colors[idx])
            styles[eta_idx] = f"color: {color_hex}; font-weight: 500"

        # 주문번호, 경로 볼드
        if "주문번호" in view.columns:
            inv_idx = view.columns.get_loc("주문번호")
            styles[inv_idx] = "font-weight: 600"

        if "경로" in view.columns:
            route_idx = view.columns.get_loc("경로")
            styles[route_idx] = "font-weight: 600"

        return styles

    styled = (
        view.style.apply(apply_styles, axis=1)
        .set_properties(
            **{
                "padding": "10px 14px",
                "font-size": "13.5px",
                "line-height": "1.3",
                "text-align": "left",
            }
        )
        .set_table_styles(
            [
                {
                    "selector": "thead th",
                    "props": [
                        ("text-align", "left"),
                        ("font-weight", "600"),
                        ("color", "#374151"),
                        ("padding", "10px 14px"),
                    ],
                }
            ]
        )
    )

    # ========================================
    # 6단계: Streamlit 렌더링
    # ========================================
    st.write(styled.to_html(escape=False, index=False), unsafe_allow_html=True)

    # 캡션
    st.caption("※ 예상 도착일 —🟢 곧 도착 | 🔴 지연 | 🟠 미확인")
    st.caption("※ 예상 입고일은 해당 경로 출발→입고 평균 리드타임 기준입니다.")
