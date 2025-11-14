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
    입고 예정 원본 데이터를 SKU별 행으로 변환합니다.

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
            - pred_inbound_date: 예상 입고일

        sku_color_map: SKU → 색상 매핑 딕셔너리 (선택사항)

    Returns:
        변환된 테이블 데이터프레임
            컬럼:
            - 주문번호: 주문번호
            - 경로: 경로 (예: "KR → US (태광KR)")
            - 제품(SKU): 제품명 + SKU (HTML, 제품명 검정, SKU 코드만 색상)
            - 수량: 수량 (숫자)
            - 운송모드: 운송모드
            - 출발일: 출발일 (YYYY-MM-DD)
            - 예상 도착일: ETA 표시 텍스트 (YYYY-MM-DD 또는 "미확인")
            - eta_color: ETA 색상 코드 (내부용, "red"/"green"/"gray"/"orange")

    Notes:
        - 각 SKU를 별도 행으로 표시 (그룹핑 없음)
        - 제품명은 검정, SKU 코드만 색상 적용 (HTML)
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

    # ========================================
    # 2단계: 날짜 및 수량 정규화
    # ========================================
    # onboard_date: 출발일
    df["onboard_date"] = pd.to_datetime(df["onboard_date"], errors="coerce")

    # pred_inbound_date: 예상 입고일 ("not_defined" → NaT)
    df["pred_inbound_date"] = df["pred_inbound_date"].replace("not_defined", pd.NaT)
    df["pred_inbound_date"] = pd.to_datetime(df["pred_inbound_date"], errors="coerce")

    # expected_inbound_date: 리드타임 기반 예상 입고일
    if "expected_inbound_date" in df.columns:
        df["expected_inbound_date"] = pd.to_datetime(
            df["expected_inbound_date"], errors="coerce"
        )
    else:
        df["expected_inbound_date"] = pd.NaT

    # qty_ea: 수량
    df["qty_ea"] = pd.to_numeric(df["qty_ea"], errors="coerce").fillna(0).astype(int)

    # sku_color_map 기본값 처리
    if sku_color_map is None:
        sku_color_map = {}

    # ========================================
    # 3단계: 각 행을 요약 테이블 행으로 변환 (groupby 제거)
    # ========================================
    rows = []
    today = pd.Timestamp.today().normalize()

    for _, row in df.iterrows():
        # 기본 정보
        inv = str(row["invoice_no"]) if pd.notna(row["invoice_no"]) else "N/A"
        sku_code = str(row["resource_code"]) if pd.notna(row["resource_code"]) else ""
        sku_name = str(row["resource_name"]) if pd.notna(row["resource_name"]) else ""
        qty = int(row["qty_ea"])
        mode = str(row["carrier_mode"]) if pd.notna(row["carrier_mode"]) else ""
        center = str(row["to_center"]) if pd.notna(row["to_center"]) else ""

        # 경로 생성
        from_country = str(row["from_country"]) if pd.notna(row["from_country"]) else ""
        to_country = str(row["to_country"]) if pd.notna(row["to_country"]) else ""
        route = (
            f"{from_country} → {to_country} ({center})"
            if center
            else f"{from_country} → {to_country}"
        )

        # 제품(SKU) HTML 생성: 제품명 검정, SKU 코드만 색상
        sku_color = sku_color_map.get(sku_code, "#b91c1c")  # 기본 빨강
        if sku_name:
            product_html = (
                f"{sku_name} (<span style='color:{sku_color}'>{sku_code}</span>)"
            )
        else:
            product_html = f"<span style='color:{sku_color}'>{sku_code}</span>"

        # 출발일
        onboard = row["onboard_date"]
        onboard_str = onboard.strftime("%Y-%m-%d") if pd.notna(onboard) else ""

        # ETA 및 색상
        eta = row["pred_inbound_date"]
        if pd.isna(eta):
            eta_text, eta_color = "미확인", "orange"
        else:
            d = (eta.date() - today.date()).days
            eta_text = eta.strftime("%Y-%m-%d")

            if d < 0:
                eta_color = "red"
            elif d <= 5:
                eta_color = "green"
            else:
                eta_color = "gray"

        # 행 추가
        rows.append(
            {
                "주문번호": inv,
                "경로": route,
                "제품(SKU)": product_html,
                "수량": qty,
                "운송모드": mode,
                "출발일": onboard_str,
                "예상 도착일": eta_text,
                "eta_color": eta_color,  # 내부용
            }
        )

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)

    # ========================================
    # 4단계: 정렬 (출발일 오름차순 - 오래된 것부터)
    # ========================================
    # 출발일을 날짜로 변환하여 정렬
    out["_onboard_sort"] = pd.to_datetime(out["출발일"], errors="coerce")
    out = out.sort_values("_onboard_sort", ascending=True, na_position="last")
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
            (제품(SKU) 컬럼에 이미 HTML 색상이 적용되어 있어야 함)
        title: 테이블 제목 (기본: "📦 입고 예정 현황")
        height: 테이블 높이 (픽셀, 기본: 520)
        sku_color_map: (사용 안 함, 하위 호환성 유지용)

    Notes:
        - SKU별 한 줄씩 표시
        - 제품명은 검정, SKU 코드만 색상 (build 단계에서 HTML 생성됨)
        - 주문번호·경로 볼드
        - ETA 색상만 상태별 변경 (빨강/초록/주황/회색)
        - 수량은 숫자만 표시 (ea 제거)
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
    # 3단계: 데이터 준비
    # ========================================
    view = df.copy()

    # 수량 포맷팅 (숫자만, ea 제거)
    view["수량"] = view["수량"].apply(lambda x: f"{x:,}")

    display_cols = [
        "주문번호",
        "경로",
        "제품(SKU)",
        "수량",
        "운송모드",
        "출발일",
        "예상 도착일",
    ]
    view = view[[col for col in display_cols if col in view.columns]]

    # 인덱스 리셋 (숫자 인덱스 제거)
    view = view.reset_index(drop=True)

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
        view.style.hide(axis="index")
        .apply(apply_styles, axis=1)
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
