"""
AI 챗봇 단순 버전: 벡터 검색 없이 직접 데이터 전달
- 복잡도 제거: Chroma, 임베딩, 세션 관리 없음
- 간단한 접근: 필터링된 데이터를 텍스트로 변환 → Gemini에 전달
- 10분 구현 목표
"""

import streamlit as st
import pandas as pd
import google.generativeai as genai
import plotly.express as px
import plotly.graph_objects as go


def prepare_data_context(
    snapshot_df: pd.DataFrame,
    moves_df: pd.DataFrame = None,
    timeline_df: pd.DataFrame = None,
    max_rows: int = 50
) -> str:
    """
    데이터프레임을 텍스트 컨텍스트로 변환

    Args:
        snapshot_df: 필터링된 스냅샷 데이터
        moves_df: 판매/입고 이동 데이터 (옵션)
        timeline_df: 30일 시계열 + 예측 데이터 (옵션)
        max_rows: 최대 포함할 행 수 (토큰 제한 고려)

    Returns:
        텍스트 형태의 데이터 요약
    """
    if snapshot_df.empty:
        return "데이터가 없습니다."

    df = snapshot_df

    # 최신 날짜만 유지
    if "date" in df.columns:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date").groupby(["center", "resource_code"], as_index=False).last()

    # 상위 N개 행만 사용 (토큰 제한)
    sample = df.head(max_rows)

    # 요약 통계
    stats = f"""
📊 데이터 요약:
- 총 재고: {df['stock_qty'].sum():,.0f}개
- 센터 수: {df['center'].nunique()}곳
- SKU 수: {df['resource_code'].nunique()}개
- 최신 날짜: {df['date'].max().strftime('%Y-%m-%d') if 'date' in df.columns else 'N/A'}

센터별 재고:
"""
    for center, group in df.groupby("center")["stock_qty"].sum().items():
        stats += f"- {center}: {group:,.0f}개\n"

    # 상위 SKU
    stats += f"\n상위 SKU (재고량):\n"
    for sku, qty in df.groupby("resource_code")["stock_qty"].sum().nlargest(10).items():
        stats += f"- {sku}: {qty:,.0f}개\n"

    # 판매/입고 데이터 추가!
    if moves_df is not None and not moves_df.empty:
        stats += f"\n📦 판매/입고 데이터 (최근 30일):\n"

        # 최근 30일 필터
        if "date" in moves_df.columns:
            moves_recent = moves_df.copy()
            moves_recent["date"] = pd.to_datetime(moves_recent["date"], errors="coerce")
            cutoff_date = moves_recent["date"].max() - pd.Timedelta(days=30)
            moves_recent = moves_recent[moves_recent["date"] >= cutoff_date]

            # 센터/SKU 필터 (선택된 것만)
            if "center" in moves_recent.columns and "center" in df.columns:
                centers_in_snapshot = df["center"].unique()
                moves_recent = moves_recent[moves_recent["center"].isin(centers_in_snapshot)]
            if "resource_code" in moves_recent.columns:
                skus_in_snapshot = df["resource_code"].unique()
                moves_recent = moves_recent[moves_recent["resource_code"].isin(skus_in_snapshot)]

            # 판매/입고 집계 (30일 전체)
            if "quantity" in moves_recent.columns:
                stats += "전체 집계 (30일):\n"
                # move_type별 집계
                if "move_type" in moves_recent.columns:
                    for move_type, group in moves_recent.groupby("move_type")["quantity"].sum().items():
                        stats += f"- {move_type}: {group:,.0f}개\n"

                # SKU별 판매량
                stats += f"\nSKU별 이동량 (상위 5개):\n"
                sku_moves = moves_recent.groupby("resource_code")["quantity"].sum().nlargest(5)
                for sku, qty in sku_moves.items():
                    stats += f"- {sku}: {qty:,.0f}개\n"

                # 최근 7일 일별 상세 데이터 추가!
                latest_date = moves_recent["date"].max()
                moves_last_7days = moves_recent[moves_recent["date"] >= latest_date - pd.Timedelta(days=7)]

                if not moves_last_7days.empty:
                    stats += f"\n📅 최근 7일 일별 상세 (상위 3개 SKU):\n"

                    # 상위 3개 SKU만
                    top_skus = moves_recent.groupby("resource_code")["quantity"].sum().nlargest(3).index

                    for sku in top_skus:
                        sku_data = moves_last_7days[moves_last_7days["resource_code"] == sku]
                        if not sku_data.empty:
                            stats += f"\n{sku}:\n"

                            # 날짜별로 정렬
                            sku_data_sorted = sku_data.sort_values("date", ascending=False)

                            # 날짜별 + move_type별로 그룹화
                            for date, date_group in sku_data_sorted.groupby("date"):
                                # NaT 체크
                                if pd.isna(date):
                                    continue

                                date_str = date.strftime('%Y-%m-%d')

                                # 센터별/타입별 세분화
                                for idx, row in date_group.iterrows():
                                    center = row.get("center", "N/A")
                                    move_type = row.get("move_type", "N/A")
                                    qty = row.get("quantity", 0)
                                    stats += f"  · {date_str} | {center} | {move_type}: {qty:,.0f}개\n"

    # 30일 시계열 + 예측 데이터 추가!
    if timeline_df is not None and not timeline_df.empty:
        stats += f"\n📈 재고 추세 및 예측 데이터:\n"

        timeline = timeline_df.copy()
        if "date" in timeline.columns:
            timeline["date"] = pd.to_datetime(timeline["date"], errors="coerce")
            timeline = timeline.sort_values("date")

            # 실제 데이터와 예측 데이터 구분
            if "is_forecast" in timeline.columns:
                actual_data = timeline[timeline["is_forecast"] == False]
                forecast_data = timeline[timeline["is_forecast"] == True]

                if not actual_data.empty:
                    actual_min = actual_data["date"].min()
                    actual_max = actual_data["date"].max()
                    if pd.notna(actual_min) and pd.notna(actual_max):
                        actual_min_str = actual_min.strftime('%Y-%m-%d')
                        actual_max_str = actual_max.strftime('%Y-%m-%d')
                        stats += f"- 📊 실제 데이터 기간: {actual_min_str} ~ {actual_max_str}\n"

                if not forecast_data.empty:
                    forecast_min = forecast_data["date"].min()
                    forecast_max = forecast_data["date"].max()
                    if pd.notna(forecast_min) and pd.notna(forecast_max):
                        forecast_min_str = forecast_min.strftime('%Y-%m-%d')
                        forecast_max_str = forecast_max.strftime('%Y-%m-%d')
                        stats += f"- 🔮 예측 데이터 기간: {forecast_min_str} ~ {forecast_max_str}\n"
            else:
                # is_forecast 컬럼이 없으면 전체 범위만 표시
                date_min = timeline["date"].min().strftime('%Y-%m-%d') if pd.notna(timeline["date"].min()) else 'N/A'
                date_max = timeline["date"].max().strftime('%Y-%m-%d') if pd.notna(timeline["date"].max()) else 'N/A'
                stats += f"- 전체 기간: {date_min} ~ {date_max}\n"

            # 센터/SKU 필터 (선택된 것만)
            if "center" in timeline.columns and "center" in df.columns:
                centers_in_snapshot = df["center"].unique()
                timeline = timeline[timeline["center"].isin(centers_in_snapshot)]
            if "resource_code" in timeline.columns:
                skus_in_snapshot = df["resource_code"].unique()
                timeline = timeline[timeline["resource_code"].isin(skus_in_snapshot)]

            # SKU별 실제 추세 분석 (상위 5개)
            if "resource_code" in timeline.columns and "stock_qty" in timeline.columns:
                # 실제 데이터만 사용해서 추세 계산
                if "is_forecast" in timeline.columns:
                    actual_timeline = timeline[timeline["is_forecast"] == False]
                else:
                    actual_timeline = timeline

                if not actual_timeline.empty:
                    stats += f"\n📊 실제 재고 추세 (상위 5개 SKU):\n"

                    # 각 SKU의 실제 추세 계산
                    for sku in skus_in_snapshot[:5]:  # 상위 5개만
                        sku_timeline = actual_timeline[actual_timeline["resource_code"] == sku].sort_values("date")
                        if len(sku_timeline) >= 2:
                            # 최신 vs 최초
                            first_qty = sku_timeline.iloc[0]["stock_qty"]
                            last_qty = sku_timeline.iloc[-1]["stock_qty"]
                            change = last_qty - first_qty
                            trend = "↗️ 증가" if change > 0 else "↘️ 감소" if change < 0 else "→ 유지"

                            # 평균 재고
                            avg_qty = sku_timeline["stock_qty"].mean()

                            stats += f"- {sku}: {first_qty:,.0f}개 → {last_qty:,.0f}개 ({trend}, 평균 {avg_qty:,.0f}개)\n"

            # SKU별 예측 정보
            if "is_forecast" in timeline.columns:
                forecast_data = timeline[timeline["is_forecast"] == True]
                if not forecast_data.empty and "resource_code" in forecast_data.columns:
                    stats += f"\n🔮 SKU별 예측 재고 (상위 3개):\n"

                    # SKU별 최종 예측값
                    for sku in skus_in_snapshot[:3]:
                        sku_forecast = forecast_data[forecast_data["resource_code"] == sku]
                        if not sku_forecast.empty:
                            final_forecast = sku_forecast.iloc[-1]["stock_qty"]
                            final_date_val = sku_forecast.iloc[-1]["date"]
                            if pd.notna(final_date_val):
                                final_date = final_date_val.strftime('%Y-%m-%d')
                                stats += f"- {sku}: {final_forecast:,.0f}개 (예측일: {final_date})\n"

    # 샘플 데이터 (상위 N개)
    stats += f"\n📋 재고 상세 데이터 (상위 {min(max_rows, len(df))}개):\n"
    for idx, row in sample.iterrows():
        stats += (
            f"  · {row.get('center', 'N/A')} | "
            f"{row.get('resource_code', 'N/A')} | "
            f"재고: {row.get('stock_qty', 0):,.0f}개"
        )
        if pd.notna(row.get('resource_name')):
            stats += f" ({row.get('resource_name')})"
        stats += "\n"

    if len(df) > max_rows:
        stats += f"\n... 외 {len(df) - max_rows}개 항목\n"

    return stats


def detect_stockout_risks(
    snapshot_df: pd.DataFrame,
    moves_df: pd.DataFrame,
    timeline_df: pd.DataFrame = None,
    days_threshold: int = 7
) -> list[dict]:
    """
    품절 임박 SKU 감지

    Args:
        snapshot_df: 현재 재고 데이터
        moves_df: 판매 데이터
        timeline_df: 예측 데이터 (옵션)
        days_threshold: 품절 임박 기준 (일)

    Returns:
        품절 임박 SKU 리스트
    """
    risks = []

    if snapshot_df.empty or moves_df is None or moves_df.empty:
        return risks

    try:
        # 최근 7일 평균 판매량 계산
        moves_recent = moves_df.copy()
        if "date" in moves_recent.columns:
            moves_recent["date"] = pd.to_datetime(moves_recent["date"], errors="coerce")
            cutoff_date = moves_recent["date"].max() - pd.Timedelta(days=7)
            moves_recent = moves_recent[moves_recent["date"] >= cutoff_date]

            # 판매만 필터 (CustomerShipment 등)
            if "move_type" in moves_recent.columns:
                sales_types = ["CustomerShipment", "출고", "판매"]
                moves_recent = moves_recent[moves_recent["move_type"].isin(sales_types)]

            # SKU별 일평균 판매량
            if "resource_code" in moves_recent.columns and "quantity" in moves_recent.columns:
                daily_sales = moves_recent.groupby("resource_code")["quantity"].sum() / 7

                # 현재 재고와 비교
                for sku in daily_sales.index:
                    if daily_sales[sku] <= 0:
                        continue

                    current_stock = snapshot_df[snapshot_df["resource_code"] == sku]["stock_qty"].sum()
                    days_left = current_stock / daily_sales[sku]

                    if 0 < days_left <= days_threshold:
                        risks.append({
                            "sku": sku,
                            "current_stock": current_stock,
                            "daily_sales": daily_sales[sku],
                            "days_left": days_left,
                            "severity": "high" if days_left <= 3 else "medium"
                        })

        # 심각도 순으로 정렬
        risks.sort(key=lambda x: x["days_left"])

    except Exception as e:
        st.warning(f"품절 위험 감지 오류: {e}")

    return risks[:5]  # 상위 5개만


def detect_anomalies(
    snapshot_df: pd.DataFrame,
    timeline_df: pd.DataFrame = None,
    threshold: float = 0.5
) -> list[dict]:
    """
    재고 이상치 감지 (급증/급감)

    Args:
        snapshot_df: 현재 재고
        timeline_df: 시계열 데이터
        threshold: 변화율 임계값 (50% = 0.5)

    Returns:
        이상치 SKU 리스트
    """
    anomalies = []

    if timeline_df is None or timeline_df.empty:
        return anomalies

    try:
        timeline = timeline_df.copy()
        if "date" in timeline.columns and "resource_code" in timeline.columns:
            timeline["date"] = pd.to_datetime(timeline["date"], errors="coerce")

            # 실제 데이터만
            if "is_forecast" in timeline.columns:
                timeline = timeline[timeline["is_forecast"] == False]

            # SKU별 최근 7일 vs 이전 7일 비교
            latest_date = timeline["date"].max()
            recent_7days = timeline[timeline["date"] >= latest_date - pd.Timedelta(days=7)]
            prev_7days = timeline[
                (timeline["date"] >= latest_date - pd.Timedelta(days=14)) &
                (timeline["date"] < latest_date - pd.Timedelta(days=7))
            ]

            for sku in timeline["resource_code"].unique():
                recent_avg = recent_7days[recent_7days["resource_code"] == sku]["stock_qty"].mean()
                prev_avg = prev_7days[prev_7days["resource_code"] == sku]["stock_qty"].mean()

                if pd.notna(recent_avg) and pd.notna(prev_avg) and prev_avg > 0:
                    change_rate = (recent_avg - prev_avg) / prev_avg

                    if abs(change_rate) >= threshold:
                        anomalies.append({
                            "sku": sku,
                            "recent_avg": recent_avg,
                            "prev_avg": prev_avg,
                            "change_rate": change_rate,
                            "type": "급증" if change_rate > 0 else "급감"
                        })

            # 변화율 절댓값 순으로 정렬
            anomalies.sort(key=lambda x: abs(x["change_rate"]), reverse=True)

    except Exception as e:
        st.warning(f"이상치 감지 오류: {e}")

    return anomalies[:3]  # 상위 3개만


def check_data_quality(
    snapshot_df: pd.DataFrame,
    moves_df: pd.DataFrame = None,
    timeline_df: pd.DataFrame = None
) -> list[dict]:
    """
    데이터 품질 이슈 감지

    Returns:
        품질 이슈 리스트
    """
    issues = []

    try:
        # 1. 음수 재고 체크
        if "stock_qty" in snapshot_df.columns:
            negative_stock = snapshot_df[snapshot_df["stock_qty"] < 0]
            if not negative_stock.empty:
                issues.append({
                    "type": "negative_stock",
                    "severity": "high",
                    "message": f"⚠️ 음수 재고 발견: {len(negative_stock)}개 SKU",
                    "details": negative_stock[["resource_code", "center", "stock_qty"]].head(3).to_dict("records")
                })

        # 2. 날짜 누락 체크 (moves_df)
        if moves_df is not None and not moves_df.empty and "date" in moves_df.columns:
            moves_df_copy = moves_df.copy()
            moves_df_copy["date"] = pd.to_datetime(moves_df_copy["date"], errors="coerce")
            null_dates = moves_df_copy["date"].isna().sum()
            if null_dates > 0:
                issues.append({
                    "type": "missing_dates",
                    "severity": "medium",
                    "message": f"⚠️ 판매 데이터 날짜 누락: {null_dates}건",
                    "details": None
                })

        # 3. 최신 데이터 확인
        if "date" in snapshot_df.columns:
            snapshot_df_copy = snapshot_df.copy()
            snapshot_df_copy["date"] = pd.to_datetime(snapshot_df_copy["date"], errors="coerce")
            latest_date = snapshot_df_copy["date"].max()
            if pd.notna(latest_date):
                from datetime import datetime, timedelta
                days_old = (datetime.now() - latest_date).days
                if days_old > 1:
                    issues.append({
                        "type": "stale_data",
                        "severity": "low",
                        "message": f"ℹ️ 재고 데이터가 {days_old}일 전입니다 (최신: {latest_date.strftime('%Y-%m-%d')})",
                        "details": None
                    })

    except Exception as e:
        st.warning(f"데이터 품질 체크 오류: {e}")

    return issues


def render_proactive_insights(
    snapshot_df: pd.DataFrame,
    moves_df: pd.DataFrame,
    timeline_df: pd.DataFrame
):
    """
    프로액티브 인사이트 UI 렌더링
    """
    # 인사이트 감지
    stockout_risks = detect_stockout_risks(snapshot_df, moves_df, timeline_df)
    anomalies = detect_anomalies(snapshot_df, timeline_df)
    quality_issues = check_data_quality(snapshot_df, moves_df, timeline_df)

    # 인사이트가 하나라도 있으면 표시
    if stockout_risks or anomalies or quality_issues:
        with st.expander("🔔 주목할 이슈", expanded=True):
            col1, col2, col3 = st.columns(3)

            # 품절 위험
            with col1:
                if stockout_risks:
                    st.markdown("**⚠️ 품절 임박**")
                    for risk in stockout_risks[:3]:
                        severity_icon = "🔴" if risk["severity"] == "high" else "🟡"
                        st.caption(
                            f"{severity_icon} {risk['sku']}: "
                            f"{risk['days_left']:.1f}일 남음 "
                            f"(재고 {risk['current_stock']:.0f}개)"
                        )

            # 이상치
            with col2:
                if anomalies:
                    st.markdown("**📊 급격한 변화**")
                    for anomaly in anomalies[:3]:
                        icon = "📈" if anomaly["type"] == "급증" else "📉"
                        st.caption(
                            f"{icon} {anomaly['sku']}: "
                            f"{anomaly['type']} {abs(anomaly['change_rate'])*100:.0f}% "
                            f"({anomaly['prev_avg']:.0f}→{anomaly['recent_avg']:.0f})"
                        )

            # 데이터 품질
            with col3:
                if quality_issues:
                    st.markdown("**🔍 데이터 이슈**")
                    for issue in quality_issues[:3]:
                        st.caption(issue["message"])


def calculate_kpi(
    function_name: str,
    snapshot_df: pd.DataFrame,
    moves_df: pd.DataFrame = None,
    **kwargs
) -> dict:
    """
    KPI 계산 함수 (Function calling용)

    Args:
        function_name: 호출할 함수 이름
        snapshot_df: 재고 데이터
        moves_df: 판매 데이터
        **kwargs: 추가 파라미터

    Returns:
        계산 결과 dict
    """
    try:
        if function_name == "calculate_total_stock":
            total = snapshot_df["stock_qty"].sum()
            return {"total_stock": float(total), "unit": "개"}

        elif function_name == "get_stock_by_center":
            center_stock = snapshot_df.groupby("center")["stock_qty"].sum().to_dict()
            return {"center_stock": {k: float(v) for k, v in center_stock.items()}, "unit": "개"}

        elif function_name == "get_stock_by_sku":
            sku = kwargs.get("sku")
            if sku:
                sku_data = snapshot_df[snapshot_df["resource_code"] == sku]
                if not sku_data.empty:
                    total = sku_data["stock_qty"].sum()
                    by_center = sku_data.groupby("center")["stock_qty"].sum().to_dict()
                    return {
                        "sku": sku,
                        "total_stock": float(total),
                        "by_center": {k: float(v) for k, v in by_center.items()},
                        "unit": "개"
                    }
            return {"error": "SKU not found"}

        elif function_name == "calculate_stockout_days":
            sku = kwargs.get("sku")
            if sku and moves_df is not None:
                # 최근 7일 평균 판매량
                moves_recent = moves_df.copy()
                moves_recent["date"] = pd.to_datetime(moves_recent["date"], errors="coerce")
                cutoff_date = moves_recent["date"].max() - pd.Timedelta(days=7)
                moves_recent = moves_recent[
                    (moves_recent["date"] >= cutoff_date) &
                    (moves_recent["resource_code"] == sku)
                ]

                if not moves_recent.empty:
                    daily_sales = moves_recent["quantity"].sum() / 7
                    current_stock = snapshot_df[snapshot_df["resource_code"] == sku]["stock_qty"].sum()

                    if daily_sales > 0:
                        days_left = current_stock / daily_sales
                        return {
                            "sku": sku,
                            "current_stock": float(current_stock),
                            "daily_sales_avg": float(daily_sales),
                            "days_until_stockout": float(days_left),
                            "status": "urgent" if days_left < 3 else "warning" if days_left < 7 else "ok"
                        }

            return {"error": "Cannot calculate stockout days"}

        elif function_name == "get_top_selling_skus":
            limit = kwargs.get("limit", 5)
            if moves_df is not None:
                moves_recent = moves_df.copy()
                moves_recent["date"] = pd.to_datetime(moves_recent["date"], errors="coerce")
                cutoff_date = moves_recent["date"].max() - pd.Timedelta(days=30)
                moves_recent = moves_recent[moves_recent["date"] >= cutoff_date]

                top_skus = moves_recent.groupby("resource_code")["quantity"].sum().nlargest(limit)
                return {
                    "top_skus": {k: float(v) for k, v in top_skus.items()},
                    "period": "last_30_days",
                    "unit": "개"
                }

            return {"error": "No sales data available"}

    except Exception as e:
        return {"error": str(e)}

    return {"error": "Unknown function"}


def detect_kpi_need(question: str) -> tuple[bool, str, dict]:
    """
    질문에서 KPI 계산 필요 여부 감지

    Returns:
        (need_kpi, function_name, kwargs)
    """
    question_lower = question.lower()

    # 총 재고
    if "총 재고" in question_lower or "전체 재고" in question_lower:
        return (True, "calculate_total_stock", {})

    # 센터별 재고
    if ("센터별" in question_lower or "center" in question_lower) and "재고" in question_lower:
        return (True, "get_stock_by_center", {})

    # SKU별 재고
    import re
    sku_pattern = r'\b[A-Z]{2}\d{5}\b'
    skus = re.findall(sku_pattern, question)
    if skus and "재고" in question_lower:
        return (True, "get_stock_by_sku", {"sku": skus[0]})

    # 품절 임박
    if skus and ("품절" in question_lower or "소진" in question_lower or "남은" in question_lower):
        return (True, "calculate_stockout_days", {"sku": skus[0]})

    # 상위 판매
    if "상위" in question_lower and ("판매" in question_lower or "판매량" in question_lower):
        return (True, "get_top_selling_skus", {"limit": 5})

    return (False, None, {})


def ask_ai(question: str, data_context: str, snapshot_df: pd.DataFrame = None, moves_df: pd.DataFrame = None) -> str:
    """
    Gemini에게 질문하기 (Function calling 통합)

    Args:
        question: 사용자 질문
        data_context: 데이터 컨텍스트
        snapshot_df: 재고 데이터 (KPI 계산용)
        moves_df: 판매 데이터 (KPI 계산용)

    Returns:
        AI 답변
    """
    try:
        from datetime import datetime
        today = datetime.now().strftime('%Y-%m-%d')

        # 1. KPI 계산 필요 여부 감지
        need_kpi, func_name, kwargs = detect_kpi_need(question)

        kpi_result = None
        if need_kpi and snapshot_df is not None:
            kpi_result = calculate_kpi(func_name, snapshot_df, moves_df, **kwargs)

        genai.configure(api_key=st.secrets["gemini"]["api_key"])
        # Gemini 2.0 모델 사용 (최신 버전)
        model = genai.GenerativeModel("gemini-2.0-flash-exp")

        # 2. KPI 결과가 있으면 프롬프트에 추가
        kpi_section = ""
        if kpi_result:
            import json
            kpi_section = f"""

[정확한 계산 결과]
{json.dumps(kpi_result, ensure_ascii=False, indent=2)}

**중요:** 위 계산 결과는 코드로 정확하게 계산된 값입니다. 반드시 이 숫자를 사용하세요."""

        prompt = f"""당신은 SCM 재고 관리 전문가입니다.

**현재 날짜: {today}**

아래 재고 데이터를 참고해서 사용자 질문에 답변하세요.

[재고 데이터]
{data_context}{kpi_section}

[사용자 질문]
{question}

**답변 규칙:**
1. [정확한 계산 결과]가 있으면 반드시 그 숫자를 사용하세요 (텍스트 데이터 대신)
2. 숫자는 쉼표로 포맷팅하세요 (예: 1,234개)
3. 2-4문장으로 간결하게 작성하세요
4. 데이터에 없는 내용은 "데이터에서 확인할 수 없습니다"라고 답변하세요
5. 날짜를 언급할 때는 현재 날짜({today})를 기준으로 과거/미래를 명확히 구분하세요
6. "📊 실제 데이터 기간"은 과거 실제 데이터, "🔮 예측 데이터 기간"은 미래 예측 데이터입니다
7. 한국어로 작성하세요

답변:"""

        response = model.generate_content(prompt)
        return response.text

    except Exception as e:
        return f"⚠️ 오류 발생: {e}\n\n제공된 데이터:\n{data_context}"


def suggest_followup_questions(question: str, answer: str, data_context: str) -> list[str]:
    """
    답변을 기반으로 후속 질문 제안

    Args:
        question: 원래 질문
        answer: AI 답변
        data_context: 데이터 컨텍스트 (간략 버전)

    Returns:
        후속 질문 3개
    """
    try:
        genai.configure(api_key=st.secrets["gemini"]["api_key"])
        model = genai.GenerativeModel("gemini-2.0-flash-exp")

        # 데이터 컨텍스트 요약 (토큰 절약)
        context_summary = data_context[:500] + "..." if len(data_context) > 500 else data_context

        prompt = f"""당신은 SCM 재고 관리 전문가입니다.

사용자가 다음 질문을 했고, 답변을 받았습니다:

[질문] {question}
[답변] {answer}

[이용 가능한 데이터]
{context_summary}

이제 사용자가 궁금해할 만한 **후속 질문 3개**를 제안하세요.

**규칙:**
1. 원래 질문과 관련되고 자연스럽게 이어지는 질문
2. 제공된 데이터로 답변 가능한 질문만 제안
3. 각 질문은 15자 이내로 간결하게
4. 구체적인 SKU/센터/날짜가 있으면 포함
5. 한 줄에 하나씩, 번호 없이 작성

예시:
BA00021의 판매 추세는?
다음주 예상 재고는?
어느 센터가 재고가 부족한가요?

후속 질문:"""

        response = model.generate_content(prompt)
        questions = [q.strip() for q in response.text.strip().split('\n') if q.strip()]
        return questions[:3]  # 상위 3개만

    except Exception as e:
        # 실패 시 기본 질문 반환
        return [
            "센터별 재고 분포는?",
            "재고가 부족한 SKU는?",
            "최근 판매 추세는?"
        ]


def extract_entities_from_question(question: str, snapshot_df: pd.DataFrame, moves_df: pd.DataFrame = None) -> dict:
    """
    질문에서 엔티티 추출 (SKU, 센터, 날짜 등)

    Returns:
        {"skus": [list], "centers": [list], "date_range": tuple or None}
    """
    import re
    from datetime import datetime, timedelta

    entities = {
        "skus": [],
        "centers": [],
        "date_range": None
    }

    # 1. SKU 추출 (BA00021 형식)
    sku_pattern = r'\b[A-Z]{2}\d{5}\b'
    found_skus = re.findall(sku_pattern, question)
    if found_skus and "resource_code" in snapshot_df.columns:
        # 실제 존재하는 SKU만
        valid_skus = snapshot_df["resource_code"].unique()
        entities["skus"] = [sku for sku in found_skus if sku in valid_skus]

    # 2. 센터 추출
    question_upper = question.upper()
    if "center" in snapshot_df.columns:
        all_centers = snapshot_df["center"].unique()
        for center in all_centers:
            if center in question_upper or center.lower() in question.lower():
                entities["centers"].append(center)

    # AMZUS, KR01 등 흔한 패턴
    center_patterns = [r'\bAMZUS\b', r'\bAMZKR\b', r'\bKR0[1-9]\b']
    for pattern in center_patterns:
        matches = re.findall(pattern, question_upper)
        entities["centers"].extend(matches)

    entities["centers"] = list(set(entities["centers"]))  # 중복 제거

    # 3. 날짜 추출 (상대적 표현)
    today = datetime.now()
    question_lower = question.lower()

    if "오늘" in question_lower:
        entities["date_range"] = (today, today)
    elif "어제" in question_lower:
        yesterday = today - timedelta(days=1)
        entities["date_range"] = (yesterday, yesterday)
    elif "최근 7일" in question_lower or "지난 일주일" in question_lower:
        entities["date_range"] = (today - timedelta(days=7), today)
    elif "최근 30일" in question_lower or "지난 한달" in question_lower:
        entities["date_range"] = (today - timedelta(days=30), today)
    elif "이번주" in question_lower:
        # 이번 주 월요일부터
        weekday = today.weekday()
        monday = today - timedelta(days=weekday)
        entities["date_range"] = (monday, today)

    # 절대 날짜 패턴 (YYYY-MM-DD)
    date_pattern = r'\d{4}-\d{2}-\d{2}'
    date_matches = re.findall(date_pattern, question)
    if date_matches:
        try:
            date_obj = datetime.strptime(date_matches[0], '%Y-%m-%d')
            entities["date_range"] = (date_obj, date_obj)
        except:
            pass

    return entities


def analyze_question_for_chart(question: str) -> dict:
    """
    질문을 분석해서 차트 필요 여부 및 타입 판단

    Returns:
        {"need_chart": bool, "chart_type": str, "entities": dict}
    """
    question_lower = question.lower()

    # 차트가 필요한 키워드
    chart_keywords = ["추세", "변화", "비교", "분포", "그래프", "차트", "시각화", "트렌드"]
    need_chart = any(kw in question_lower for kw in chart_keywords)

    # 차트 타입 판단
    chart_type = None
    if "추세" in question_lower or "변화" in question_lower or "트렌드" in question_lower:
        chart_type = "line"  # 시계열
    elif "비교" in question_lower or "분포" in question_lower or "센터별" in question_lower or "sku별" in question_lower:
        chart_type = "bar"  # 바 차트
    elif "비율" in question_lower or "점유" in question_lower:
        chart_type = "pie"  # 파이 차트

    # 엔티티 추출 (간단 버전)
    entities = {
        "has_sku": bool([s for s in question if s.isupper() and len(s) >= 6]),  # BA00021 같은 패턴
        "has_center": any(c in question_lower for c in ["amz", "kr0", "센터"]),
        "time_related": any(t in question_lower for t in ["일", "주", "월", "날짜", "기간", "어제", "오늘"])
    }

    return {
        "need_chart": need_chart or chart_type is not None,
        "chart_type": chart_type,
        "entities": entities
    }


def generate_chart(
    question: str,
    snapshot_df: pd.DataFrame,
    moves_df: pd.DataFrame = None,
    timeline_df: pd.DataFrame = None
):
    """
    질문에 맞는 차트 자동 생성

    Returns:
        plotly figure 또는 None
    """
    try:
        analysis = analyze_question_for_chart(question)

        if not analysis["need_chart"]:
            return None

        chart_type = analysis["chart_type"]
        entities = analysis["entities"]

        # 1. 시계열 차트 (추세, 변화)
        if chart_type == "line" and timeline_df is not None and not timeline_df.empty:
            timeline = timeline_df.copy()
            if "date" in timeline.columns and "stock_qty" in timeline.columns:
                timeline["date"] = pd.to_datetime(timeline["date"], errors="coerce")
                timeline = timeline.sort_values("date")

                # 특정 SKU가 언급되었으면 그것만
                if entities["has_sku"] and "resource_code" in timeline.columns:
                    # 질문에서 SKU 추출 (간단 버전)
                    import re
                    sku_pattern = r'\b[A-Z]{2}\d{5}\b'
                    skus = re.findall(sku_pattern, question)
                    if skus:
                        timeline = timeline[timeline["resource_code"].isin(skus)]

                # 실제 vs 예측 구분
                if "is_forecast" in timeline.columns:
                    fig = go.Figure()

                    actual = timeline[timeline["is_forecast"] == False]
                    forecast = timeline[timeline["is_forecast"] == True]

                    if "resource_code" in timeline.columns:
                        for sku in timeline["resource_code"].unique()[:3]:  # 최대 3개
                            sku_actual = actual[actual["resource_code"] == sku]
                            sku_forecast = forecast[forecast["resource_code"] == sku]

                            if not sku_actual.empty:
                                fig.add_trace(go.Scatter(
                                    x=sku_actual["date"],
                                    y=sku_actual["stock_qty"],
                                    name=f"{sku} (실제)",
                                    mode="lines+markers"
                                ))

                            if not sku_forecast.empty:
                                fig.add_trace(go.Scatter(
                                    x=sku_forecast["date"],
                                    y=sku_forecast["stock_qty"],
                                    name=f"{sku} (예측)",
                                    mode="lines",
                                    line=dict(dash="dash")
                                ))
                    else:
                        fig.add_trace(go.Scatter(x=actual["date"], y=actual["stock_qty"], name="실제"))
                        if not forecast.empty:
                            fig.add_trace(go.Scatter(
                                x=forecast["date"],
                                y=forecast["stock_qty"],
                                name="예측",
                                line=dict(dash="dash")
                            ))

                    fig.update_layout(
                        title="재고 추세",
                        xaxis_title="날짜",
                        yaxis_title="재고량",
                        height=400
                    )
                    return fig

                else:
                    fig = px.line(
                        timeline,
                        x="date",
                        y="stock_qty",
                        color="resource_code" if "resource_code" in timeline.columns else None,
                        title="재고 추세"
                    )
                    fig.update_layout(height=400)
                    return fig

        # 2. 바 차트 (센터별, SKU별 비교)
        elif chart_type == "bar":
            if "센터" in question or "center" in question.lower():
                # 센터별 재고
                center_stock = snapshot_df.groupby("center")["stock_qty"].sum().reset_index()
                center_stock = center_stock.sort_values("stock_qty", ascending=False)

                fig = px.bar(
                    center_stock,
                    x="center",
                    y="stock_qty",
                    title="센터별 재고",
                    labels={"center": "센터", "stock_qty": "재고량"}
                )
                fig.update_layout(height=400)
                return fig

            elif "sku" in question.lower() or entities["has_sku"]:
                # SKU별 재고 (상위 10개)
                sku_stock = snapshot_df.groupby("resource_code")["stock_qty"].sum().reset_index()
                sku_stock = sku_stock.sort_values("stock_qty", ascending=False).head(10)

                fig = px.bar(
                    sku_stock,
                    x="resource_code",
                    y="stock_qty",
                    title="SKU별 재고 (상위 10개)",
                    labels={"resource_code": "SKU", "stock_qty": "재고량"}
                )
                fig.update_layout(height=400)
                return fig

        # 3. 파이 차트 (비율, 점유율)
        elif chart_type == "pie":
            center_stock = snapshot_df.groupby("center")["stock_qty"].sum().reset_index()

            fig = px.pie(
                center_stock,
                names="center",
                values="stock_qty",
                title="센터별 재고 비율"
            )
            fig.update_layout(height=400)
            return fig

    except Exception as e:
        st.warning(f"차트 생성 오류: {e}")

    return None


def render_simple_chatbot_tab(
    snapshot_df: pd.DataFrame,
    moves_df: pd.DataFrame,
    timeline_df: pd.DataFrame,
    selected_centers: list[str],
    selected_skus: list[str]
):
    """
    간단한 AI 챗봇 탭 렌더링

    Args:
        snapshot_df: 전체 스냅샷 데이터
        moves_df: 판매/입고 이동 데이터
        timeline_df: 30일 시계열 + 예측 데이터
        selected_centers: 선택된 센터
        selected_skus: 선택된 SKU
    """
    st.subheader("🤖 AI 어시스턴트 (30일 추세 + 예측 포함)")

    # 필터링
    snap = snapshot_df.copy()
    if "center" in snap.columns:
        snap = snap[snap["center"].astype(str).isin(selected_centers)]
    if "resource_code" in snap.columns:
        snap = snap[snap["resource_code"].astype(str).isin(selected_skus)]

    if snap.empty:
        st.warning("선택된 필터에 데이터가 없습니다")
        return

    st.caption(f"📊 필터링된 데이터: {len(snap):,}행 (센터 {snap['center'].nunique()}곳, SKU {snap['resource_code'].nunique()}개)")

    # 프로액티브 인사이트 표시
    render_proactive_insights(snap, moves_df, timeline_df)

    st.divider()

    # 세션 상태 초기화
    if "last_question" not in st.session_state:
        st.session_state.last_question = ""
    if "last_answer" not in st.session_state:
        st.session_state.last_answer = ""
    if "last_context" not in st.session_state:
        st.session_state.last_context = ""

    # 질문 입력
    question = st.text_input(
        "질문을 입력하세요",
        placeholder="예: 총 재고는? / BA00021은 어느 센터에? / 재고가 가장 많은 센터는?",
        key="simple_q",
        value=st.session_state.get("pending_question", "")
    )

    # pending_question이 있으면 자동 실행 후 클리어
    if "pending_question" in st.session_state and st.session_state.pending_question:
        st.session_state.pop("pending_question")
        st.rerun()

    if st.button("💬 질문하기", type="primary", key="simple_ask") and question:
        with st.spinner("🤔 생각 중..."):
            # 질문에서 엔티티 추출 (SKU, 센터, 날짜)
            entities = extract_entities_from_question(question, snap, moves_df)

            # 자동 필터링
            filtered_snap = snap.copy()
            filtered_moves = moves_df.copy() if moves_df is not None else None
            filtered_timeline = timeline_df.copy() if timeline_df is not None else None

            filter_applied = False
            filter_msg = ""

            if entities["skus"]:
                filtered_snap = filtered_snap[filtered_snap["resource_code"].isin(entities["skus"])]
                if filtered_timeline is not None and "resource_code" in filtered_timeline.columns:
                    filtered_timeline = filtered_timeline[filtered_timeline["resource_code"].isin(entities["skus"])]
                filter_msg += f"SKU: {', '.join(entities['skus'])} "
                filter_applied = True

            if entities["centers"]:
                filtered_snap = filtered_snap[filtered_snap["center"].isin(entities["centers"])]
                if filtered_moves is not None and "center" in filtered_moves.columns:
                    filtered_moves = filtered_moves[filtered_moves["center"].isin(entities["centers"])]
                if filtered_timeline is not None and "center" in filtered_timeline.columns:
                    filtered_timeline = filtered_timeline[filtered_timeline["center"].isin(entities["centers"])]
                filter_msg += f"센터: {', '.join(entities['centers'])} "
                filter_applied = True

            if entities["date_range"] and filtered_moves is not None:
                start_date, end_date = entities["date_range"]
                if "date" in filtered_moves.columns:
                    filtered_moves["date"] = pd.to_datetime(filtered_moves["date"], errors="coerce")
                    filtered_moves = filtered_moves[
                        (filtered_moves["date"] >= start_date) &
                        (filtered_moves["date"] <= end_date)
                    ]
                filter_msg += f"기간: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}"
                filter_applied = True

            if filter_applied:
                st.info(f"🎯 자동 필터 적용: {filter_msg}")

            # 데이터 컨텍스트 준비 (필터링된 데이터 사용!)
            context = prepare_data_context(filtered_snap, filtered_moves, filtered_timeline, max_rows=50)

            # AI에게 질문 (KPI 계산 지원)
            answer = ask_ai(question, context, filtered_snap, filtered_moves)

            # 세션에 저장 (필터링된 데이터도 함께)
            st.session_state.last_question = question
            st.session_state.last_answer = answer
            st.session_state.last_context = context
            st.session_state.last_filtered_snap = filtered_snap
            st.session_state.last_filtered_timeline = filtered_timeline

    # 답변 표시 (세션에서 로드)
    if st.session_state.last_answer:
        st.markdown("### 📊 답변")
        st.markdown(st.session_state.last_answer)

        # 차트 자동 생성 (필터링된 데이터 사용)
        chart_snap = st.session_state.get("last_filtered_snap", snap)
        chart_timeline = st.session_state.get("last_filtered_timeline", timeline_df)

        chart_fig = generate_chart(
            st.session_state.last_question,
            chart_snap,
            moves_df,
            chart_timeline
        )
        if chart_fig:
            st.plotly_chart(chart_fig, use_container_width=True)

        # 후속 질문 제안
        with st.spinner("💡 후속 질문 제안 중..."):
            followup_questions = suggest_followup_questions(
                st.session_state.last_question,
                st.session_state.last_answer,
                st.session_state.last_context
            )

        if followup_questions:
            st.caption("**💬 이런 것도 궁금하신가요?**")
            cols = st.columns(3)
            for i, fq in enumerate(followup_questions):
                with cols[i]:
                    if st.button(fq, key=f"followup_{i}"):
                        st.session_state.pending_question = fq
                        st.rerun()

        # 컨텍스트 확인 (디버깅용)
        with st.expander("🔍 AI가 본 데이터"):
            st.text(st.session_state.last_context)

    # 예시 질문
    st.divider()
    st.caption("**💡 예시 질문:**")

    col1, col2 = st.columns(2)
    with col1:
        st.caption("**재고 조회**")
        st.caption("• 총 재고는?")
        st.caption("• 센터별 재고는?")
        st.caption("• BA00021은 어느 센터에 있나요?")

    with col2:
        st.caption("**추세/예측 분석 🆕**")
        st.caption("• BA00021의 재고 추세는?")
        st.caption("• 다음주 예상 재고는?")
        st.caption("• 어느 SKU가 재고가 증가하고 있나요?")
