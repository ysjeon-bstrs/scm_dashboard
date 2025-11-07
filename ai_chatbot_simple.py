"""
AI 챗봇 단순 버전: 벡터 검색 없이 직접 데이터 전달
- 복잡도 제거: Chroma, 임베딩, 세션 관리 없음
- 간단한 접근: 필터링된 데이터를 텍스트로 변환 → Gemini에 전달
- 10분 구현 목표
"""

import streamlit as st
import pandas as pd
import google.generativeai as genai


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


def ask_ai(question: str, data_context: str) -> str:
    """
    Gemini에게 질문하기

    Args:
        question: 사용자 질문
        data_context: 데이터 컨텍스트

    Returns:
        AI 답변
    """
    try:
        from datetime import datetime
        today = datetime.now().strftime('%Y-%m-%d')

        genai.configure(api_key=st.secrets["gemini"]["api_key"])
        # Gemini 2.0 모델 사용 (최신 버전)
        model = genai.GenerativeModel("gemini-2.0-flash-exp")

        prompt = f"""당신은 SCM 재고 관리 전문가입니다.

**현재 날짜: {today}**

아래 재고 데이터를 참고해서 사용자 질문에 답변하세요.

[재고 데이터]
{data_context}

[사용자 질문]
{question}

**답변 규칙:**
1. 위 데이터에 있는 정보만 사용하세요
2. 숫자는 정확하게 인용하세요
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

    # 질문 입력
    question = st.text_input(
        "질문을 입력하세요",
        placeholder="예: 총 재고는? / BA00021은 어느 센터에? / 재고가 가장 많은 센터는?",
        key="simple_q"
    )

    if st.button("💬 질문하기", type="primary", key="simple_ask") and question:
        with st.spinner("🤔 생각 중..."):
            # 데이터 컨텍스트 준비 (판매/입고 + 시계열 예측 데이터 포함!)
            context = prepare_data_context(snap, moves_df, timeline_df, max_rows=50)

            # AI에게 질문
            answer = ask_ai(question, context)

            # 답변 표시
            st.markdown("### 📊 답변")
            st.markdown(answer)

            # 컨텍스트 확인 (디버깅용)
            with st.expander("🔍 AI가 본 데이터"):
                st.text(context)

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
