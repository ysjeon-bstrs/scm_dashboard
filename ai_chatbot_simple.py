"""
AI 챗봇 단순 버전: 벡터 검색 없이 직접 데이터 전달
- 복잡도 제거: Chroma, 임베딩, 세션 관리 없음
- 간단한 접근: 필터링된 데이터를 텍스트로 변환 → Gemini에 전달
- 10분 구현 목표
"""

import streamlit as st
import pandas as pd
import google.generativeai as genai


def prepare_data_context(df: pd.DataFrame, max_rows: int = 50) -> str:
    """
    데이터프레임을 텍스트 컨텍스트로 변환

    Args:
        df: 필터링된 스냅샷 데이터
        max_rows: 최대 포함할 행 수 (토큰 제한 고려)

    Returns:
        텍스트 형태의 데이터 요약
    """
    if df.empty:
        return "데이터가 없습니다."

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

    # 샘플 데이터 (상위 N개)
    stats += f"\n📋 상세 데이터 (상위 {min(max_rows, len(df))}개):\n"
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
        genai.configure(api_key=st.secrets["gemini"]["api_key"])
        model = genai.GenerativeModel("gemini-1.5-flash")

        prompt = f"""당신은 SCM 재고 관리 전문가입니다.

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
5. 한국어로 작성하세요

답변:"""

        response = model.generate_content(prompt)
        return response.text

    except Exception as e:
        return f"⚠️ 오류 발생: {e}\n\n제공된 데이터:\n{data_context}"


def render_simple_chatbot_tab(
    snapshot_df: pd.DataFrame,
    selected_centers: list[str],
    selected_skus: list[str]
):
    """
    간단한 AI 챗봇 탭 렌더링

    Args:
        snapshot_df: 전체 스냅샷 데이터
        selected_centers: 선택된 센터
        selected_skus: 선택된 SKU
    """
    st.subheader("🤖 AI 어시스턴트 (간단 버전)")

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
            # 데이터 컨텍스트 준비
            context = prepare_data_context(snap, max_rows=50)

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
        st.caption("**분석 질문**")
        st.caption("• 재고가 가장 많은 센터는?")
        st.caption("• 태광KR에는 어떤 SKU가 있나요?")
        st.caption("• 최근 스냅샷 날짜는?")
