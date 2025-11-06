"""
AI 챗봇 1.5단계: 하이브리드 시스템 (정량 계산 + 벡터 검색)

핵심 개선:
- 질문 분류: 정량형 vs 탐색형 자동 구분
- 정량 계산: 판다스로 정확한 숫자 계산 (합계, 평균, 최대/최소 등)
- 탐색 검색: 벡터 검색으로 관련 문서 찾기
- AI 역할: 계산 결과 설명 + 근거 요약 (환각 방지)
"""

import streamlit as st
import pandas as pd
import chromadb
import google.generativeai as genai
import uuid
import hashlib
from typing import Tuple, List, Dict, Optional, Literal
import re


# ============================================================================
# 1. 질문 분류 엔진
# ============================================================================

def classify_question(question: str) -> Literal["quantitative", "exploratory", "business"]:
    """
    질문을 3가지 유형으로 분류

    Args:
        question: 사용자 질문

    Returns:
        - "quantitative": 정량 계산 필요 (합계, 평균, 개수 등)
        - "exploratory": 탐색 검색 필요 (어디에, 무엇이, 언제 등)
        - "business": 비즈니스 판단 필요 (부족, 충분, 권고 등) - 향후 2단계
    """
    q = question.lower()

    # 정량형 키워드 (판다스 계산 필요)
    quantitative_keywords = [
        # 집계 함수
        "총", "전체", "합계", "총합", "모두", "all", "total", "sum",
        "평균", "average", "avg", "mean",
        "최대", "최소", "가장 많", "가장 적", "max", "min", "maximum", "minimum",
        "개수", "몇 개", "몇개", "count", "number of",

        # 센터별/SKU별 집계
        "센터별", "SKU별", "제품별", "창고별",
        "각 센터", "각 SKU", "각각",

        # 비교
        "많은", "적은", "높은", "낮은",
    ]

    # 탐색형 키워드 (벡터 검색 필요)
    exploratory_keywords = [
        "어디", "어느", "무엇", "언제", "어떤",
        "where", "what", "when", "which",
        "목록", "리스트", "list",
        "보유", "있는", "존재",
    ]

    # 비즈니스형 키워드 (향후 2단계)
    business_keywords = [
        "부족", "충분", "권고", "권장", "위험", "경고",
        "안전재고", "리드타임", "입고 필요", "발주",
        "insufficient", "enough", "recommend", "risk",
    ]

    # 키워드 매칭
    if any(kw in q for kw in quantitative_keywords):
        return "quantitative"
    elif any(kw in q for kw in business_keywords):
        return "business"
    elif any(kw in q for kw in exploratory_keywords):
        return "exploratory"
    else:
        # 기본값: 탐색형 (벡터 검색이 더 안전)
        return "exploratory"


def extract_entities(question: str, available_centers: List[str], available_skus: List[str]) -> Dict[str, List[str]]:
    """
    질문에서 센터/SKU 개체명 추출

    Args:
        question: 사용자 질문
        available_centers: 현재 필터의 센터 목록
        available_skus: 현재 필터의 SKU 목록

    Returns:
        {"centers": [...], "skus": [...]}
    """
    q = question.upper()

    # 센터 추출
    centers = [c for c in available_centers if c.upper() in q]

    # SKU 추출
    skus = [s for s in available_skus if s.upper() in q]

    return {
        "centers": centers if centers else available_centers,  # 없으면 전체
        "skus": skus if skus else available_skus,
    }


# ============================================================================
# 2. 정량 계산 엔진 (판다스)
# ============================================================================

def calculate_quantitative(
    question: str,
    snapshot_df: pd.DataFrame,
    entities: Dict[str, List[str]]
) -> Dict[str, any]:
    """
    판다스로 정확한 정량 계산 수행

    Args:
        question: 사용자 질문
        snapshot_df: 필터링된 스냅샷 데이터
        entities: 추출된 센터/SKU 정보

    Returns:
        {
            "type": "total" | "average" | "max" | "min" | "count" | "group_by",
            "result": 계산 결과 (숫자 또는 DataFrame),
            "description": 결과 설명 텍스트
        }
    """
    q = question.lower()

    # 데이터 필터링
    df = snapshot_df.copy()
    if "center" in df.columns:
        df = df[df["center"].isin(entities["centers"])]
    if "resource_code" in df.columns:
        df = df[df["resource_code"].isin(entities["skus"])]

    # 최신 스냅샷만 사용 (센터-SKU별 최신 날짜)
    if "date" in df.columns and not df.empty:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date").groupby(["center", "resource_code"], as_index=False).last()

    if df.empty:
        return {
            "type": "empty",
            "result": None,
            "description": "해당 조건에 맞는 데이터가 없습니다."
        }

    # 재고 수량 숫자 변환
    df["stock_qty"] = pd.to_numeric(df.get("stock_qty", 0), errors="coerce").fillna(0)

    # 1. 총 재고 / 합계
    if any(kw in q for kw in ["총", "전체", "합계", "총합", "total", "sum"]):
        total = df["stock_qty"].sum()
        centers_str = ", ".join(entities["centers"][:3])
        if len(entities["centers"]) > 3:
            centers_str += f" 외 {len(entities["centers"]) - 3}곳"

        return {
            "type": "total",
            "result": total,
            "description": f"{centers_str}의 총 재고는 **{total:,.0f}개**입니다.",
            "breakdown": df.groupby("center")["stock_qty"].sum().to_dict()
        }

    # 2. 평균
    if any(kw in q for kw in ["평균", "average", "avg", "mean"]):
        avg = df["stock_qty"].mean()
        return {
            "type": "average",
            "result": avg,
            "description": f"평균 재고는 **{avg:,.0f}개**입니다.",
        }

    # 3. 최대 / 최소
    if any(kw in q for kw in ["최대", "가장 많", "max", "maximum"]):
        max_row = df.loc[df["stock_qty"].idxmax()]
        return {
            "type": "max",
            "result": max_row["stock_qty"],
            "description": f"재고가 가장 많은 것은 **{max_row['center']} {max_row['resource_code']}** ({max_row['stock_qty']:,.0f}개)입니다.",
        }

    if any(kw in q for kw in ["최소", "가장 적", "min", "minimum"]):
        min_row = df.loc[df["stock_qty"].idxmin()]
        return {
            "type": "min",
            "result": min_row["stock_qty"],
            "description": f"재고가 가장 적은 것은 **{min_row['center']} {min_row['resource_code']}** ({min_row['stock_qty']:,.0f}개)입니다.",
        }

    # 4. 센터별 / SKU별 집계
    if any(kw in q for kw in ["센터별", "각 센터", "창고별"]):
        by_center = df.groupby("center")["stock_qty"].sum().sort_values(ascending=False)
        lines = [f"- **{center}**: {qty:,.0f}개" for center, qty in by_center.items()]
        return {
            "type": "group_by_center",
            "result": by_center.to_dict(),
            "description": "센터별 재고:\n" + "\n".join(lines)
        }

    if any(kw in q for kw in ["SKU별", "제품별", "각 SKU"]):
        by_sku = df.groupby("resource_code")["stock_qty"].sum().sort_values(ascending=False)
        lines = [f"- **{sku}**: {qty:,.0f}개" for sku, qty in by_sku.head(10).items()]
        if len(by_sku) > 10:
            lines.append(f"... 외 {len(by_sku) - 10}개 SKU")
        return {
            "type": "group_by_sku",
            "result": by_sku.to_dict(),
            "description": "SKU별 재고 (상위 10개):\n" + "\n".join(lines)
        }

    # 5. 개수
    if any(kw in q for kw in ["개수", "몇 개", "몇개", "count"]):
        if "센터" in q:
            count = df["center"].nunique()
            return {
                "type": "count",
                "result": count,
                "description": f"센터 개수는 **{count}개**입니다.",
            }
        elif "SKU" in q or "제품" in q:
            count = df["resource_code"].nunique()
            return {
                "type": "count",
                "result": count,
                "description": f"SKU 개수는 **{count}개**입니다.",
            }

    # 기본: 총 재고 반환
    total = df["stock_qty"].sum()
    return {
        "type": "total",
        "result": total,
        "description": f"조회된 재고는 **{total:,.0f}개**입니다.",
    }


# ============================================================================
# 3. 벡터 검색 엔진 (기존 로직)
# ============================================================================

@st.cache_resource
def _chroma_cloud():
    """Chroma Cloud 클라이언트"""
    try:
        client = chromadb.CloudClient(
            api_key=st.secrets["chroma"]["api_key"],
            tenant=st.secrets["chroma"]["tenant"],
            database=st.secrets["chroma"]["database"],
        )
        return client
    except Exception as e:
        st.error(f"Chroma Cloud 연결 실패: {e}")
        return None


def _get_filter_hash(centers: list, skus: list) -> str:
    """필터 조합 해시"""
    key = f"{sorted(centers)}_{sorted(skus)}"
    return hashlib.md5(key.encode()).hexdigest()[:8]


def _build_session_collection_name() -> str:
    """세션 컬렉션명"""
    sid = st.session_state.get("_ai_session_id")
    if not sid:
        sid = st.session_state["_ai_session_id"] = uuid.uuid4().hex[:8]
    return f"scm_session_{sid}"


def _embed_batch(texts: list[str], batch_size: int = 100) -> list[list[float]]:
    """Gemini 임베딩 (배치 처리)"""
    if not texts:
        return []

    genai.configure(api_key=st.secrets["gemini"]["api_key"])
    model = st.secrets["gemini"].get("embedding_model", "text-embedding-004")

    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            res = genai.embed_content(
                model=model,
                content=batch,
                task_type="retrieval_document"
            )
            if hasattr(res, "embeddings"):
                all_embeddings.extend([e.values for e in res.embeddings])
            elif isinstance(res, dict) and "embeddings" in res:
                all_embeddings.extend([r["embedding"] for r in res["embeddings"]])
            else:
                all_embeddings.append(res["embedding"])
        except Exception as e:
            st.warning(f"임베딩 배치 {i//batch_size + 1} 실패: {e}")
            continue

    return all_embeddings


def _documents_from_snapshot(snap: pd.DataFrame, max_rows: int = 2000) -> Tuple[List[str], List[Dict], List[str]]:
    """스냅샷 → 문서 변환"""
    docs, metas, ids = [], [], []
    use = snap.head(max_rows)

    for i, r in use.iterrows():
        # 날짜를 안전하게 문자열로 변환
        date_val = r.get('date')
        if pd.isna(date_val):
            date_str = "N/A"
        else:
            try:
                date_str = pd.to_datetime(date_val).strftime('%Y-%m-%d')
            except:
                date_str = str(date_val)

        doc = (
            f"[SNAPSHOT] "
            f"날짜:{date_str} "
            f"센터:{r.get('center')} "
            f"SKU:{r.get('resource_code')} "
            f"재고:{r.get('stock_qty')}개"
        )
        if pd.notna(r.get('resource_name')):
            doc += f" ({r.get('resource_name')})"

        docs.append(doc)
        # "type" 키워드는 Chroma 내부 예약어이므로 "doc_type"으로 변경
        metas.append({
            "doc_type": "snapshot",
            "center": str(r.get("center", "")) if pd.notna(r.get("center")) else "",
            "sku": str(r.get("resource_code", "")) if pd.notna(r.get("resource_code")) else "",
            "date": date_str,
        })
        ids.append(f"snap-{i}")

    return docs, metas, ids


def _ensure_session_index(snap_filtered: pd.DataFrame, filter_hash: str, max_rows: int = 2000):
    """세션 인덱스 생성/재사용 (캐싱)"""
    client = _chroma_cloud()
    if client is None:
        return None, 0

    col_name = _build_session_collection_name()

    # 캐싱: 같은 필터면 재사용
    if st.session_state.get("_last_filter_hash") == filter_hash:
        try:
            col = client.get_collection(col_name)
            return col, col.count()
        except:
            pass

    # 새 인덱싱
    try:
        client.delete_collection(col_name)
    except:
        pass

    col = client.create_collection(col_name)

    docs, metas, ids = _documents_from_snapshot(snap_filtered, max_rows)
    if not docs:
        return col, 0

    embs = _embed_batch(docs, batch_size=100)
    if len(embs) != len(docs):
        docs = docs[:len(embs)]
        metas = metas[:len(embs)]
        ids = ids[:len(embs)]

    if embs:
        col.add(ids=ids, documents=docs, metadatas=metas, embeddings=embs)

    st.session_state["_last_filter_hash"] = filter_hash
    return col, len(docs)


def search_documents(col: chromadb.Collection, question: str, k: int = 5) -> List[str]:
    """벡터 검색"""
    try:
        res = col.query(query_texts=[question], n_results=k)
        return res.get("documents", [[]])[0]
    except Exception as e:
        st.error(f"검색 실패: {e}")
        return []


# ============================================================================
# 4. 하이브리드 답변 생성
# ============================================================================

def generate_hybrid_answer(
    question: str,
    question_type: str,
    calc_result: Optional[Dict] = None,
    search_docs: Optional[List[str]] = None,
    session_digest: str = ""
) -> str:
    """
    하이브리드 답변 생성

    Args:
        question: 사용자 질문
        question_type: "quantitative" | "exploratory"
        calc_result: 정량 계산 결과
        search_docs: 벡터 검색 문서
        session_digest: 세션 요약

    Returns:
        AI 답변 텍스트
    """
    try:
        genai.configure(api_key=st.secrets["gemini"]["api_key"])
        model_name = st.secrets["gemini"].get("generation_model", "gemini-1.5-flash")
        model = genai.GenerativeModel(model_name)

        if question_type == "quantitative":
            # 정량형: 계산 결과를 설명/해석
            prompt = f"""당신은 SCM 재고 관리 전문가입니다.

아래 계산 결과를 사용자에게 간결하게 설명하세요.

[질문]
{question}

[정확한 계산 결과]
{calc_result.get('description', '')}

[세션 컨텍스트]
{session_digest}

**답변 규칙:**
1. 계산 결과를 그대로 사용 (숫자 변경 금지)
2. 2-3문장으로 간결하게
3. 추가 인사이트나 해석 제공 가능
4. 한국어로 작성

답변:"""

        else:  # exploratory
            # 탐색형: 검색된 문서 요약
            prompt = f"""당신은 SCM 재고 관리 전문가입니다.

아래 근거 데이터만 사용해서 질문에 답하세요.

[질문]
{question}

[근거 데이터]
{chr(10).join(f"- {d}" for d in search_docs)}

[세션 컨텍스트]
{session_digest}

**답변 규칙:**
1. 근거 데이터에 있는 정보만 사용
2. 수치/센터/SKU를 명시적으로 포함
3. 2-4문장으로 간결하게
4. 추정하지 말 것
5. 한국어로 작성

답변:"""

        response = model.generate_content(prompt)
        return response.text

    except Exception as e:
        st.error(f"답변 생성 실패: {e}")
        # 폴백
        if calc_result:
            return calc_result.get('description', '계산 결과를 확인하세요.')
        else:
            return "답변 생성 중 오류가 발생했습니다.\n\n" + \
                   "\n".join(f"• {d}" for d in (search_docs or [])[:3])


# ============================================================================
# 5. 메인 UI 함수
# ============================================================================

def render_hybrid_chatbot_tab(
    snapshot_df: pd.DataFrame,
    selected_centers: List[str],
    selected_skus: List[str]
):
    """
    하이브리드 AI 챗봇 탭 렌더링

    Args:
        snapshot_df: 전체 스냅샷 데이터
        selected_centers: 선택된 센터
        selected_skus: 선택된 SKU
    """
    st.subheader("🤖 AI 어시스턴트 (1.5단계: 하이브리드)")

    # 필터링
    snap = snapshot_df.copy()
    if "center" in snap.columns:
        snap = snap[snap["center"].astype(str).isin(selected_centers)]
    if "resource_code" in snap.columns:
        snap = snap[snap["resource_code"].astype(str).isin(selected_skus)]

    if snap.empty:
        st.warning("선택된 필터에 데이터가 없습니다")
        return

    # 세션 요약
    session_digest = (
        f"총 재고={pd.to_numeric(snap.get('stock_qty'), errors='coerce').fillna(0).sum():,.0f}개 / "
        f"센터 {snap.get('center', pd.Series([], dtype=str)).nunique()}곳 / "
        f"SKU {snap.get('resource_code', pd.Series([], dtype=str)).nunique()}개 / "
        f"최신일={pd.to_datetime(snap.get('date'), errors='coerce').max():%Y-%m-%d}"
    )

    # 필터 해시
    filter_hash = _get_filter_hash(selected_centers, selected_skus)

    # 자동 인덱싱 (탐색형 질문 대비)
    if st.session_state.get("_last_filter_hash") != filter_hash:
        with st.spinner("📚 벡터 인덱스 준비 중..."):
            try:
                col, n = _ensure_session_index(snap, filter_hash, max_rows=2000)
                if col and n > 0:
                    st.success(f"✅ {n:,}건 인덱싱 완료")
            except Exception as e:
                st.warning(f"인덱싱 실패 (정량 계산은 가능): {e}")

    # 세션 정보
    with st.expander("ℹ️ 세션 정보"):
        st.caption(session_digest)
        st.caption("**시스템 모드:** 하이브리드 (정량 계산 + 벡터 검색)")

    st.divider()

    # 질문 입력
    q = st.text_input(
        "질문을 입력하세요",
        placeholder="예: REVVN의 총 재고는? / 재고가 가장 많은 센터는?",
        key="hybrid_q"
    )

    col1, col2 = st.columns([4, 1])
    with col2:
        k = st.slider("검색 결과", 1, 10, 5, key="hybrid_k")

    if st.button("💬 질문하기", type="primary") and q:
        with st.spinner("🔍 분석 중..."):
            # 1. 질문 분류
            q_type = classify_question(q)
            entities = extract_entities(q, selected_centers, selected_skus)

            st.caption(f"🔖 질문 유형: **{q_type}** | 대상: {entities['centers'][:2]} × {entities['skus'][:2]}")

            # 2. 질문 유형별 처리
            if q_type == "quantitative":
                # 정량 계산
                calc_result = calculate_quantitative(q, snap, entities)

                # AI 설명 생성
                answer = generate_hybrid_answer(
                    question=q,
                    question_type="quantitative",
                    calc_result=calc_result,
                    session_digest=session_digest
                )

                st.markdown("### 📊 답변")
                st.markdown(answer)

                # 계산 상세
                with st.expander("🔢 계산 상세"):
                    st.write(f"**계산 유형:** {calc_result['type']}")
                    st.write(f"**결과:** {calc_result['result']}")
                    if "breakdown" in calc_result:
                        st.write("**센터별 분해:**")
                        for center, qty in calc_result["breakdown"].items():
                            st.write(f"- {center}: {qty:,.0f}개")

            elif q_type == "exploratory":
                # 벡터 검색
                try:
                    client = _chroma_cloud()
                    col = client.get_collection(_build_session_collection_name())
                    docs = search_documents(col, q, k=k)

                    if not docs:
                        st.info("관련 데이터를 찾지 못했습니다. 필터를 조정해보세요.")
                    else:
                        # AI 요약 생성
                        answer = generate_hybrid_answer(
                            question=q,
                            question_type="exploratory",
                            search_docs=docs,
                            session_digest=session_digest
                        )

                        st.markdown("### 📊 답변")
                        st.markdown(answer)

                        # 근거
                        with st.expander("🔎 근거 데이터"):
                            for i, d in enumerate(docs, 1):
                                st.write(f"{i}. {d}")

                except Exception as e:
                    st.error(f"검색 실패: {e}")

            else:  # business
                st.info("🚧 비즈니스 판단 질문은 2단계에서 지원 예정입니다.")
                st.caption("현재는 정량 계산과 탐색 검색만 가능합니다.")

    # 예시 질문
    st.divider()
    st.caption("**💡 예시 질문:**")

    col1, col2 = st.columns(2)
    with col1:
        st.caption("**정량 계산형**")
        st.caption("• 총 재고는?")
        st.caption("• 센터별 재고는?")
        st.caption("• 재고가 가장 많은 센터는?")
        st.caption("• SKU 개수는?")

    with col2:
        st.caption("**탐색형**")
        st.caption("• REVVN에는 어떤 SKU가 있나요?")
        st.caption("• BA00021은 어느 센터에 있나요?")
        st.caption("• 최근 스냅샷 날짜는?")
        st.caption("• 태광KR의 재고 현황은?")
