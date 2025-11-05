"""
AI 챗봇 통합 코드 (개선 버전)
v9_app.py에 통합하기 위한 완전한 구현

주요 개선사항:
- Chroma Cloud API 호환성 수정
- 임베딩 배치 크기 제한 (100건)
- 필터 변경 시에만 재인덱싱 (캐싱)
- 강화된 에러 핸들링
- 자동 인덱싱 UX
- 비용/사용량 모니터링
"""

import streamlit as st
import pandas as pd
import chromadb
import google.generativeai as genai
import uuid
import hashlib
from typing import Tuple, List, Dict, Optional


# ============================================================================
# 1. 클라이언트 & 유틸리티
# ============================================================================

@st.cache_resource
def _chroma_cloud():
    """Chroma Cloud 클라이언트 (싱글톤)"""
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
    """필터 조합의 해시값 생성 (캐싱용)"""
    key = f"{sorted(centers)}_{sorted(skus)}"
    return hashlib.md5(key.encode()).hexdigest()[:8]


def _build_session_collection_name() -> str:
    """세션 단위 에페메럴 컬렉션명"""
    sid = st.session_state.get("_ai_session_id")
    if not sid:
        sid = st.session_state["_ai_session_id"] = uuid.uuid4().hex[:8]
    return f"scm_session_{sid}"


# ============================================================================
# 2. 임베딩 & 문서화
# ============================================================================

def _embed_batch(texts: list[str], batch_size: int = 100) -> list[list[float]]:
    """
    Gemini 배치 임베딩 (배치 크기 제한)

    Args:
        texts: 임베딩할 텍스트 리스트
        batch_size: 한 번에 처리할 최대 개수 (기본 100)

    Returns:
        임베딩 벡터 리스트
    """
    if not texts:
        return []

    genai.configure(api_key=st.secrets["gemini"]["api_key"])
    model = st.secrets["gemini"].get("embedding_model", "text-embedding-004")

    all_embeddings = []
    failed_batches = 0

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            res = genai.embed_content(
                model=model,
                content=batch,
                task_type="retrieval_document"
            )

            # 응답 포맷 방어적 처리
            if hasattr(res, "embeddings"):
                all_embeddings.extend([e.values for e in res.embeddings])
            elif isinstance(res, dict) and "embeddings" in res:
                all_embeddings.extend([r["embedding"] for r in res["embeddings"]])
            else:
                all_embeddings.append(res["embedding"])

        except Exception as e:
            failed_batches += 1
            st.warning(f"임베딩 배치 {i//batch_size + 1} 실패: {e}")
            # 실패한 배치는 빈 임베딩으로 (컬렉션 추가 시 스킵됨)
            continue

    # 사용량 추적
    if "ai_usage" not in st.session_state:
        st.session_state["ai_usage"] = {"embeddings": 0, "generations": 0, "total_docs": 0}
    st.session_state["ai_usage"]["embeddings"] += len(texts)
    st.session_state["ai_usage"]["total_docs"] = max(
        st.session_state["ai_usage"]["total_docs"],
        len(texts)
    )

    if failed_batches > 0:
        st.warning(f"⚠️ {failed_batches}개 배치 임베딩 실패 (계속 진행)")

    return all_embeddings


def _documents_from_snapshot(
    snap: pd.DataFrame,
    *,
    max_rows: int = 2000
) -> Tuple[List[str], List[Dict], List[str]]:
    """
    스냅샷 데이터프레임을 검색 가능한 문서로 변환

    Args:
        snap: 스냅샷 데이터프레임
        max_rows: 최대 처리 행 수 (비용 통제)

    Returns:
        (documents, metadatas, ids) 튜플
    """
    docs, metas, ids = [], [], []
    use = snap.head(max_rows)

    for i, r in use.iterrows():
        # 문서 텍스트: 간결하고 검색 가능한 형태
        doc = (
            f"[SNAPSHOT] "
            f"날짜:{r.get('date')} "
            f"센터:{r.get('center')} "
            f"SKU:{r.get('resource_code')} "
            f"재고:{r.get('stock_qty')}개"
        )

        # 품명이 있으면 추가
        if pd.notna(r.get('resource_name')):
            doc += f" ({r.get('resource_name')})"

        docs.append(doc)
        metas.append({
            "type": "snapshot",
            "center": str(r.get("center", "")),
            "sku": str(r.get("resource_code", "")),
            "date": str(r.get("date", "")),
        })
        ids.append(f"snap-{i}")

    return docs, metas, ids


# ============================================================================
# 3. 인덱싱 (캐싱 적용)
# ============================================================================

def _ensure_session_index(
    snap_filtered: pd.DataFrame,
    filter_hash: str,
    max_rows: int = 2000
) -> Tuple[Optional[chromadb.Collection], int]:
    """
    세션 컬렉션 인덱스 생성/재사용

    Args:
        snap_filtered: 필터링된 스냅샷 데이터
        filter_hash: 현재 필터 조합의 해시값
        max_rows: 최대 인덱싱 행 수

    Returns:
        (collection, indexed_count) 튜플
    """
    client = _chroma_cloud()
    if client is None:
        return None, 0

    col_name = _build_session_collection_name()

    # 캐싱: 이미 같은 필터로 인덱싱했으면 재사용
    if st.session_state.get("_last_filter_hash") == filter_hash:
        try:
            col = client.get_collection(col_name)
            count = col.count()
            if count > 0:
                return col, count
        except Exception:
            pass  # 컬렉션이 없으면 아래에서 새로 생성

    # 새 인덱싱 필요
    try:
        # 기존 컬렉션 삭제 (올바른 메서드 사용)
        try:
            client.delete_collection(col_name)
        except Exception:
            pass

        # 새 컬렉션 생성
        col = client.create_collection(col_name)

    except Exception as e:
        st.error(f"컬렉션 생성 실패: {e}")
        return None, 0

    # 문서 생성
    docs, metas, ids = _documents_from_snapshot(snap_filtered, max_rows=max_rows)
    if not docs:
        st.warning("인덱싱할 데이터가 없습니다")
        return col, 0

    # 임베딩 생성 (배치 처리)
    embs = _embed_batch(docs, batch_size=100)

    # 임베딩 실패한 것 제외하고 추가
    if len(embs) != len(docs):
        st.warning(f"일부 문서 임베딩 실패: {len(docs) - len(embs)}건 제외")
        # 성공한 것만 추가
        docs = docs[:len(embs)]
        metas = metas[:len(embs)]
        ids = ids[:len(embs)]

    if not embs:
        st.error("모든 임베딩 실패")
        return col, 0

    # Chroma에 추가
    try:
        col.add(
            ids=ids,
            documents=docs,
            metadatas=metas,
            embeddings=embs
        )
    except Exception as e:
        st.error(f"문서 추가 실패: {e}")
        return col, 0

    # 캐싱: 이번 필터 해시 저장
    st.session_state["_last_filter_hash"] = filter_hash

    return col, len(docs)


# ============================================================================
# 4. 검색 & 답변 생성
# ============================================================================

def _query_session(col: chromadb.Collection, question: str, k: int = 5) -> List[str]:
    """
    세션 컬렉션에서 관련 문서 검색

    Args:
        col: Chroma 컬렉션
        question: 질문 텍스트
        k: 검색할 최대 결과 수

    Returns:
        관련 문서 리스트
    """
    try:
        res = col.query(query_texts=[question], n_results=k)
        return res.get("documents", [[]])[0]
    except Exception as e:
        st.error(f"검색 실패: {e}")
        return []


def _gen_answer(
    question: str,
    session_docs: List[str],
    session_digest: str = ""
) -> str:
    """
    근거 문서를 기반으로 AI 답변 생성

    Args:
        question: 사용자 질문
        session_docs: 검색된 근거 문서 리스트
        session_digest: 세션 요약 (선택)

    Returns:
        AI 답변 텍스트
    """
    try:
        genai.configure(api_key=st.secrets["gemini"]["api_key"])
        model_name = st.secrets["gemini"].get("generation_model", "gemini-1.5-flash")
        model = genai.GenerativeModel(model_name)

        # 프롬프트 구성
        prompt = f"""아래 '세션요약'과 '근거'만 사용해서 간결하게 답하세요.

**답변 규칙:**
- 수치/센터/SKU를 문장에 명시적으로 포함
- 근거에 없는 내용은 추정하지 말 것
- 2-4문장으로 간결하게
- 한국어로 작성

[세션요약]
{session_digest}

[근거 데이터]
{chr(10).join(f"- {d}" for d in session_docs)}

질문: {question}
"""

        response = model.generate_content(prompt)

        # 사용량 추적
        if "ai_usage" not in st.session_state:
            st.session_state["ai_usage"] = {"embeddings": 0, "generations": 0, "total_docs": 0}
        st.session_state["ai_usage"]["generations"] += 1

        return response.text

    except Exception as e:
        st.error(f"답변 생성 실패: {e}")
        # 폴백: 근거만 나열
        return (
            f"⚠️ 답변 생성 중 오류가 발생했습니다.\n\n"
            f"**관련 데이터:**\n" +
            "\n".join(f"• {d}" for d in session_docs[:3])
        )


# ============================================================================
# 5. UI 통합 함수
# ============================================================================

def render_ai_chatbot_tab(
    snapshot_df: pd.DataFrame,
    selected_centers: List[str],
    selected_skus: List[str]
):
    """
    AI 챗봇 탭 렌더링

    Args:
        snapshot_df: 전체 스냅샷 데이터
        selected_centers: 선택된 센터 리스트
        selected_skus: 선택된 SKU 리스트
    """
    st.subheader("🤖 세션 벡터검색 + Gemini 요약 (MVP)")

    # 필터링된 스냅샷
    snap = snapshot_df.copy()
    if "center" in snap.columns:
        snap = snap[snap["center"].astype(str).isin(selected_centers)]
    if "resource_code" in snap.columns:
        snap = snap[snap["resource_code"].astype(str).isin(selected_skus)]

    if snap.empty:
        st.warning("선택된 필터에 해당하는 데이터가 없습니다")
        return

    # 세션 요약 생성
    session_digest = (
        f"총 재고={pd.to_numeric(snap.get('stock_qty'), errors='coerce').fillna(0).sum():,.0f}개 / "
        f"센터 {snap.get('center', pd.Series([], dtype=str)).nunique()}곳 / "
        f"SKU {snap.get('resource_code', pd.Series([], dtype=str)).nunique()}개 / "
        f"최신일={pd.to_datetime(snap.get('date'), errors='coerce').max():%Y-%m-%d}"
    )

    # 필터 해시 계산
    filter_hash = _get_filter_hash(selected_centers, selected_skus)

    # 자동 인덱싱 (필터 변경 시)
    if "ai_indexed" not in st.session_state or st.session_state.get("_last_filter_hash") != filter_hash:
        with st.spinner("📚 현재 필터 범위 인덱싱 중..."):
            try:
                col, n = _ensure_session_index(snap, filter_hash, max_rows=2000)
                if col and n > 0:
                    st.session_state["ai_indexed"] = True
                    st.success(
                        f"✅ {n:,}건 인덱싱 완료 "
                        f"(센터: {len(selected_centers)}, SKU: {len(selected_skus)})"
                    )
                else:
                    st.session_state["ai_indexed"] = False
            except Exception as e:
                st.error(f"인덱싱 실패: {e}")
                st.session_state["ai_indexed"] = False

    # 세션 정보 표시
    with st.expander("ℹ️ 세션 정보", expanded=False):
        st.caption(session_digest)
        usage = st.session_state.get("ai_usage", {})
        st.caption(f"임베딩: {usage.get('embeddings', 0):,}건 | 답변 생성: {usage.get('generations', 0):,}회")

    # 수동 재인덱싱 버튼
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🔄 재인덱싱", help="필터는 자동으로 반영됩니다"):
            st.session_state.pop("_last_filter_hash", None)
            st.rerun()

    st.divider()

    # 질문 입력
    q = st.text_input(
        "질문을 입력하세요",
        placeholder="예: REVVN의 BA00021 재고는 얼마인가요?",
        key="ai_q"
    )

    col1, col2 = st.columns([4, 1])
    with col1:
        k = st.slider("검색 결과 수", min_value=1, max_value=10, value=5, step=1)

    if st.button("💬 질문하기", type="primary") and q:
        if not st.session_state.get("ai_indexed"):
            st.warning("먼저 인덱싱을 완료해주세요 (자동으로 시도 중)")
            return

        try:
            client = _chroma_cloud()
            if client is None:
                st.error("Chroma Cloud 연결 실패")
                return

            col_name = _build_session_collection_name()
            col = client.get_collection(col_name)

        except Exception as e:
            st.warning(f"컬렉션 로드 실패: {e}. 재인덱싱을 시도하세요.")
            return

        with st.spinner("🔍 검색 및 답변 생성 중..."):
            # 검색
            docs = _query_session(col, q, k=k)

            if not docs:
                st.info("관련 근거를 찾지 못했습니다. 필터를 조정하거나 질문을 바꿔보세요.")
            else:
                # 답변 생성
                ans = _gen_answer(q, docs, session_digest)

                # 결과 표시
                st.markdown("### 답변")
                st.markdown(ans)

                # 근거 표시
                with st.expander("🔎 사용된 근거 보기"):
                    for i, d in enumerate(docs, 1):
                        st.write(f"{i}. {d}")

    # 예시 질문
    st.divider()
    st.caption("**예시 질문:**")
    example_questions = [
        "REVVN 센터의 총 재고는?",
        "BA00021 SKU의 센터별 재고 현황",
        "재고가 가장 많은 센터는?",
        "최근 스냅샷 날짜는?",
    ]
    for eq in example_questions:
        st.caption(f"• {eq}")
