# AI 챗봇 v9_app.py 통합 가이드

## 1. 의존성 추가

### requirements.txt에 추가:
```txt
# AI/ML
chromadb>=0.5,<0.6
google-generativeai>=0.6,<0.7
duckdb>=1.0,<2.0
```

### 설치:
```bash
pip install chromadb google-generativeai duckdb
```

## 2. Secrets 설정

### .streamlit/secrets.toml에 추가:
```toml
[gemini]
api_key = "YOUR_GEMINI_API_KEY"
generation_model = "gemini-1.5-flash"
embedding_model = "text-embedding-004"

[chroma]
api_key = "YOUR_CHROMA_CLOUD_API_KEY"
tenant = "1468ca5f-87c8-4184-b5ec-bfcb1d03546b"
database = "SCM_DASHBOARD"
```

**⚠️ 보안 주의:**
- `.streamlit/secrets.toml`을 `.gitignore`에 추가
- Streamlit Cloud에서는 웹 UI로 secrets 추가

## 3. v9_app.py 수정

### 3-1. 상단에 임포트 추가:

```python
# 기존 임포트 아래에 추가
from ai_chatbot_improved import render_ai_chatbot_tab
```

### 3-2. main() 함수 내 탭 구성 변경:

기존 코드를 찾아서:
```python
# 현재는 탭이 없는 구조
st.subheader("요약 KPI")
# ... 기존 코드 ...
```

다음과 같이 변경:
```python
# 탭 구성 추가
tab_dashboard, tab_ai = st.tabs(["📊 대시보드", "🤖 AI 베타"])

with tab_dashboard:
    # ========================================
    # 9단계: KPI 요약 카드 렌더링
    # ========================================
    st.subheader("요약 KPI")
    # ... 기존 코드 모두 이 블록 안으로 이동 ...

    # ========================================
    # 17단계: 로트 상세까지 모든 코드
    # ========================================
    render_lot_details(...)

# AI 탭 추가
with tab_ai:
    render_ai_chatbot_tab(
        snapshot_df=snapshot_df,
        selected_centers=selected_centers,
        selected_skus=selected_skus
    )
```

### 3-3. 완전한 통합 예시:

```python
def main() -> None:
    """v9 대시보드 메인 함수"""
    logger.info("SCM Dashboard v9 시작")

    # 1-8단계: 기존 설정/로드/필터 코드 그대로
    st.set_page_config(page_title="SCM Dashboard v9", layout="wide")
    st.title("SCM Dashboard v9")
    # ... 데이터 로드, 필터 설정 등 ...

    # 탭 구성
    tab_dashboard, tab_ai = st.tabs(["📊 대시보드", "🤖 AI 베타"])

    # ============================================================================
    # 대시보드 탭 (기존 코드)
    # ============================================================================
    with tab_dashboard:
        st.subheader("요약 KPI")
        render_sku_summary_cards(...)
        st.divider()

        # 타임라인 빌드
        timeline_actual = build_core_timeline(...)
        # ... 나머지 모든 기존 코드 ...

        render_lot_details(...)

    # ============================================================================
    # AI 탭 (새로 추가)
    # ============================================================================
    with tab_ai:
        render_ai_chatbot_tab(
            snapshot_df=snapshot_df,
            selected_centers=selected_centers,
            selected_skus=selected_skus
        )
```

## 4. 테스트 시나리오

### 4-1. 로컬 테스트:
```bash
streamlit run v9_app.py
```

### 4-2. 확인 항목:
- [ ] AI 탭이 정상적으로 표시됨
- [ ] 필터 변경 시 자동 재인덱싱
- [ ] 질문 입력 후 답변 생성
- [ ] 근거 문서 확인 가능
- [ ] 에러 발생 시 우아한 메시지

### 4-3. 예시 질문:
- "REVVN 센터의 총 재고는?"
- "BA00021 SKU의 센터별 재고 현황"
- "재고가 가장 많은 센터는?"

## 5. 비용 관리

### 5-1. 초기 설정 (보수적):
- `max_rows`: 2000 (스냅샷 최대 행 수)
- `batch_size`: 100 (임베딩 배치 크기)
- `k`: 5 (검색 결과 수)

### 5-2. 예상 비용 (Gemini API 기준):
- 임베딩: 2000건 × $0.00002 = $0.04
- 생성: 1회 × $0.0001 = $0.0001
- 세션당 총: ~$0.05

### 5-3. 비용 모니터링:
```python
# 사이드바에 추가
with st.sidebar:
    st.divider()
    st.caption("🤖 AI 사용량 (이번 세션)")
    usage = st.session_state.get("ai_usage", {})
    st.caption(f"임베딩: {usage.get('embeddings', 0):,}건")
    st.caption(f"답변 생성: {usage.get('generations', 0):,}회")
    st.caption(f"예상 비용: ${usage.get('embeddings', 0) * 0.00002:.4f}")
```

## 6. 문제 해결

### 6-1. "Chroma Cloud 연결 실패"
- `secrets.toml`에 올바른 API 키 확인
- 인터넷 연결 확인
- Chroma Cloud 대시보드에서 tenant/database 확인

### 6-2. "임베딩 실패"
- Gemini API 키 확인
- API 쿼터 확인 (Google AI Studio)
- 배치 크기를 50으로 줄여보기

### 6-3. "컬렉션 로드 실패"
- 재인덱싱 버튼 클릭
- 브라우저 새로고침 (Ctrl+R)
- 세션 상태 초기화: `st.session_state.clear()`

## 7. 성능 최적화

### 7-1. 인덱싱 최적화:
- 필터 범위만 인덱싱 (캐싱 적용)
- 최대 2000행 제한
- 배치 임베딩 (100건씩)

### 7-2. 검색 최적화:
- Top-k를 5로 제한
- 메타데이터 필터링 활용 가능

### 7-3. 메모리 최적화:
- 세션별 에페메럴 컬렉션
- 임베딩 결과 캐싱 안 함 (메모리 절약)

## 8. 다음 단계

### 8-1. 1.5단계 (정량 계산):
- 판다스로 "총 재고", "센터별 합계" 등 직접 계산
- AI는 설명/권고만 생성
- 정확도↑, 비용↓

### 8-2. 2단계 (영구 맥락):
- 비즈니스 규칙 컬렉션 추가
- 센터 특성, 안전재고 기준 등
- 검색 시 세션 + 맥락 결합

### 8-3. 3단계 (모듈화):
- `scm_dashboard_v9/ai/` 폴더 생성
- `context_store.py`, `data_cache.py`, `qa.py` 분리
- 테스트 코드 추가

## 9. 체크리스트

통합 전:
- [ ] requirements.txt 업데이트
- [ ] secrets.toml 설정
- [ ] ai_chatbot_improved.py 파일 위치 확인

통합 중:
- [ ] v9_app.py에 임포트 추가
- [ ] 탭 구성 변경
- [ ] render_ai_chatbot_tab() 호출

통합 후:
- [ ] 로컬 테스트
- [ ] 예시 질문 테스트
- [ ] 에러 처리 확인
- [ ] 비용 모니터링 확인

배포:
- [ ] Streamlit Cloud에 secrets 추가
- [ ] 배포 후 동작 확인
- [ ] 사용자 가이드 작성
