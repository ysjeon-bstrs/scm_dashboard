# AI Chatbot 작업 이어가기 가이드

## 📍 현재 작업 상태

**브랜치**: `claude/add-ai-chatbot-feature-011CUouQxZW1odJrZYZqQuFX`
**최신 커밋**: `0f3f3ad` (2025-11-08)
**작업 단계**: ✅ 개발 완료 + 서브 에이전트 리뷰 완료 → 🔧 개선 사항 적용 대기 중

---

## 🎯 작업 시작 방법

```bash
# 1. 브랜치 체크아웃
git checkout claude/add-ai-chatbot-feature-011CUouQxZW1odJrZYZqQuFX

# 2. 최신 커밋 확인
git log --oneline -5
# 0f3f3ad (HEAD) Add sub-agent generated review artifacts
# d3501e1 Add comprehensive code review report for AI Chatbot
# 34c49aa Add AI Chatbot Sub-Agents System
# 4062858 Add detailed product roadmap for AI Chatbot
# c8f5033 Add comprehensive PRD for AI Chatbot feature

# 3. 현재 상태 확인
git status  # Should be clean
```

---

## 📚 필수 참고 문서 (읽는 순서)

### 1. **코드 리뷰 결과** (제일 먼저 읽기) ⭐⭐⭐
**파일**: `docs/chatbot_code_review_report.md`

**내용**:
- 4개 핵심 함수 리뷰 결과
- P0/P1/P2 이슈 분류
- Quick Wins (1시간 이내 적용 가능)
- 성능 개선 효과 예상치

**핵심 요약**:
- 현재 점수: 5.9/10
- 개선 후: 8.8/10
- 주요 이슈: detect_stockout_risks 성능 1000배 개선 가능
- 즉시 적용 필요: Phase 1 Quick Wins (1시간)

---

### 2. **서브 에이전트 사용 가이드**
**파일**: `docs/ai_chatbot_agents_guide.md`

**내용**:
- 6개 서브 에이전트 설명 (Function Reviewer, Test Generator 등)
- 사용법과 예시
- 워크플로우

**언제 참고**:
- 새 함수 추가 시 (Phase 3 구현)
- 코드 리뷰 필요 시
- 테스트 생성 필요 시

---

### 3. **PRD (Product Requirements Document)**
**파일**: `docs/prd_ai_chatbot.md`

**내용**:
- 전체 기능 명세
- 아키텍처 (v1.0 → v2.0 진화)
- 6대 주요 기능 상세 설명
- 9개 함수 API 레퍼런스

**언제 참고**:
- 기능 이해 필요 시
- 함수 동작 원리 확인 시

---

### 4. **로드맵**
**파일**: `docs/roadmap_ai_chatbot.md`

**내용**:
- Phase 3: Conversational AI (Q1 2026)
  - Multi-turn context
  - What-if scenarios (5개 새 함수)
  - Action recommendations
- Phase 4: Collaboration (Q2 2026)
- Phase 5: Enterprise Scale (Q3 2026)

**언제 참고**:
- Phase 3 구현 시작 시
- 우선순위 확인 시

---

## 🔧 다음 작업 (우선순위 순)

### 🔴 Phase 1: Quick Wins (즉시 적용 - 1시간)

**목표**: 코드 리뷰에서 발견된 P0 이슈 수정

#### Task 1.1: detect_stockout_risks 벡터화 (30분) ⭐
**파일**: `ai_chatbot_simple.py` 라인 668-682

**참고**:
- `docs/chatbot_code_review_report.md` → "Quick Win 4"
- `detect_stockout_risks_improved.py` (개선 버전 예시)

**수정 내용**:
```python
# Before: 반복문 (2-3초)
for sku in daily_sales.index:
    current_stock = snapshot_df[snapshot_df["resource_code"] == sku]["stock_qty"].sum()

# After: 벡터화 (2-3ms)
current_stock = snapshot_df.groupby("resource_code")["stock_qty"].sum()
```

**예상 효과**: 96% 성능 향상

---

#### Task 1.2: safe_float 헬퍼 추가 (2분)
**파일**: `ai_chatbot_simple.py` 상단

**참고**: `docs/chatbot_code_review_report.md` → "Quick Win 2"

**추가 코드**:
```python
import math

def safe_float(value):
    """NaN, Inf를 안전하게 처리"""
    if pd.isna(value) or math.isinf(value):
        return None
    return float(value)
```

**적용 위치**: 모든 `float()` 호출을 `safe_float()`로 교체

---

#### Task 1.3: prepare_minimal_metadata copy() 제거 (5분)
**파일**: `ai_chatbot_simple.py` 라인 54-73

**참고**: `docs/chatbot_code_review_report.md` → "Quick Win 1"

**수정**:
```python
# Before: 불필요한 복사 (50MB)
snapshot_copy = snapshot_df.copy()
snapshot_copy["date"] = pd.to_datetime(...)

# After: 직접 변환
dates = pd.to_datetime(snapshot_df["date"], errors="coerce")
```

**예상 효과**: 메모리 80% 절감

---

#### Task 1.4: None 체크 추가 (1분)
**파일**: `ai_chatbot_simple.py` 라인 29

**수정**:
```python
# Before
if snapshot_df.empty:

# After
if snapshot_df is None or snapshot_df.empty:
```

---

### 🟡 Phase 2: 안전성 강화 (이번 주 - 2시간)

#### Task 2.1: ask_ai_with_functions IndexError 방지
**파일**: `ai_chatbot_simple.py` 라인 546

**참고**: `docs/chatbot_code_review_report.md` → "P0 Critical" 섹션

---

#### Task 2.2: validate_columns 헬퍼 추가
**참고**: `docs/chatbot_code_review_report.md` → "P1 High Priority" 섹션

---

#### Task 2.3: max_iterations 조정
**파일**: `ai_chatbot_simple.py` 라인 526
**수정**: `max_iterations=5` → `max_iterations=3`

---

### 🟢 Phase 3: 새 기능 구현 (향후)

**참고**: `docs/roadmap_ai_chatbot.md` → Phase 3

새 함수 5개:
1. `simulate_demand_change()` - 수요 변동 시뮬레이션
2. `simulate_supply_delay()` - 공급 지연 시뮬레이션
3. `simulate_inbound()` - 입고 시뮬레이션
4. `simulate_promotion_impact()` - 프로모션 영향 예측
5. `generate_action_recommendations()` - 액션 추천

**작업 방식**:
```python
from ai_chatbot_agents import full_review_pipeline

# 각 함수 구현 후 전체 리뷰 파이프라인 실행
prompt = full_review_pipeline("simulate_demand_change")
# Task tool로 실행
```

---

## 🛠️ 유용한 헬퍼 도구

### 1. 서브 에이전트 사용 (Python)
**파일**: `ai_chatbot_agents.py`

```python
from ai_chatbot_agents import (
    review_function,      # 함수 리뷰
    generate_tests,       # 테스트 생성
    analyze_performance   # 성능 분석
)

# 예시: 수정 후 재검토
prompt = review_function("detect_stockout_risks")
# Task tool에 전달
```

---

### 2. 성능 벤치마크
**파일**: `test_performance_comparison.py`

```bash
# 개선 전/후 성능 비교
python test_performance_comparison.py
```

---

## 📊 현재 상태 요약

### 완료된 작업 ✅
1. ✅ AI 챗봇 v2.0 구현 (Function Calling)
2. ✅ 6대 주요 기능 완료
   - Proactive insights
   - Follow-up questions
   - Auto chart generation
   - NLP entity extraction
   - Function calling (9개 함수)
   - Minimal metadata
3. ✅ 서브 에이전트 시스템 구축
4. ✅ 코드 리뷰 완료 (4개 핵심 함수)
5. ✅ PRD, 로드맵, 가이드 문서화

### 대기 중인 작업 🔧
1. 🔴 Phase 1 Quick Wins 적용 (1시간)
2. 🟡 Phase 2 안전성 강화 (2시간)
3. 🟢 Phase 3 신규 기능 (향후)

### 현재 점수
- **코드 품질**: 5.9/10 → 8.8/10 (Phase 1 적용 후)
- **성능**: 평균 1.8초 → 1.0초 (Phase 1 적용 후)
- **메모리**: 50MB → 10MB (Phase 1 적용 후)

---

## 💬 작업 시작 시 권장 명령어

```bash
# 1. 브랜치 확인
git checkout claude/add-ai-chatbot-feature-011CUouQxZW1odJrZYZqQuFX

# 2. 최신 상태 확인
git pull origin claude/add-ai-chatbot-feature-011CUouQxZW1odJrZYZqQuFX

# 3. 코드 리뷰 리포트 읽기
cat docs/chatbot_code_review_report.md | head -100

# 4. 주요 파일 확인
ls -lh ai_chatbot*.py docs/

# 5. Phase 1 작업 시작
vim ai_chatbot_simple.py  # 또는 선호하는 에디터
```

---

## 🎯 작업 완료 후 체크리스트

Phase 1 완료 후:
- [ ] detect_stockout_risks 벡터화 완료
- [ ] safe_float 헬퍼 추가 완료
- [ ] copy() 제거 완료
- [ ] None 체크 추가 완료
- [ ] 커밋: "Apply Phase 1 Quick Wins from code review"
- [ ] 푸시: `git push origin claude/add-ai-chatbot-feature-011CUouQxZW1odJrZYZqQuFX`
- [ ] 재검토: `review_function("detect_stockout_risks")` 실행
- [ ] 성능 테스트: `python test_performance_comparison.py`

---

## 🔗 관련 링크

- **메인 코드**: `ai_chatbot_simple.py`
- **헬퍼 모듈**: `ai_chatbot_agents.py`
- **메인 앱**: `v9_app.py` (라인 1019에서 챗봇 호출)

---

## ❓ 문제 발생 시

1. **코드 리뷰 리포트 참고**: `docs/chatbot_code_review_report.md`
2. **서브 에이전트로 재검토**:
   ```python
   from ai_chatbot_agents import review_function
   prompt = review_function("문제_함수명")
   ```
3. **개선 예시 코드**: `detect_stockout_risks_improved.py`

---

**작성일**: 2025-11-08
**문서 버전**: 1.0
**최신 커밋**: 0f3f3ad
