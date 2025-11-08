# AI Chatbot Roadmap

## 📋 Document Information

- **Product**: AI-Powered Chatbot Assistant for SCM Dashboard
- **Document Type**: Product Roadmap
- **Date**: 2025-11-08
- **Version**: 1.0
- **Status**: Planning Phase

---

## 🎯 Vision

SCM 대시보드 AI 챗봇을 단순 조회 도구에서 **능동적이고 실행 가능한 AI 어시스턴트**로 진화시킨다.

**From**: "데이터를 보여주는 챗봇"
**To**: "업무를 대신 처리하고 최적 의사결정을 제안하는 AI 파트너"

---

## 📊 Current Status (Phase 2 ✅)

### Completed Features (v2.0)

| Feature | Status | Impact |
|---------|--------|--------|
| Gemini 2.0 Function Calling | ✅ | 90% token savings |
| 9 KPI Functions | ✅ | 100% accuracy |
| Proactive Insights | ✅ | Auto-detect issues |
| Auto Chart Generation | ✅ | Visual answers |
| NLP Entity Extraction | ✅ | Auto-filtering |
| Follow-up Suggestions | ✅ | Conversation flow |

**Key Metrics:**
- Token usage: 5,000 → 500 (90% reduction)
- Response time: ~2.5s average
- Calculation accuracy: 100% (Python functions)

---

## 🗺️ Roadmap Overview

```
Phase 1 (✅ Complete)    Phase 2 (✅ Complete)    Phase 3 (🔄 Q1 2026)    Phase 4 (📅 Q2 2026)    Phase 5 (📅 Q3 2026)
─────────────────────   ─────────────────────   ────────────────────   ────────────────────   ────────────────────
│                       │                       │                     │                     │
│ Text-based RAG        │ Function Calling      │ Conversational AI   │ Collaboration       │ Enterprise Scale
│                       │                       │                     │                     │
├─ Basic Q&A           ├─ 9 Functions          ├─ Multi-turn         ├─ Feedback System    ├─ Automation
├─ Manual filtering    ├─ Minimal Metadata     ├─ What-if Scenarios  ├─ Bookmarks          ├─ External APIs
└─ Static answers      ├─ Proactive Insights   ├─ Action Recs        ├─ Templates          ├─ Multimodal
                       ├─ Auto Charts          └─ Standardization    └─ Sharing            └─ Mobile
                       └─ NLP Extraction
```

---

## 🚀 Phase 3: Conversational AI (Q1 2026)

**Timeline**: 2026년 1월 ~ 3월 (12주)
**Goal**: 대화형 AI로 자연스러운 멀티턴 인터랙션

---

### 3.1 Multi-turn Context Maintenance 🔴 High Priority

**Priority**: P0 (Critical)
**Effort**: 2주
**Impact**: 사용자 경험 혁신

#### Problem
현재는 매 질문마다 독립적으로 처리. 이전 대화 맥락이 유지되지 않음.

```
❌ Current:
User: "총 재고는?"
AI: "12,345개입니다"
User: "그 중에서 AMZUS는?"
AI: "무엇의 AMZUS를 말씀하시나요?" ← 맥락 상실
```

#### Solution
Gemini Chat History API를 사용하여 대화 히스토리 유지

```python
# Implementation
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 새 질문 처리
st.session_state.chat_history.append({
    "role": "user",
    "parts": [question]
})

# Gemini에 히스토리 전달
chat = model.start_chat(history=st.session_state.chat_history)
response = chat.send_message(new_question)

# 응답 저장
st.session_state.chat_history.append({
    "role": "model",
    "parts": [response.text]
})
```

#### Expected Behavior
```
✅ Improved:
User: "총 재고는?"
AI: "12,345개입니다"
User: "그 중에서 AMZUS는?"
AI: "AMZUS 센터의 재고는 8,500개입니다 (전체의 69%)" ← 맥락 유지
```

#### Implementation Tasks
- [ ] Session state에 chat_history 추가
- [ ] Gemini Chat API 통합
- [ ] 대화 히스토리 UI 표시
- [ ] 히스토리 초기화 버튼
- [ ] 컨텍스트 윈도우 관리 (최근 10턴만)
- [ ] 함수 호출 결과도 히스토리에 포함

#### Success Metrics
- 맥락 유지율: 95% 이상
- 평균 대화 턴 수: 1.5 → 4.0
- 사용자 만족도: +40%

---

### 3.2 What-if Scenario Simulation 🔴 High Priority

**Priority**: P0 (Critical)
**Effort**: 3주
**Impact**: 의사결정 지원 강화

#### Use Cases

**1. 수요 변동 시나리오**
```
User: "만약 BA00021 수요가 20% 증가하면 언제 품절되나?"

AI: [Function: simulate_demand_change(sku="BA00021", percent=20)]

📊 시나리오 분석:

기준선 (현재):
• 품절 예상: 5.3일 후
• 일평균 판매: 18개
• 현재 재고: 95개

시나리오 (수요 +20%):
• 품절 예상: 4.4일 후 ⚠️
• 일평균 판매: 21.6개 (↑3.6)
• 필요 재고: 114개 (↑19)

💡 권장 액션:
1. 즉시 재주문 필요 (안전재고 확보)
2. 대체 센터 확인 (KR01에 50개 가용)
```

**2. 공급 지연 시나리오**
```
User: "공급이 3일 지연되면?"

AI: [Function: simulate_supply_delay(sku="BA00021", days=3)]

⚠️ 리스크 분석:

정상 입고 (예정: 11월 11일):
• 품절 없음 (안전 마진 1.8일)

지연 입고 (11월 14일):
• 품절 발생: 11월 12일 ⛔
• 손실 예상: 2일분 판매 (36개)

💰 영향:
• 매출 손실: ~$1,080 (단가 $30 기준)
• 대체 공급 필요
```

**3. 프로모션 시나리오**
```
User: "BA00033을 세일하면 판매가 얼마나 늘어날까?"

AI: [Function: simulate_promotion_impact(sku="BA00033", type="20% off")]

📈 예상 영향 (과거 프로모션 데이터 기반):

정상가:
• 일평균 판매: 15개
• 재고 소진: 13.3일

20% 할인:
• 예상 판매: 35개/일 (+133%)
• 재고 소진: 5.7일 ⚠️
• 추가 재고 필요: 150개

💡 제안:
1. 프로모션 전 재주문 필수
2. 기간 제한 (5일 이내)
3. 센터 간 재배치 검토
```

#### New Functions to Implement

```python
# 1. 수요 변동 시뮬레이션
def simulate_demand_change(
    sku: str,
    percent: float,  # +20 또는 -20
    duration_days: int = 7
) -> dict:
    """
    Returns:
    {
        "baseline": {
            "days_until_stockout": 5.3,
            "daily_sales": 18.0,
            "current_stock": 95
        },
        "scenario": {
            "days_until_stockout": 4.4,
            "daily_sales": 21.6,
            "required_stock": 114,
            "shortage": 19
        },
        "recommendation": "immediate_reorder"
    }
    """

# 2. 공급 지연 시뮬레이션
def simulate_supply_delay(
    sku: str,
    delay_days: int,
    expected_quantity: int = None,  # None이면 평균 발주량
    expected_date: str = None  # None이면 다음 예정일
) -> dict:
    """
    Returns:
    {
        "stockout_date": "2025-11-12",
        "impact": {
            "lost_sales_units": 36,
            "lost_revenue_usd": 1080,
            "affected_days": 2
        },
        "mitigation": [
            {"action": "transfer_from_KR01", "quantity": 50},
            {"action": "expedite_shipping", "cost_usd": 200}
        ]
    }
    """

# 3. 입고 시뮬레이션
def simulate_inbound(
    sku: str,
    quantity: int,
    arrival_date: str
) -> dict:
    """
    Returns:
    {
        "before": {"days_until_stockout": 2.5},
        "after": {"days_until_stockout": 14.2},
        "impact": {
            "stockout_prevented": true,
            "coverage_days": 14.2,
            "optimal_order_qty": 200
        }
    }
    """

# 4. 프로모션 영향 예측
def simulate_promotion_impact(
    sku: str,
    promotion_type: str,  # "10% off", "BOGO", etc.
    duration_days: int = 7
) -> dict:
    """
    과거 프로모션 데이터 기반 예측
    Returns:
    {
        "sales_lift_percent": 133,
        "estimated_daily_sales": 35,
        "inventory_risk": "high",
        "required_additional_stock": 150,
        "recommended_duration_days": 5
    }
    """

# 5. 시나리오 비교
def compare_scenarios(
    baseline: dict,
    scenarios: list[dict]
) -> dict:
    """
    여러 시나리오를 나란히 비교
    Returns:
    {
        "comparison": [
            {"name": "현재", "stockout_days": 5.3, "rank": 2},
            {"name": "수요+20%", "stockout_days": 4.4, "rank": 3},
            {"name": "입고+100", "stockout_days": 10.9, "rank": 1}
        ],
        "best_scenario": "입고+100",
        "worst_scenario": "수요+20%"
    }
    """
```

#### Implementation Tasks
- [ ] 함수 5개 구현
- [ ] 과거 프로모션 데이터 수집 (moves_df에서)
- [ ] Gemini function declarations 업데이트
- [ ] UI: 시나리오 비교 테이블
- [ ] 차트: Before/After 비교 그래프

#### Success Metrics
- What-if 질문 처리율: 90% 이상
- 시나리오 정확도: ±10% 오차 이내
- 비즈니스 의사결정 활용률: 주 5회 이상

---

### 3.3 Action Recommendations 🟡 Medium Priority

**Priority**: P1
**Effort**: 2주
**Impact**: 조회 → 실행 전환

#### Problem
현재는 문제만 알려주고 해결책은 사용자가 직접 고민해야 함

#### Solution
AI가 구체적이고 실행 가능한 액션 플랜을 제안

#### Template Structure
```python
RECOMMENDATION_TEMPLATE = """
📊 현황: {current_situation}

⚠️ 문제: {problem_statement}

💡 권장 액션 (우선순위 순):

1. {action_1_title}
   • 내용: {action_1_description}
   • 예상 효과: {action_1_impact}
   • 실행 방법: {action_1_steps}
   • 소요 시간: {action_1_duration}
   • 담당자: {action_1_owner}

2. {action_2_title}
   • 내용: {action_2_description}
   • 예상 효과: {action_2_impact}
   ...

🔗 대체 옵션:
• {alternative_1}
• {alternative_2}

⏱️ 긴급도: {urgency_level}
📅 권장 실행 시점: {recommended_timing}
"""
```

#### Example Recommendations

**1. 품절 임박**
```
📊 현황: BA00021 재고 45개, 2.5일 후 품절 예상

⚠️ 문제: 정상 리드타임(10일)으로는 품절 불가피

💡 권장 액션:

1. 🚨 긴급 재주문 (우선순위: 최상)
   • 내용: 200개 긴급 발주 (익일 배송)
   • 예상 효과: 품절 방지, 14일분 재고 확보
   • 실행 방법:
     - 공급사: ABC Corp (긴급 배송 가능)
     - 단가: $32 (정상가 대비 +$2)
     - 도착 예정: 11월 9일
   • 소요 시간: 1시간 (발주 승인)
   • 담당자: 구매팀

2. 🔄 센터 간 이동 (우선순위: 높음)
   • 내용: KR01 → AMZUS 50개 이동
   • 예상 효과: 품절 시점 2.5일 → 5.3일 연장
   • 실행 방법:
     - Transfer Order 생성
     - 배송 소요: 2일
   • 소요 시간: 30분
   • 담당자: 물류팀

3. 📉 수요 조절 (우선순위: 중간)
   • 내용: 프로모션 일시 중단
   • 예상 효과: 판매량 -30% (18개 → 12.6개/일)
   • 실행 방법:
     - 마케팅팀과 협의
     - 프로모션 페이지 비활성화
   • 영향: 매출 감소 주의

🔗 대체 옵션:
• 유사 SKU로 교차 판매 (BA00022 재고 충분)
• 백오더 허용 (고객 대기 시간 5-7일)

⏱️ 긴급도: 🔴 최고 (24시간 내 결정 필요)
📅 권장 실행 시점: 오늘 중 (11월 8일 18시 이전)
```

**2. 재고 과다**
```
📊 현황: BA00055 재고 500개, 평균 판매 5개/일 → 100일분

⚠️ 문제: 과잉 재고로 인한 보관비 증가 및 유동성 악화

💡 권장 액션:

1. 🎯 프로모션 진행
   • 내용: 15% 할인 프로모션 (2주간)
   • 예상 효과: 일판매 5 → 12개 (+140%)
   • 예상 재고 감축: 168개 (2주 * 12개/일)
   • ROI: 할인 손실 < 보관비 절감

2. 📦 센터 재배치
   • 내용: AMZUS 300개 → 타 센터 분산
   • 목적: 보관비 절감 (AMZUS가 가장 비쌈)

3. 🔄 반품/리콜 검토
   • 조건: 공급사 반품 정책 확인 필요
```

#### New Functions

```python
def generate_action_recommendations(
    sku: str,
    issue_type: str,  # "stockout_risk" | "overstock" | "anomaly"
    context: dict
) -> dict:
    """
    Returns:
    {
        "current_situation": "...",
        "problem": "...",
        "actions": [
            {
                "priority": 1,
                "title": "긴급 재주문",
                "description": "...",
                "impact": "품절 방지, 14일분 확보",
                "steps": ["...", "..."],
                "duration": "1시간",
                "owner": "구매팀",
                "cost_usd": 6400,
                "risk": "low"
            },
            ...
        ],
        "alternatives": ["...", "..."],
        "urgency": "critical",
        "deadline": "2025-11-08T18:00:00"
    }
    """

def calculate_reorder_quantity(
    sku: str,
    lead_time_days: int,
    safety_stock_percent: float = 20.0
) -> dict:
    """
    최적 재주문 수량 계산
    Returns:
    {
        "recommended_quantity": 200,
        "breakdown": {
            "lead_time_consumption": 126,  # 7일 * 18개/일
            "safety_stock": 40,
            "buffer": 34
        },
        "cost_estimate_usd": 6000,
        "coverage_days": 14
    }
    """
```

#### Implementation Tasks
- [ ] Recommendation template 구현
- [ ] generate_action_recommendations() 함수
- [ ] calculate_reorder_quantity() 함수
- [ ] UI: 액션 카드 컴포넌트
- [ ] 액션 실행 추적 (선택사항)

---

### 3.4 Answer Template Standardization 🟢 Low Priority

**Priority**: P2
**Effort**: 1주
**Impact**: 일관성 및 가독성 향상

#### Problem
AI 답변 형식이 질문마다 다름. 중요 정보를 찾기 어려움.

#### Solution
구조화된 템플릿으로 일관성 확보

#### Standard Template

```markdown
📌 결론
[한 줄 핵심 요약]

📊 핵심 수치
• [지표 1]: [값] ([변화율])
• [지표 2]: [값] ([상태])
• [지표 3]: [값]

🔍 분석
[왜 이런 결과가 나왔는지 2-3줄 설명]

💡 다음 액션
1. [우선순위 1 액션]
2. [우선순위 2 액션]

🔗 근거
• 데이터 소스: [snapshot_df | moves_df | timeline_df]
• 계산 방법: [함수 호출 내역]
• 기준 날짜: [YYYY-MM-DD]
```

#### Examples

**Before (Unstructured):**
```
BA00021 재고는 95개이고, 최근 7일 평균 판매량이 18개라서
약 5.3일 후에 품절될 것 같습니다.
```

**After (Structured):**
```
📌 결론
BA00021은 5.3일 후 품절 예상 (⚠️ 경고)

📊 핵심 수치
• 현재 재고: 95개
• 일평균 판매: 18개 (최근 7일 기준)
• 품절 예상일: 2025-11-13
• 안전재고 미달: -25개

🔍 분석
최근 판매 추세가 지속될 경우 정상 리드타임(10일) 내 재입고가
불가능합니다. 안전재고 기준(5일분 = 90개)에도 미달합니다.

💡 다음 액션
1. 긴급 재주문 (200개, 익일 배송)
2. 센터 간 이동 검토 (KR01에 50개 가용)

🔗 근거
• 데이터: snapshot_df (11/7), moves_df (10/31~11/7)
• 함수: calculate_stockout_days("BA00021")
• 기준: 2025-11-07 23:59
```

#### Implementation
```python
def format_answer(
    conclusion: str,
    metrics: dict,
    analysis: str,
    actions: list[str],
    sources: dict
) -> str:
    template = f"""
📌 결론
{conclusion}

📊 핵심 수치
{format_metrics(metrics)}

🔍 분석
{analysis}

💡 다음 액션
{format_actions(actions)}

🔗 근거
{format_sources(sources)}
"""
    return template
```

---

## 🤝 Phase 4: Collaboration & Feedback (Q2 2026)

**Timeline**: 2026년 4월 ~ 6월 (12주)
**Goal**: 팀 협업 및 지속적 개선

---

### 4.1 Quality Feedback System 🟡 Medium Priority

**Priority**: P1
**Effort**: 2주

#### Features

**1. Thumbs Up/Down**
```python
col1, col2 = st.columns(2)
with col1:
    if st.button("👍 도움됨"):
        log_feedback(
            question=question,
            answer=answer,
            rating="positive",
            timestamp=datetime.now()
        )
        st.success("피드백 감사합니다!")

with col2:
    if st.button("👎 개선 필요"):
        reason = st.radio("이유 선택:", [
            "부정확한 답변",
            "너무 장황함",
            "응답이 느림",
            "이해하기 어려움",
            "필요한 정보 누락"
        ])
        detail = st.text_input("상세 의견 (선택)")
        log_feedback(
            question=question,
            answer=answer,
            rating="negative",
            reason=reason,
            detail=detail
        )
```

**2. Analytics Dashboard**
```python
# 주간 리포트
def generate_weekly_report():
    return {
        "total_queries": 245,
        "positive_rate": 0.87,  # 87%
        "avg_response_time": 2.3,
        "most_asked": [
            {"question": "총 재고", "count": 45},
            {"question": "품절 임박", "count": 38},
        ],
        "low_rated_patterns": [
            "복잡한 비교 질문",
            "과거 특정 날짜 데이터"
        ]
    }
```

**3. Regression Testing**
```python
BENCHMARK_QUESTIONS = [
    {
        "question": "총 재고는?",
        "expected_function": "get_total_stock",
        "expected_accuracy": ">= 99%"
    },
    {
        "question": "BA00021 품절 예상일",
        "expected_function": "calculate_stockout_days",
        "expected_response_time": "<= 3s"
    },
    # ... 100+ test cases
]

def run_regression_test():
    results = []
    for test in BENCHMARK_QUESTIONS:
        answer = ask_ai(test["question"])
        results.append({
            "test": test,
            "passed": evaluate(answer, test["expected"])
        })
    return results
```

---

### 4.2 Bookmark & Sharing 🟢 Low Priority

**Priority**: P2
**Effort**: 3주

#### Features

**1. Bookmark Q&A**
```python
if st.button("⭐ 북마크"):
    save_bookmark(
        question=question,
        answer=answer,
        tags=["재고", "품절"],
        created_by=current_user
    )
    st.success("북마크에 저장했습니다!")
```

**2. Share Link**
```python
share_url = f"https://dashboard.com/chat?id={chat_id}"
st.code(share_url, language="text")

if st.button("📋 링크 복사"):
    pyperclip.copy(share_url)
    st.success("링크가 복사되었습니다!")
```

**3. Q&A Library**
```
북마크 라이브러리
├─ 📁 재고 관리
│  ├─ ⭐ "품절 임박 SKU 찾기" (15회 재사용)
│  ├─ ⭐ "센터별 재고 분포" (12회)
│  └─ ⭐ "주간 재고 변화" (8회)
├─ 📁 판매 분석
│  └─ ⭐ "베스트 셀러 TOP 10" (20회)
└─ 📁 예측
   └─ ⭐ "다음 주 품절 리스크" (10회)
```

**4. Template Creation**
```python
# 매주 월요일 자동 실행
TEMPLATE = {
    "name": "주간 품절 리포트",
    "questions": [
        "7일 이내 품절 예상 SKU는?",
        "재주문 권장 SKU와 수량은?",
        "센터별 리스크 분포는?"
    ],
    "schedule": "every Monday 09:00",
    "recipients": ["scm-team@company.com"]
}
```

---

### 4.3 Team Comments 🟢 Low Priority

**Priority**: P2
**Effort**: 2주

```python
# Q&A에 댓글 기능
st.write("💬 팀원 댓글")
comments = get_comments(chat_id)
for c in comments:
    st.markdown(f"**{c['author']}** ({c['time']})")
    st.write(c['text'])

new_comment = st.text_area("댓글 추가")
if st.button("게시"):
    post_comment(chat_id, new_comment, current_user)
```

---

## 🚀 Phase 5: Enterprise Scale (Q3 2026)

**Timeline**: 2026년 7월 ~ 9월 (12주)
**Goal**: 엔터프라이즈급 자동화 및 통합

---

### 5.1 Automation Actions 🔴 High Priority

**Priority**: P0
**Effort**: 4주

#### Vision
조회를 넘어 **실제 시스템에 액션을 실행**

#### Safety Controls

```python
AUTOMATION_RULES = {
    "purchase_order": {
        "max_quantity": 500,  # 500개 이하만 자동 발주
        "max_value_usd": 10000,  # $10K 이하만
        "require_approval": True,  # 승인 필요
        "approvers": ["manager@company.com"]
    },
    "transfer_order": {
        "max_quantity": 100,
        "auto_approve": True  # 센터 이동은 자동 승인
    }
}
```

#### Example Flow

**1. Auto Purchase Order**
```
User: "BA00021 긴급 재주문 200개 해줘"

AI: [Detects action intent]
    → Function: create_purchase_order()

Safety Check:
✅ Quantity (200) < max (500)
✅ Value ($6,000) < max ($10,000)
⚠️ Requires approval

AI Response:
"발주서 초안을 생성했습니다.

📋 발주서 #PO-2025-1108-001
• SKU: BA00021
• 수량: 200개
• 공급사: ABC Corp
• 단가: $30
• 총액: $6,000
• 배송지: AMZUS
• 도착 예정: 11월 18일

⚠️ 승인이 필요합니다.
승인 요청을 manager@company.com에게 발송할까요?"

User: "응"

AI: [Send approval request]
    → Email sent
    → Status: Pending Approval

"✅ 승인 요청을 발송했습니다.
   승인되면 자동으로 ERP에 등록됩니다."
```

**2. Auto Transfer Order**
```
User: "BA00021 KR01에서 AMZUS로 50개 옮겨줘"

AI: [Function: create_transfer_order()]

Safety Check:
✅ Quantity (50) < max (100)
✅ Auto-approve enabled

AI: [Execute]
    → WMS API called
    → Transfer order created

"✅ 이동 지시서가 생성되었습니다.

📦 Transfer Order #TO-2025-1108-003
• SKU: BA00021
• From: KR01 (재고 충분: 150개)
• To: AMZUS
• 수량: 50개
• 예상 도착: 11월 10일
• 상태: 승인됨 → 물류팀 할당됨

[실시간 추적 링크]"
```

#### New Functions

```python
def create_purchase_order(
    sku: str,
    quantity: int,
    supplier: str = "default",
    delivery_center: str = "AMZUS",
    urgency: str = "normal",  # normal | urgent
    requested_by: str = None
) -> dict:
    """
    Returns:
    {
        "po_number": "PO-2025-1108-001",
        "status": "pending_approval" | "approved" | "sent",
        "total_cost_usd": 6000,
        "approval_required": true,
        "approval_sent_to": ["manager@company.com"],
        "estimated_delivery": "2025-11-18"
    }
    """

def create_transfer_order(
    sku: str,
    from_center: str,
    to_center: str,
    quantity: int
) -> dict:
    """
    WMS 시스템과 연동
    """

def send_notification(
    recipients: list[str],
    message: str,
    channel: str = "email",  # email | slack | teams
    priority: str = "normal"
) -> dict:
    """
    이메일/Slack 등으로 알림 발송
    """

def check_automation_safety(
    action: str,
    parameters: dict
) -> dict:
    """
    자동화 액션의 안전성 검증
    Returns:
    {
        "allowed": true,
        "requires_approval": true,
        "risk_level": "medium",
        "warnings": ["High value transaction"]
    }
    """
```

---

### 5.2 External Integrations 🟡 Medium Priority

**Priority**: P1
**Effort**: 4주

#### Target Systems

**1. ERP Integration (SAP/Oracle)**
```python
def sync_with_erp():
    """
    - 발주서 자동 등록
    - 입고 예정 조회
    - 재고 실사 결과 동기화
    """
```

**2. WMS Integration**
```python
def sync_with_wms():
    """
    - 실시간 재고 업데이트
    - 피킹/패킹 상태 조회
    - 센터 간 이동 지시
    """
```

**3. Messaging (Slack/Teams)**
```python
# Slack Bot
@slack_bot.command("/inventory")
def slack_inventory_command(sku):
    answer = ask_ai(f"{sku}의 재고는?")
    return answer

# 자동 알림
if stockout_risk_detected:
    slack_bot.post_message(
        channel="#scm-alerts",
        text=f"⚠️ {sku} 품절 임박 (2.5일)",
        attachments=[{
            "title": "상세 보기",
            "title_link": f"{dashboard_url}?sku={sku}"
        }]
    )
```

**4. Email Automation**
```python
# 일일 리포트
def send_daily_digest():
    report = generate_report([
        "품절 임박 SKU",
        "재주문 권장",
        "이상치 감지"
    ])

    send_email(
        to=["scm-team@company.com"],
        subject=f"SCM Daily Digest - {today}",
        html=render_template("daily_digest.html", data=report)
    )
```

---

### 5.3 Multimodal Input 🟢 Low Priority

**Priority**: P2
**Effort**: 3주

#### Use Cases

**1. Invoice Verification**
```python
uploaded_file = st.file_uploader("인보이스 업로드", type=["pdf", "jpg"])

if uploaded_file:
    # Gemini Vision API
    invoice_data = extract_invoice_data(uploaded_file)

    AI: "인보이스를 분석했습니다.

    📄 인보이스 내용:
    • SKU: BA00021
    • 수량: 200개
    • 단가: $30
    • 총액: $6,000
    • 공급사: ABC Corp

    🔍 시스템 비교:
    • 발주서 #PO-2025-1101-045
    • 예상 수량: 200개 ✅
    • 예상 금액: $6,000 ✅

    ✅ 모든 항목이 일치합니다. 입고 처리를 진행할까요?"
```

**2. Physical Inventory Check**
```python
# 사진으로 재고 실사
uploaded_image = st.camera_input("재고 사진 촬영")

AI: [Gemini Vision + OCR]
    "사진에서 BA00021 박스를 35개 감지했습니다.

    시스템 재고: 40개
    실사 재고: 35개
    차이: -5개 ⚠️

    재고 조정을 생성할까요?"
```

**3. Barcode/QR Scan**
```python
from streamlit_webrtc import webrtc_streamer

# 바코드 스캔으로 빠른 조회
scanned_code = barcode_scanner()
if scanned_code:
    answer = ask_ai(f"{scanned_code}의 재고는?")
```

---

### 5.4 Mobile App 🟢 Low Priority

**Priority**: P2
**Effort**: 6주

#### Features
- Progressive Web App (PWA)
- Push notifications
- Voice input
- Offline mode (cached queries)
- Quick actions (1-tap reorder)

```python
# Push notification 예시
if stockout_detected:
    send_push_notification(
        user_tokens=get_subscribed_users(),
        title="품절 임박 알림",
        body=f"{sku}가 2일 후 품절 예상됩니다",
        data={"sku": sku, "action": "view_details"}
    )
```

---

## 📊 Roadmap Summary Table

| Phase | Feature | Priority | Effort | Impact | Timeline |
|-------|---------|----------|--------|--------|----------|
| **3** | Multi-turn Context | 🔴 P0 | 2주 | High | 2026 Q1 |
| **3** | What-if Scenarios | 🔴 P0 | 3주 | High | 2026 Q1 |
| **3** | Action Recommendations | 🟡 P1 | 2주 | Medium | 2026 Q1 |
| **3** | Answer Templates | 🟢 P2 | 1주 | Low | 2026 Q1 |
| **4** | Feedback System | 🟡 P1 | 2주 | Medium | 2026 Q2 |
| **4** | Bookmarks & Sharing | 🟢 P2 | 3주 | Low | 2026 Q2 |
| **4** | Team Comments | 🟢 P2 | 2주 | Low | 2026 Q2 |
| **5** | Automation Actions | 🔴 P0 | 4주 | High | 2026 Q3 |
| **5** | ERP/WMS Integration | 🟡 P1 | 4주 | Medium | 2026 Q3 |
| **5** | Multimodal Input | 🟢 P2 | 3주 | Low | 2026 Q3 |
| **5** | Mobile App | 🟢 P2 | 6주 | Low | 2026 Q3 |

---

## 🎯 Success Metrics by Phase

### Phase 3 Targets
- Multi-turn conversations: 평균 4+ 턴
- What-if accuracy: ±10% 오차
- Action recommendation usage: 주 10회+
- Template compliance: 95%+

### Phase 4 Targets
- Feedback collection: 80%+ participation
- Bookmark usage: 월 50+ saves
- Team collaboration: 월 20+ shared Q&As

### Phase 5 Targets
- Automation success rate: 95%+
- ERP sync accuracy: 99%+
- Mobile DAU: 100+ users
- Push notification CTR: 40%+

---

## 🔧 Technical Prerequisites

### Phase 3
- Gemini Chat API 통합
- Session state management 강화
- Historical data analysis (과거 프로모션 데이터)

### Phase 4
- Database (feedback, bookmarks 저장)
- User authentication
- Sharing infrastructure

### Phase 5
- ERP/WMS API keys 및 권한
- Message queue (Celery/Redis)
- Mobile backend (Firebase)
- Approval workflow system

---

## 🚧 Risks & Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Gemini API rate limits | High | Implement caching, queue management |
| ERP integration complexity | High | Start with read-only, gradual rollout |
| Automation errors | Critical | Multi-layer safety checks, approval workflow |
| User adoption | Medium | Training, templates, quick wins |
| Data privacy | High | Encryption, access control, audit logs |

---

## 💰 Resource Requirements

### Phase 3
- 1 Backend Engineer (full-time)
- 1 Data Analyst (part-time)
- Gemini API costs: ~$500/month

### Phase 4
- 1 Full-stack Engineer
- 1 UX Designer (part-time)
- Database hosting: ~$200/month

### Phase 5
- 2 Backend Engineers
- 1 Mobile Developer
- 1 DevOps Engineer
- Infrastructure: ~$1,500/month

---

## 📝 Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2025-11-08 | Phase 3: Multi-turn first | Foundation for all future features |
| 2025-11-08 | Phase 5: Automation with approval | Safety first, trust building |
| 2025-11-08 | Mobile as Phase 5 | Desktop usage proven first |

---

## 📚 References

- [Gemini Chat API Docs](https://ai.google.dev/gemini-api/docs/chat)
- [Streamlit State Management](https://docs.streamlit.io/library/api-reference/session-state)
- [SAP API Integration](https://api.sap.com/)
- [Slack Bot Tutorial](https://api.slack.com/bot-users)

---

**Document Version**: 1.0
**Last Updated**: 2025-11-08
**Next Review**: 2026-01-01 (Q1 시작 전)

---

## Appendix: Future Ideas (Not Scheduled)

아직 우선순위에 들지 않았지만 향후 검토할 아이디어:

1. **AI Training Interface**: 사용자가 직접 AI를 학습시킬 수 있는 UI
2. **Custom Dashboards**: AI가 자동으로 맞춤 대시보드 생성
3. **Predictive Alerts**: 품절 전 3일이 아닌 예측 기반 알림
4. **Cross-functional Insights**: 재고 + 재무 + 마케팅 통합 분석
5. **Voice Assistant**: "Alexa, 총 재고 알려줘"
6. **AR Warehouse View**: AR로 창고 내 재고 위치 시각화
7. **Blockchain Tracking**: 공급망 투명성 확보

---

**End of Roadmap Document**
