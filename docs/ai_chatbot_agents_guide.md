# AI Chatbot Sub-Agents 사용 가이드

## 📋 개요

AI 챗봇의 개발, 리뷰, 테스트, 최적화를 위한 **6개의 전문 서브 에이전트 시스템**입니다.

각 에이전트는 특정 작업에 특화되어 있으며, `ai_chatbot_agents.py` 헬퍼 모듈을 통해 쉽게 호출할 수 있습니다.

---

## 🤖 서브 에이전트 목록

| 에이전트 | 역할 | 사용 시점 | 우선순위 |
|---------|-----|----------|---------|
| **Function Reviewer** | 함수 코드 리뷰 | 새 함수 구현 후 | 🔴 필수 |
| **Test Generator** | 테스트 케이스 생성 | 함수 구현 완료 후 | 🔴 필수 |
| **Prompt Optimizer** | 프롬프트 최적화 | 토큰 절약 필요 시 | 🟡 권장 |
| **Performance Analyzer** | 성능 분석 | 느린 응답 발견 시 | 🟡 권장 |
| **Integration Tester** | 통합 테스트 | 배포 전 | 🔴 필수 |
| **Documentation Writer** | 문서 자동 생성 | 함수 완성 후 | 🟢 선택 |

---

## 🚀 빠른 시작

### 설치
```bash
# 이미 scm_dashboard 디렉토리에 포함됨
cd /home/user/scm_dashboard
```

### 기본 사용법

```python
# Python에서 직접 사용
from ai_chatbot_agents import review_function, generate_tests

# 함수 리뷰 프롬프트 생성
prompt = review_function("calculate_stockout_days")
print(prompt)

# 이 프롬프트를 Claude Code의 Task tool에 전달
```

**또는 Claude Code에서 직접:**

```
User: @ai_chatbot_agents.py의 review_function("calculate_stockout_days")를
      Task tool로 실행해줘
```

---

## 📖 상세 가이드

### 1. Function Reviewer 🔍

**목적**: 함수 코드의 품질, 안전성, 성능을 종합 리뷰

**검토 항목:**
- ✅ 함수 시그니처 (타입 힌트, 기본값)
- ✅ 에러 핸들링 (try-except, 엣지 케이스)
- ✅ 데이터 정합성 (None 체크, 필수 컬럼)
- ✅ 성능 (벡터화, 메모리 효율)
- ✅ Gemini Function Calling 규격 준수

**사용 예시:**

```python
from ai_chatbot_agents import review_function

# 기본 사용
prompt = review_function("calculate_stockout_days")

# 특정 영역에 집중
prompt = review_function(
    function_name="get_sku_trend",
    focus_areas=["performance", "error_handling"]
)

# 다른 파일의 함수 리뷰
prompt = review_function(
    function_name="simulate_demand_change",
    file_path="ai_chatbot_scenarios.py"
)
```

**출력 형식:**
```markdown
## 함수 리뷰: calculate_stockout_days

### ✅ 잘된 점
- 명확한 타입 힌트 사용
- NaN 값 처리가 안전함

### ⚠️ 개선 필요
- [P0 Critical] 빈 DataFrame 처리 누락
- [P1 High] 성능: groupby 후 불필요한 copy()

### 🔧 수정 제안
[구체적인 Before/After 코드]

### 📊 평가
- 안전성: 7/10
- 성능: 6/10
- 종합: 7/10
```

**Workflow:**
```
새 함수 구현 → Function Reviewer 실행 → P0/P1 이슈 수정 → 재검토
```

---

### 2. Test Generator 🧪

**목적**: pytest 형식의 포괄적인 테스트 케이스 자동 생성

**테스트 유형:**
- ✅ Happy Path (정상 케이스)
- ✅ Edge Cases (빈 DataFrame, 단일 행, 경계값)
- ✅ Error Cases (None, 잘못된 타입, 누락 컬럼)
- ✅ Business Logic (계산 정확도, 정렬)

**사용 예시:**

```python
from ai_chatbot_agents import generate_tests

# 모든 테스트 타입 생성
prompt = generate_tests("calculate_stockout_days")

# 특정 테스트만 생성
prompt = generate_tests(
    function_name="get_sales_summary",
    test_types=["happy_path", "edge_cases"]
)
```

**출력 예시:**
```python
# tests/test_calculate_stockout_days.py
import pytest
import pandas as pd
from ai_chatbot_simple import calculate_stockout_days

class TestCalculateStockoutDays:

    @pytest.fixture
    def sample_snapshot(self):
        return pd.DataFrame({
            "resource_code": ["BA00021", "BA00022"],
            "stock_qty": [95, 200],
            ...
        })

    def test_happy_path(self, sample_snapshot, sample_moves):
        result = calculate_stockout_days("BA00021", sample_snapshot, sample_moves)
        assert result["days_until_stockout"] == pytest.approx(5.3, 0.1)
        assert result["status"] == "warning"

    def test_edge_case_zero_sales(self, sample_snapshot):
        # 판매량이 0이면 무한대 반환
        empty_moves = pd.DataFrame()
        result = calculate_stockout_days("BA00021", sample_snapshot, empty_moves)
        assert result["status"] == "no_sales_data"

    def test_error_case_sku_not_found(self, sample_snapshot, sample_moves):
        result = calculate_stockout_days("INVALID", sample_snapshot, sample_moves)
        assert "error" in result
```

**Workflow:**
```
함수 구현 → Test Generator 실행 → pytest 실행 → 실패한 테스트 수정
```

---

### 3. Prompt Optimizer 📝

**목적**: LLM 프롬프트의 명확성, 간결성, 토큰 효율 개선

**최적화 영역:**
- 🎯 명확성 (모호한 표현 제거, 구체적 예시)
- 📏 간결성 (불필요한 단어 제거, 중복 제거)
- 📊 구조화 (섹션 분리, 번호/불릿)
- 💰 토큰 효율 (핵심만 남기기)

**사용 예시:**

```python
from ai_chatbot_agents import optimize_prompt

# 시스템 프롬프트 최적화
prompt = optimize_prompt("시스템 프롬프트")

# 직접 프롬프트 텍스트 전달
current = """
당신은 SCM 대시보드 AI 어시스턴트입니다.
사용자의 질문에 답변해주세요. 재고, 판매, 예측 데이터를 활용하세요.
정확하게 답변하되 친절하게 대해주세요. 모르면 모른다고 하세요.
"""

prompt = optimize_prompt(
    prompt_name="시스템 프롬프트",
    current_prompt=current,
    goals=["reduce_tokens", "improve_clarity"]
)
```

**출력 예시:**
```markdown
## 프롬프트 최적화 분석

### 📊 현재 프롬프트 통계
- 토큰 수: ~120
- 명확성: 6/10
- 구조화: 4/10

### ⚠️ 문제점
1. 모호한 표현: "활용하세요" → 어떻게?
2. 중복: "답변" 반복
3. 비구조화: 섹션 없음

### ✅ 최적화된 프롬프트

\`\`\`
<role>SCM 대시보드 AI 어시스턴트</role>

<data>
- 재고: snapshot_df
- 판매: moves_df
- 예측: timeline_df
</data>

<rules>
1. 데이터 기반 정확한 답변
2. 확실하지 않으면 "데이터 부족" 명시
3. 숫자는 쉼표 구분 (예: 12,345개)
</rules>
\`\`\`

### 📉 개선 효과
- 토큰 절감: 120 → 85 (-29%)
- 명확성: 6 → 9 (+3)
- 구조화: 4 → 10 (+6)
```

---

### 4. Performance Analyzer 📊

**목적**: 토큰 사용량, 응답 속도, 비용 분석 및 최적화 제안

**분석 지표:**
- ⏱️ 응답 시간 (API, 데이터 준비, 함수 실행)
- 💰 토큰 사용량 (입력/출력)
- 🔧 함수 호출 패턴 (중복, 병목)
- 💾 캐싱 기회
- 🐌 병목 지점

**사용 예시:**

```python
from ai_chatbot_agents import analyze_performance

# 전체 챗봇 성능 분석
prompt = analyze_performance()

# 특정 기능 분석
prompt = analyze_performance(
    feature_name="품절 조회 기능",
    focus_metrics=["tokens", "latency"]
)

# 벤치마크 데이터 포함
benchmark = {
    "avg_tokens": 850,
    "avg_time_sec": 4.2,
    "function_calls_per_query": 2.5
}

prompt = analyze_performance(
    feature_name="What-if 시뮬레이션",
    benchmark_data=benchmark
)
```

**출력 예시:**
```markdown
## 성능 분석 리포트

### 📊 현재 성능
| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| 평균 응답 시간 | 4.2s | 3s | ⚠️ |
| 평균 토큰 사용 | 850 | 500 | ⚠️ |
| 함수 호출/쿼리 | 2.5 | 2.0 | ✅ |
| 일일 비용 (100쿼리) | $0.08 | $0.05 | ⚠️ |

### 🐌 병목 지점
1. prepare_data_context(): 1,200ms (전체의 29%)
   - 원인: 불필요한 DataFrame.copy() 3회
   - 해결: view 사용 또는 in-place 연산

2. get_sku_trend(): 800ms (19%)
   - 원인: 전체 timeline_df 순회
   - 해결: 특정 SKU만 필터링 후 연산

### 💡 최적화 제안
1. [High Impact] metadata를 session state에 캐싱
   → 응답 시간 -1.2s, 토큰 -350

2. [Medium Impact] 자주 호출되는 함수 결과 memoization
   → 응답 시간 -0.8s

3. [Low Impact] 프롬프트 압축
   → 토큰 -50

### 💰 예상 개선 효과
- 응답 시간: 4.2s → 2.2s (-48%)
- 토큰 사용: 850 → 450 (-47%)
- 비용 절감: $0.08 → $0.04 (-50%)
```

---

### 5. Integration Tester 🔗

**목적**: End-to-end 통합 테스트 및 시스템 일관성 검증

**테스트 시나리오:**
- 🔄 E2E Flow (질문 → 응답 → UI)
- 💬 Multi-turn Conversation (맥락 유지)
- ⚠️ Error Recovery (재시도, fallback)
- 📊 Data Consistency (DataFrame 정합성)
- 🖥️ UI Integration (차트, 버튼)

**사용 예시:**

```python
from ai_chatbot_agents import run_integration_tests

# 모든 시나리오 테스트
prompt = run_integration_tests()

# 특정 시나리오만
prompt = run_integration_tests(
    test_scenarios=["e2e_flow", "error_recovery"]
)

# UI 테스트 제외
prompt = run_integration_tests(
    test_scenarios=["multi_turn", "data_consistency"],
    include_ui=False
)
```

**출력 예시:**
```markdown
## 통합 테스트 결과

### ✅ Passed Tests (8/10)
1. ✅ E2E: 총 재고 조회
2. ✅ E2E: 품절 임박 SKU 검색
3. ✅ Error Recovery: API timeout 재시도
4. ✅ Data Consistency: snapshot + moves 일치
...

### ❌ Failed Tests (2/10)
1. ❌ Multi-turn: 맥락 상실
   - Input: "총 재고는?" → "그 중 AMZUS는?"
   - Expected: "AMZUS는 8,500개입니다"
   - Actual: "무엇의 AMZUS를 말씀하시나요?"
   - Root cause: chat_history not maintained in session_state
   - Fix: ai_chatbot_simple.py:145에 chat_history 추가

2. ❌ UI: 차트 생성 실패 (간헐적)
   - Frequency: 2/5 runs
   - Root cause: timeline_df에 is_forecast 컬럼 누락
   - Fix: v9_app.py:859에서 timeline_df 검증 추가

### 🔧 수정 필요 항목
- [ ] Multi-turn 대화 히스토리 구현 (Phase 3)
- [ ] timeline_df 스키마 검증 추가
```

---

### 6. Documentation Writer 📚

**목적**: 독스트링, 사용자 가이드, API 레퍼런스 자동 생성

**문서 유형:**
- 📝 Google 스타일 독스트링
- 📖 사용자 가이드 (마크다운)
- 📊 API 레퍼런스
- 🔗 관련 함수 매핑

**사용 예시:**

```python
from ai_chatbot_agents import generate_documentation

# 모든 문서 타입 생성
prompt = generate_documentation("calculate_stockout_days")

# 독스트링만
prompt = generate_documentation(
    function_name="get_sku_trend",
    doc_types=["docstring"]
)

# 사용자 가이드만
prompt = generate_documentation(
    function_name="simulate_demand_change",
    doc_types=["user_guide"]
)
```

**출력 예시:**

**독스트링:**
```python
def calculate_stockout_days(
    sku: str,
    snapshot_df: pd.DataFrame,
    moves_df: pd.DataFrame,
    days_lookback: int = 7
) -> dict:
    """
    특정 SKU가 품절될 때까지 남은 일수를 계산합니다.

    최근 N일간의 평균 판매량을 기반으로 현재 재고가 소진되는 시점을 예측합니다.
    판매량이 0이거나 데이터가 없는 경우 적절한 상태 메시지를 반환합니다.

    Args:
        sku (str): SKU 코드 (예: "BA00021")
        snapshot_df (pd.DataFrame): 현재 재고 스냅샷
        moves_df (pd.DataFrame): 이동 내역 (판매/입고)
        days_lookback (int, optional): 평균 계산 기간. Defaults to 7.

    Returns:
        dict: {
            "sku": (str) SKU 코드,
            "days_until_stockout": (float) 품절까지 남은 일수,
            "status": (str) "urgent" | "warning" | "ok" | "no_sales_data",
            "current_stock": (float) 현재 재고량,
            "daily_sales_avg": (float) 일평균 판매량
        }

    Raises:
        ValueError: sku가 snapshot_df에 없는 경우
        KeyError: 필수 컬럼 누락 시 ("stock_qty", "resource_code")

    Examples:
        >>> result = calculate_stockout_days("BA00021", snapshot, moves)
        >>> print(result["days_until_stockout"])
        5.3
        >>> print(result["status"])
        'warning'

    Notes:
        - 성능: O(n) where n = len(moves_df)
        - 제약: moves_df must have "move_type"=="OUT" for sales
        - 경고: 판매 데이터가 없으면 status="no_sales_data" 반환

    Version History:
        - v2.0: Function calling으로 전환, 반환값 JSON 직렬화 보장
        - v1.0: 초기 구현
    """
```

**사용자 가이드 (docs/functions/calculate_stockout_days.md):**
```markdown
## `calculate_stockout_days`

### 📝 설명
현재 재고와 최근 판매 추세를 기반으로 특정 SKU가 품절될 시점을 예측합니다.

### 🎯 사용 시점
- 품절 리스크를 사전에 파악하고 싶을 때
- 재주문 시점을 결정해야 할 때
- 안전재고 수준을 평가하고 싶을 때

### 📊 입력/출력

**Input:**
\`\`\`python
calculate_stockout_days(
    sku="BA00021",
    snapshot_df=current_inventory,
    moves_df=sales_history,
    days_lookback=7
)
\`\`\`

**Output:**
\`\`\`json
{
    "sku": "BA00021",
    "days_until_stockout": 5.3,
    "status": "warning",
    "current_stock": 95.0,
    "daily_sales_avg": 18.0
}
\`\`\`

### ⚠️ 주의사항
- 판매 데이터가 없으면 "no_sales_data" 상태 반환
- 재고가 이미 0이면 days_until_stockout = 0
- 프로모션 등으로 판매량이 급변할 경우 정확도 저하 가능

### 🔗 관련 함수
- `search_low_stock_skus()`: 여러 SKU를 한번에 검사
- `simulate_demand_change()`: What-if 시나리오 (Phase 3)
- `get_sales_summary()`: 판매 상세 내역 조회
```

---

## 🎯 통합 워크플로우

### Workflow 1: 새 함수 개발 (Full Pipeline)

```python
from ai_chatbot_agents import full_review_pipeline

# 1단계: 함수 구현 (직접 작성)
# ai_chatbot_simple.py에 simulate_demand_change() 추가

# 2단계: 전체 리뷰 파이프라인 실행
prompt = full_review_pipeline("simulate_demand_change")

# Task tool로 실행하면:
# → 코드 리뷰
# → 테스트 생성
# → 문서 생성
# → 통합 테스트
# → 최종 승인/거부 결정
```

**파이프라인 출력:**
```markdown
# simulate_demand_change 리뷰 완료

## 🎯 종합 평가
- 코드 품질: 8/10
- 테스트 커버리지: 92%
- 문서화: ✅
- 통합: ✅

## ✅ 승인 조건 충족 여부
- [x] P0 이슈 없음
- [x] 테스트 통과율 95% 이상
- [x] 문서화 완료
- [x] 통합 테스트 통과

## 🚀 배포 가능 여부
- **YES**

## 📋 남은 작업
- [x] 모두 완료
```

---

### Workflow 2: 성능 문제 해결

```python
from ai_chatbot_agents import analyze_performance, review_function

# 1단계: 성능 분석
perf_prompt = analyze_performance(
    feature_name="What-if 시뮬레이션",
    focus_metrics=["tokens", "latency", "bottlenecks"]
)

# Task tool 실행 → 병목 지점 발견
# 예: simulate_demand_change()가 느림

# 2단계: 해당 함수 집중 리뷰
review_prompt = review_function(
    function_name="simulate_demand_change",
    focus_areas=["performance"]
)

# Task tool 실행 → 최적화 제안 받음

# 3단계: 수정 후 재분석
perf_prompt_after = analyze_performance(
    feature_name="What-if 시뮬레이션",
    benchmark_data={"avg_time_sec": 2.1}  # 이전 4.2s
)
```

---

### Workflow 3: 프롬프트 최적화

```python
from ai_chatbot_agents import optimize_prompt, analyze_performance

# 1단계: 현재 토큰 사용량 확인
perf = analyze_performance(focus_metrics=["tokens"])

# 토큰이 많이 사용됨 발견 (평균 850)

# 2단계: 시스템 프롬프트 최적화
opt_prompt = optimize_prompt(
    prompt_name="시스템 프롬프트",
    goals=["reduce_tokens", "improve_clarity"]
)

# 3단계: 최적화 적용 후 재측정
perf_after = analyze_performance(
    benchmark_data={"avg_tokens": 450}  # 이전 850
)
```

---

## 🛠️ 고급 사용법

### 병렬 에이전트 실행

여러 에이전트를 동시에 실행하여 시간 절약:

```python
# Claude Code에서:
# Task 1: 코드 리뷰
# Task 2: 테스트 생성
# Task 3: 문서 생성
# → 3개를 한 메시지에서 병렬 실행
```

### 커스텀 체크리스트

특정 프로젝트 요구사항을 추가:

```python
custom_checklist = """
추가 검토 항목:
- SKU 형식이 BA[0-9]{5}인지 확인
- 모든 금액은 USD 기준인지 확인
- 날짜는 UTC 기준인지 확인
"""

prompt = review_function("new_function") + custom_checklist
```

### 에이전트 체이닝

한 에이전트의 출력을 다음 에이전트 입력으로:

```python
# 1. 리뷰 → P1 이슈 발견
review_result = "P1: 에러 핸들링 부족"

# 2. 리뷰 결과를 기반으로 테스트 생성
test_prompt = generate_tests("function_name") + f"""
특히 다음 이슈를 테스트하세요:
{review_result}
"""
```

---

## 📊 에이전트별 예상 실행 시간

| 에이전트 | 평균 시간 | 복잡도 |
|---------|----------|-------|
| Function Reviewer | 2-3분 | 함수 크기에 비례 |
| Test Generator | 3-5분 | 테스트 타입 개수 |
| Prompt Optimizer | 1-2분 | 프롬프트 길이 |
| Performance Analyzer | 3-4분 | 분석 범위 |
| Integration Tester | 5-10분 | 시나리오 개수 |
| Documentation Writer | 2-3분 | 문서 타입 |
| **Full Pipeline** | **15-25분** | 전체 |

---

## 🎓 Best Practices

### 1. 개발 단계별 에이전트 사용

| 단계 | 에이전트 | 목적 |
|-----|---------|-----|
| 설계 | Prompt Optimizer | 함수 설명 프롬프트 작성 |
| 구현 | - | 직접 코딩 |
| 리뷰 | Function Reviewer | 코드 품질 검증 |
| 테스트 | Test Generator | 테스트 자동 생성 |
| 문서화 | Documentation Writer | 독스트링/가이드 생성 |
| 통합 | Integration Tester | E2E 검증 |
| 최적화 | Performance Analyzer | 병목 제거 |
| 배포 | Full Pipeline | 최종 검증 |

### 2. 에이전트 결과 신뢰도

모든 에이전트 출력은 **제안**이지 **절대적 진실**이 아닙니다:

- ✅ **신뢰 가능**: 문법 오류, 타입 불일치, 명백한 버그
- ⚠️ **검토 필요**: 성능 추정치, 비즈니스 로직 검증
- ❌ **참고만**: 스타일 선호도, 주관적 평가

### 3. 반복 개선

첫 실행 결과가 완벽하지 않을 수 있음:

```
함수 구현 → 리뷰 (7/10) → 수정 → 재리뷰 (9/10) → 승인
```

### 4. 에이전트 조합

복잡한 문제는 여러 에이전트를 조합:

```python
# 느린 응답 문제
analyze_performance()  # 병목 발견
↓
review_function(focus=["performance"])  # 최적화 제안
↓
optimize_prompt()  # 토큰 절감
↓
analyze_performance()  # 개선 확인
```

---

## 🔧 문제 해결

### Q: 에이전트가 너무 일반적인 답변만 줍니다

**A**: 구체적인 컨텍스트를 추가하세요:

```python
# ❌ 나쁜 예
review_function("my_function")

# ✅ 좋은 예
review_function(
    function_name="calculate_stockout_days",
    focus_areas=["edge_cases", "NaT_handling"]
) + """
이 함수는 Gemini 2.0 Function Calling에서 사용됩니다.
반환값이 반드시 JSON 직렬화 가능해야 합니다.
"""
```

### Q: 생성된 테스트가 실패합니다

**A**: 정상입니다! 테스트가 버그를 발견한 것:

1. 실패 원인 파악
2. 함수 수정 (또는 테스트 수정)
3. 재실행

### Q: 에이전트가 코드를 읽지 못합니다

**A**: 파일 경로를 명확히 지정:

```python
review_function(
    function_name="new_function",
    file_path="/home/user/scm_dashboard/ai_chatbot_simple.py"
)
```

---

## 📚 참고 자료

- **PRD**: `docs/prd_ai_chatbot.md`
- **Roadmap**: `docs/roadmap_ai_chatbot.md`
- **Main Code**: `ai_chatbot_simple.py`
- **Helper Module**: `ai_chatbot_agents.py`

---

## 🚀 다음 단계

### Phase 3 구현 시

새 함수 추가 예정:
- `simulate_demand_change()`
- `simulate_supply_delay()`
- `generate_action_recommendations()`

각 함수마다:
1. **구현** → 2. **Full Pipeline** → 3. **승인** → 4. **배포**

---

**문서 버전**: 1.0
**작성일**: 2025-11-08
**다음 업데이트**: Phase 3 시작 시
