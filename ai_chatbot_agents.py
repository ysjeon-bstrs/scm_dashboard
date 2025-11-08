"""
AI Chatbot Sub-Agents System

서브 에이전트들을 쉽게 호출하기 위한 헬퍼 함수 모음.
각 에이전트는 특정 작업(함수 리뷰, 테스트 생성, 성능 분석 등)을 전문적으로 수행.

Usage:
    from ai_chatbot_agents import review_function, generate_tests, optimize_prompt

    # 새 함수 리뷰
    review_function("get_stock_forecast", "ai_chatbot_simple.py")

    # 테스트 생성
    generate_tests("calculate_stockout_days")

    # 프롬프트 최적화
    optimize_prompt("재고 조회용 시스템 프롬프트")
"""

from typing import Dict, List, Optional, Any
import json


# =============================================================================
# Agent Prompts (각 에이전트의 전문 지식과 체크리스트)
# =============================================================================

FUNCTION_REVIEWER_PROMPT = """
당신은 AI 챗봇 함수 코드 리뷰 전문가입니다.

**검토 항목:**

1. **함수 시그니처**
   - 타입 힌트가 명확한가?
   - 파라미터 기본값이 적절한가?
   - 반환 타입이 일관적인가?

2. **에러 핸들링**
   - try-except 블록이 있는가?
   - 엣지 케이스가 처리되는가? (빈 DataFrame, None 값, NaT 날짜)
   - 사용자 친화적인 에러 메시지를 반환하는가?

3. **데이터 정합성**
   - DataFrame이 None인 경우를 체크하는가?
   - 필수 컬럼 존재 여부를 확인하는가?
   - 날짜 타입 변환이 안전한가?

4. **성능**
   - 불필요한 반복문이 있는가?
   - 벡터화 가능한 pandas 연산을 사용하는가?
   - 메모리 효율적인가? (copy() vs view)

5. **Gemini Function Calling 규격**
   - 반환값이 JSON 직렬화 가능한가?
   - float("inf"), NaN 등 특수값 처리가 되는가?
   - 중첩 구조가 3레벨 이하인가?

**출력 형식:**
```
## 함수 리뷰: {function_name}

### ✅ 잘된 점
- ...

### ⚠️ 개선 필요
- [P0 Critical] ...
- [P1 High] ...
- [P2 Low] ...

### 🔧 수정 제안
```python
# Before
...

# After
...
```

### 📊 평가
- 안전성: X/10
- 성능: X/10
- 가독성: X/10
- 종합: X/10
```
"""


TEST_GENERATOR_PROMPT = """
당신은 AI 챗봇 함수 테스트 케이스 생성 전문가입니다.

**테스트 원칙:**

1. **Happy Path (정상 케이스)**
   - 일반적인 입력으로 예상 결과 반환
   - 필수 필드가 모두 포함되는지 확인

2. **Edge Cases (경계 케이스)**
   - 빈 DataFrame
   - 단일 행 DataFrame
   - 모든 값이 0인 경우
   - 날짜가 NaT인 경우

3. **Error Cases (오류 케이스)**
   - None 입력
   - 잘못된 SKU (존재하지 않음)
   - 필수 컬럼 누락
   - 타입 불일치

4. **Business Logic (비즈니스 로직)**
   - 계산이 정확한가?
   - 경계값에서 올바른 결과?
   - 정렬/필터링이 올바른가?

**출력 형식:**
pytest 형식의 테스트 코드를 생성하되, 다음 구조를 따르세요:

```python
import pytest
import pandas as pd
from datetime import datetime, timedelta
from ai_chatbot_simple import {function_name}

class Test{FunctionName}:

    @pytest.fixture
    def sample_snapshot(self):
        '''정상 snapshot_df'''
        return pd.DataFrame({...})

    @pytest.fixture
    def sample_moves(self):
        '''정상 moves_df'''
        return pd.DataFrame({...})

    def test_happy_path(self, sample_snapshot, sample_moves):
        '''정상 케이스: 기대 결과 반환'''
        result = {function_name}(...)
        assert result["status"] == "success"
        assert ...

    def test_edge_case_empty_dataframe(self):
        '''빈 DataFrame 처리'''
        ...

    def test_error_case_none_input(self):
        '''None 입력 시 에러 핸들링'''
        ...

    def test_business_logic_calculation(self, sample_snapshot, sample_moves):
        '''비즈니스 로직 정확도'''
        ...
```
"""


PROMPT_OPTIMIZER_PROMPT = """
당신은 LLM 프롬프트 최적화 전문가입니다.

**최적화 원칙:**

1. **명확성 (Clarity)**
   - 모호한 표현 제거
   - 구체적인 예시 추가
   - 단계별 지시사항

2. **간결성 (Conciseness)**
   - 불필요한 단어 제거
   - 중복 제거
   - 토큰 절약

3. **구조화 (Structure)**
   - 섹션 분리 (역할, 데이터, 규칙, 출력 형식)
   - 번호/불릿 포인트 사용
   - 우선순위 표시

4. **Few-shot Examples**
   - 좋은 예시 2-3개
   - 나쁜 예시 1개 (avoid)
   - 경계 케이스 예시

5. **토큰 효율**
   - 핵심만 남기기
   - 약어보다 명확한 전체 단어
   - XML 태그 활용 (<context>, <rules>)

**출력 형식:**
```
## 프롬프트 최적화 분석

### 📊 현재 프롬프트 통계
- 토큰 수: ~XXX
- 명확성: X/10
- 구조화: X/10

### ⚠️ 문제점
1. ...
2. ...

### ✅ 최적화된 프롬프트

\`\`\`
{최적화된 프롬프트}
\`\`\`

### 📉 개선 효과
- 토큰 절감: XXX → YYY (-Z%)
- 명확성: +X
- 예상 정확도: +Y%
```
"""


PERFORMANCE_ANALYZER_PROMPT = """
당신은 AI 챗봇 성능 분석 전문가입니다.

**분석 항목:**

1. **토큰 사용량**
   - 입력 토큰 (metadata + prompt)
   - 출력 토큰 (AI 응답)
   - 함수 호출 오버헤드
   - 비용 계산 (Gemini 2.0 Flash: $0.075/1M input, $0.30/1M output)

2. **응답 속도**
   - API 호출 시간
   - 데이터 준비 시간
   - 함수 실행 시간
   - 총 응답 시간

3. **함수 호출 패턴**
   - 평균 함수 호출 횟수
   - 가장 많이 호출되는 함수
   - 불필요한 중복 호출

4. **캐싱 기회**
   - 반복되는 쿼리
   - 정적 데이터 (센터 목록 등)
   - 계산 결과 재사용 가능성

5. **병목 지점**
   - 느린 함수
   - 큰 DataFrame 연산
   - 불필요한 데이터 복사

**출력 형식:**
```
## 성능 분석 리포트

### 📊 현재 성능
| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| 평균 응답 시간 | X.Xs | 3s | ✅/⚠️ |
| 평균 토큰 사용 | XXX | 500 | ✅/⚠️ |
| 함수 호출/쿼리 | X.X | 2.0 | ✅/⚠️ |
| 일일 비용 (100쿼리) | $X.XX | $0.05 | ✅/⚠️ |

### 🐌 병목 지점
1. {function_name}: XXms (전체의 XX%)
   - 원인: ...
   - 해결: ...

### 💡 최적화 제안
1. [High Impact] ...
2. [Medium Impact] ...
3. [Low Impact] ...

### 💰 예상 개선 효과
- 응답 시간: X.Xs → Y.Ys (-Z%)
- 토큰 사용: XXX → YYY (-Z%)
- 비용 절감: $X.XX → $Y.YY (-Z%)
```
"""


INTEGRATION_TESTER_PROMPT = """
당신은 AI 챗봇 통합 테스트 전문가입니다.

**테스트 시나리오:**

1. **End-to-End Flow**
   - 사용자 질문 입력
   - Entity 추출
   - 함수 호출
   - 응답 생성
   - UI 렌더링

2. **Multi-turn Conversation** (Phase 3)
   - 맥락 유지
   - Follow-up 질문 처리
   - 히스토리 관리

3. **Error Recovery**
   - API 실패 시 재시도
   - 함수 실행 실패 시 fallback
   - 타임아웃 처리

4. **Data Consistency**
   - snapshot_df + moves_df + timeline_df 일관성
   - 날짜 범위 정합성
   - SKU/센터 코드 일치

5. **UI Integration**
   - 차트 생성
   - Follow-up 버튼 클릭
   - Proactive insights 표시

**출력 형식:**
```
## 통합 테스트 결과

### ✅ Passed Tests (XX/YY)
1. ✅ E2E: 총 재고 조회
2. ✅ E2E: 품절 임박 SKU 검색
...

### ❌ Failed Tests (XX/YY)
1. ❌ Multi-turn: 맥락 상실
   - Expected: "AMZUS는 8,500개"
   - Actual: "무엇의 AMZUS?"
   - Root cause: chat_history not maintained

### ⚠️ Flaky Tests (XX/YY)
1. ⚠️ API Timeout (간헐적)
   - Frequency: 3/10 runs
   - Suggestion: Increase timeout to 10s

### 🔧 수정 필요 항목
- [ ] {issue_1}
- [ ] {issue_2}
```
"""


DOCUMENTATION_WRITER_PROMPT = """
당신은 기술 문서 작성 전문가입니다.

**문서화 원칙:**

1. **함수 독스트링**
   - Google 스타일 독스트링
   - Args, Returns, Raises 명시
   - 예시 코드 포함

2. **사용자 가이드**
   - 언제 이 함수를 사용하는가?
   - 입력 예시
   - 출력 예시
   - 주의사항

3. **개발자 노트**
   - 구현 세부사항
   - 성능 고려사항
   - 알려진 제약사항

4. **버전 히스토리**
   - 추가/변경/수정 내역
   - Breaking changes

**출력 형식:**
```python
def {function_name}(...):
    \"\"\"
    {한 줄 요약}

    {상세 설명 2-3줄}

    Args:
        param1 (type): 설명
        param2 (type, optional): 설명. Defaults to X.

    Returns:
        dict: {
            "field1": (type) 설명,
            "field2": (type) 설명
        }

    Raises:
        ValueError: 조건
        KeyError: 조건

    Examples:
        >>> result = {function_name}(...)
        >>> print(result["field1"])
        123

    Notes:
        - 성능: O(n) where n = DataFrame 크기
        - 제약: snapshot_df must have 'stock_qty' column

    Version History:
        - v2.0: Function calling으로 전환
        - v1.0: 초기 구현
    \"\"\"
```

그리고 마크다운 문서도 함께 생성:

```markdown
## `{function_name}`

### 📝 설명
...

### 🎯 사용 시점
- ...
- ...

### 📊 입력/출력

**Input:**
\`\`\`python
{function_name}(
    sku="BA00021",
    days=7
)
\`\`\`

**Output:**
\`\`\`json
{
    "sku": "BA00021",
    "total_sales": 126
}
\`\`\`

### ⚠️ 주의사항
- ...

### 🔗 관련 함수
- `related_function_1()`
- `related_function_2()`
```
"""


# =============================================================================
# Helper Functions (서브 에이전트를 쉽게 호출하기 위한 래퍼)
# =============================================================================

def review_function(
    function_name: str,
    file_path: str = "ai_chatbot_simple.py",
    focus_areas: Optional[List[str]] = None
) -> str:
    """
    함수 코드 리뷰를 위한 서브 에이전트 호출

    Args:
        function_name: 리뷰할 함수 이름 (예: "calculate_stockout_days")
        file_path: 함수가 있는 파일 경로
        focus_areas: 집중 검토 영역 (예: ["error_handling", "performance"])

    Returns:
        Task agent 호출을 위한 프롬프트 문자열

    Example:
        # Claude Code에서 직접 사용:
        # Task tool로 다음 프롬프트 전달
        prompt = review_function("calculate_stockout_days")
    """
    focus = f"\n\n**특히 다음 영역에 집중:**\n" + "\n".join(f"- {area}" for area in focus_areas) if focus_areas else ""

    return f"""
{FUNCTION_REVIEWER_PROMPT}

---

**검토 대상:**
- 파일: {file_path}
- 함수: `{function_name}()`
{focus}

**작업 순서:**
1. {file_path} 파일을 읽고 {function_name} 함수 코드를 찾으세요
2. 위의 5가지 검토 항목을 모두 확인하세요
3. 구체적인 코드 예시와 함께 개선안을 제시하세요
4. 1-10점 척도로 평가하세요

**중요:** 실제 코드를 읽고 구체적으로 분석해야 합니다. 일반론만 나열하지 마세요.
"""


def generate_tests(
    function_name: str,
    file_path: str = "ai_chatbot_simple.py",
    test_types: Optional[List[str]] = None
) -> str:
    """
    테스트 케이스 자동 생성

    Args:
        function_name: 테스트할 함수 이름
        file_path: 함수가 있는 파일 경로
        test_types: 테스트 타입 (예: ["happy_path", "edge_cases", "errors"])

    Returns:
        Task agent 호출을 위한 프롬프트 문자열
    """
    test_focus = test_types or ["happy_path", "edge_cases", "error_cases", "business_logic"]

    return f"""
{TEST_GENERATOR_PROMPT}

---

**테스트 생성 대상:**
- 파일: {file_path}
- 함수: `{function_name}()`
- 테스트 타입: {", ".join(test_focus)}

**작업 순서:**
1. {file_path}에서 {function_name} 함수의 시그니처와 로직을 파악하세요
2. 함수의 모든 파라미터 조합을 고려한 테스트 케이스를 설계하세요
3. pytest 형식으로 실행 가능한 테스트 코드를 생성하세요
4. 테스트 파일을 tests/test_{function_name}.py로 저장하세요

**중요:**
- 실제 실행 가능한 코드를 작성하세요
- fixture를 활용하여 테스트 데이터를 재사용하세요
- assert 문에 명확한 실패 메시지를 포함하세요
"""


def optimize_prompt(
    prompt_name: str,
    current_prompt: Optional[str] = None,
    goals: Optional[List[str]] = None
) -> str:
    """
    프롬프트 최적화

    Args:
        prompt_name: 프롬프트 이름 (예: "시스템 프롬프트", "함수 설명")
        current_prompt: 현재 프롬프트 텍스트 (None이면 파일에서 읽음)
        goals: 최적화 목표 (예: ["reduce_tokens", "improve_clarity"])

    Returns:
        Task agent 호출을 위한 프롬프트 문자열
    """
    optimization_goals = goals or ["clarity", "conciseness", "structure", "token_efficiency"]

    prompt_source = f"다음 프롬프트를 최적화하세요:\n\n```\n{current_prompt}\n```" if current_prompt else \
                   f"ai_chatbot_simple.py에서 '{prompt_name}' 프롬프트를 찾아 최적화하세요"

    return f"""
{PROMPT_OPTIMIZER_PROMPT}

---

**최적화 대상:**
{prompt_source}

**최적화 목표:**
{", ".join(f"- {goal}" for goal in optimization_goals)}

**작업 순서:**
1. 현재 프롬프트를 분석하여 토큰 수와 문제점을 파악하세요
2. 위의 5가지 최적화 원칙을 적용하세요
3. Before/After 비교를 명확히 보여주세요
4. 토큰 절감량과 예상 정확도 개선을 정량화하세요

**제약사항:**
- 핵심 기능은 유지해야 합니다
- 토큰을 줄이되 명확성을 해치지 마세요
- Few-shot 예시는 2-3개만 유지하세요
"""


def analyze_performance(
    feature_name: str = "전체 챗봇",
    focus_metrics: Optional[List[str]] = None,
    benchmark_data: Optional[Dict[str, Any]] = None
) -> str:
    """
    성능 분석 (토큰, 속도, 비용)

    Args:
        feature_name: 분석 대상 (예: "품절 조회 기능", "전체 챗봇")
        focus_metrics: 집중 분석 지표 (예: ["tokens", "latency", "cost"])
        benchmark_data: 벤치마크 데이터 (예: {"avg_tokens": 500, "avg_time": 2.5})

    Returns:
        Task agent 호출을 위한 프롬프트 문자열
    """
    metrics = focus_metrics or ["tokens", "latency", "function_calls", "cost", "bottlenecks"]

    benchmark_context = ""
    if benchmark_data:
        benchmark_context = f"\n\n**현재 성능 데이터:**\n```json\n{json.dumps(benchmark_data, indent=2)}\n```"

    return f"""
{PERFORMANCE_ANALYZER_PROMPT}

---

**분석 대상:**
- 기능: {feature_name}
- 분석 지표: {", ".join(metrics)}
{benchmark_context}

**작업 순서:**
1. ai_chatbot_simple.py의 코드를 분석하여 성능 병목을 찾으세요
2. 각 함수의 실행 시간을 추정하세요 (복잡도 기준)
3. 토큰 사용량을 계산하세요 (metadata + 프롬프트)
4. 최적화 제안을 영향도 순으로 정렬하세요

**중요:**
- 구체적인 숫자와 함께 분석하세요
- 개선 전/후를 비교하세요
- Quick wins (쉬운데 효과 큰 것)를 먼저 제안하세요
"""


def run_integration_tests(
    test_scenarios: Optional[List[str]] = None,
    include_ui: bool = True
) -> str:
    """
    통합 테스트 실행

    Args:
        test_scenarios: 테스트 시나리오 (예: ["e2e", "multi_turn", "error_recovery"])
        include_ui: UI 테스트 포함 여부

    Returns:
        Task agent 호출을 위한 프롬프트 문자열
    """
    scenarios = test_scenarios or ["e2e_flow", "multi_turn", "error_recovery", "data_consistency"]
    if include_ui:
        scenarios.append("ui_integration")

    return f"""
{INTEGRATION_TESTER_PROMPT}

---

**테스트 시나리오:**
{chr(10).join(f"{i+1}. {scenario}" for i, scenario in enumerate(scenarios))}

**작업 순서:**
1. ai_chatbot_simple.py와 v9_app.py를 분석하여 통합 지점을 파악하세요
2. 각 시나리오별로 테스트 케이스를 설계하세요
3. 가능하면 실제로 테스트를 실행하세요 (pytest 사용)
4. 실패한 테스트에 대해 원인과 수정 방법을 제시하세요

**중요:**
- End-to-end 흐름을 실제로 추적하세요
- 데이터 흐름의 일관성을 확인하세요
- Edge case에서의 동작을 검증하세요

**출력:**
- 테스트 결과 요약 (Passed/Failed/Flaky)
- 실패 원인 분석
- 수정 우선순위
"""


def generate_documentation(
    function_name: str,
    file_path: str = "ai_chatbot_simple.py",
    doc_types: Optional[List[str]] = None
) -> str:
    """
    함수 문서 자동 생성

    Args:
        function_name: 문서화할 함수 이름
        file_path: 함수가 있는 파일 경로
        doc_types: 문서 타입 (예: ["docstring", "user_guide", "api_reference"])

    Returns:
        Task agent 호출을 위한 프롬프트 문자열
    """
    docs = doc_types or ["docstring", "user_guide", "api_reference"]

    return f"""
{DOCUMENTATION_WRITER_PROMPT}

---

**문서화 대상:**
- 파일: {file_path}
- 함수: `{function_name}()`
- 문서 타입: {", ".join(docs)}

**작업 순서:**
1. {file_path}에서 {function_name} 함수를 읽고 이해하세요
2. Google 스타일 독스트링을 작성하세요
3. 사용자 가이드 마크다운을 작성하세요 (docs/functions/{function_name}.md)
4. API 레퍼런스에 추가할 내용을 생성하세요

**중요:**
- 구체적인 예시 코드를 포함하세요
- 입력/출력 형식을 JSON으로 명확히 보여주세요
- 알려진 제약사항과 주의사항을 명시하세요
- 관련 함수와의 연관성을 설명하세요
"""


# =============================================================================
# Agent Orchestrator (여러 에이전트를 순차적으로 실행)
# =============================================================================

def full_review_pipeline(function_name: str, file_path: str = "ai_chatbot_simple.py") -> str:
    """
    새 함수에 대한 전체 리뷰 파이프라인 실행

    실행 순서:
    1. Function Review (코드 품질)
    2. Test Generation (테스트 생성)
    3. Documentation (문서화)
    4. Integration Test (통합 테스트)

    Args:
        function_name: 리뷰할 함수 이름
        file_path: 파일 경로

    Returns:
        전체 파이프라인 프롬프트
    """
    return f"""
# 🔍 AI Chatbot Function Full Review Pipeline

**대상 함수:** `{function_name}()` in {file_path}

다음 4단계를 **순차적으로** 실행하세요:

---

## Step 1: Code Review

{review_function(function_name, file_path)}

---

## Step 2: Test Generation

위의 리뷰 결과를 반영하여 테스트를 생성하세요.

{generate_tests(function_name, file_path)}

---

## Step 3: Documentation

{generate_documentation(function_name, file_path)}

---

## Step 4: Integration Test

생성된 함수가 전체 시스템과 올바르게 통합되는지 확인하세요.

{run_integration_tests(test_scenarios=[f"{function_name}_integration"])}

---

## Final Report

4단계를 모두 완료한 후 다음 형식으로 요약하세요:

```markdown
# {function_name} 리뷰 완료

## 🎯 종합 평가
- 코드 품질: X/10
- 테스트 커버리지: XX%
- 문서화: ✅/⚠️
- 통합: ✅/⚠️

## ✅ 승인 조건 충족 여부
- [ ] P0 이슈 없음
- [ ] 테스트 통과율 95% 이상
- [ ] 문서화 완료
- [ ] 통합 테스트 통과

## 🚀 배포 가능 여부
- **YES** / **NO** (조건부) / **BLOCKED**
- 사유: ...

## 📋 남은 작업
- [ ] ...
```
"""


# =============================================================================
# Quick Access Functions (자주 쓰는 패턴)
# =============================================================================

def quick_review(function_name: str) -> str:
    """빠른 리뷰 (코드 품질만)"""
    return review_function(function_name)


def quick_test(function_name: str) -> str:
    """빠른 테스트 생성"""
    return generate_tests(function_name)


def quick_doc(function_name: str) -> str:
    """빠른 문서 생성"""
    return generate_documentation(function_name)


def quick_perf() -> str:
    """빠른 성능 분석"""
    return analyze_performance()


# =============================================================================
# Usage Examples
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("AI Chatbot Sub-Agents Helper")
    print("=" * 80)
    print()
    print("사용 예시:")
    print()
    print("# 1. 함수 리뷰")
    print('review_function("calculate_stockout_days")')
    print()
    print("# 2. 테스트 생성")
    print('generate_tests("get_sku_trend")')
    print()
    print("# 3. 프롬프트 최적화")
    print('optimize_prompt("시스템 프롬프트")')
    print()
    print("# 4. 성능 분석")
    print('analyze_performance("품절 조회 기능")')
    print()
    print("# 5. 통합 테스트")
    print('run_integration_tests(["e2e", "error_recovery"])')
    print()
    print("# 6. 문서 생성")
    print('generate_documentation("compare_skus")')
    print()
    print("# 7. 전체 파이프라인")
    print('full_review_pipeline("simulate_demand_change")')
    print()
    print("=" * 80)
