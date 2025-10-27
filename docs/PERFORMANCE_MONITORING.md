# 성능 모니터링 가이드

SCM Dashboard v9에는 함수 실행 시간 측정 및 성능 메트릭 수집 기능이 내장되어 있습니다.

## 📊 기본 사용법

### 1. 함수 데코레이터 사용

가장 간단한 방법은 `@measure_time` 데코레이터를 사용하는 것입니다:

```python
from scm_dashboard_v9.common.performance import measure_time

@measure_time
def my_slow_function():
    # 시간이 오래 걸리는 작업
    result = process_large_data()
    return result
```

**로그 출력 예시:**
```
INFO - ✓ my_slow_function completed in 0.52s
WARNING - ⏱️  my_slow_function took 2.34s (threshold: 1s)
ERROR - ⚠️  SLOW: my_slow_function took 12.45s (threshold: 10s)
```

### 2. 컨텍스트 매니저 사용

코드 블록의 실행 시간을 측정하려면 컨텍스트 매니저를 사용합니다:

```python
from scm_dashboard_v9.common.performance import measure_time_context

def load_and_process_data():
    with measure_time_context("Data loading"):
        data = load_from_database()

    with measure_time_context("Data transformation"):
        transformed = transform_data(data)

    return transformed
```

### 3. 메트릭 수집 및 통계

여러 번 실행되는 함수의 통계를 수집하려면:

```python
from scm_dashboard_v9.common.performance import PerformanceMetrics

metrics = PerformanceMetrics()

for i in range(100):
    with metrics.track("api_call"):
        result = call_external_api()

# 통계 확인
stats = metrics.get_stats("api_call")
print(f"평균: {stats['avg']:.2f}s")
print(f"최소: {stats['min']:.2f}s")
print(f"최대: {stats['max']:.2f}s")
print(f"호출 횟수: {stats['count']}")
```

## 🎯 현재 모니터링 중인 함수

다음 핵심 함수들이 자동으로 성능 모니터링됩니다:

### 타임라인 빌드
- `scm_dashboard_v9.core.timeline.build_timeline()`
  - 타임라인 생성 전체 시간 측정
  - 임계값: 1초 (WARNING), 10초 (ERROR)

### 데이터 로딩
- `scm_dashboard_v9.data_sources.gsheet.load_from_gsheet()`
  - Google Sheets API 호출: `"Google Sheets API fetch"`
  - 데이터 정규화: `"Data normalization"`

### 분석 및 예측
- 소비량 추정 (consumption estimation)
- KPI 계산 (KPI calculation)

## ⚙️ 성능 임계값

시스템은 다음 임계값을 사용하여 성능 문제를 감지합니다:

| 실행 시간 | 로그 레벨 | 의미 |
|-----------|-----------|------|
| < 1초 | INFO | 정상 |
| 1초 ~ 10초 | WARNING | 느림 주의 |
| > 10초 | ERROR | 매우 느림 - 최적화 필요 |

## 📈 성능 로그 분석

### 로그 필터링

특정 함수의 성능만 확인:
```bash
# 타임라인 빌드 성능만 보기
grep "build_timeline" app.log | grep "completed in"

# 느린 작업만 보기 (1초 이상)
grep "took.*s" app.log | grep -E "(WARNING|ERROR)"

# 매우 느린 작업만 보기 (10초 이상)
grep "SLOW:" app.log
```

### 로그 레벨 조정

성능 로그의 상세도를 조정하려면 `.env` 파일에서:

```bash
# 모든 성능 로그 보기
LOG_LEVEL=DEBUG

# 경고 이상만 보기
LOG_LEVEL=WARNING
```

## 🔍 성능 문제 디버깅

### 1단계: 느린 함수 식별

```bash
# 최근 로그에서 느린 작업 찾기
tail -100 app.log | grep "took" | sort -t':' -k3 -n
```

### 2단계: 병목 지점 분석

```python
from scm_dashboard_v9.common.performance import global_metrics

# 글로벌 메트릭 사용
def my_function():
    with global_metrics.track("database_query"):
        result = query_database()

    with global_metrics.track("data_processing"):
        processed = process(result)

    return processed

# 나중에 통계 확인
all_stats = global_metrics.get_all_stats()
for operation, stats in all_stats.items():
    print(f"{operation}: avg={stats['avg']:.2f}s, max={stats['max']:.2f}s")
```

### 3단계: 최적화 적용

병목 지점을 찾았다면:
- 데이터베이스 쿼리 최적화
- 캐싱 적용
- 배치 처리로 전환
- 불필요한 연산 제거

## 💡 베스트 프랙티스

### DO ✅
- **핵심 비즈니스 로직**에 성능 모니터링 적용
- **외부 API 호출**은 항상 측정
- **대용량 데이터 처리**는 반드시 모니터링
- 정기적으로 로그 분석하여 성능 추세 파악

### DON'T ❌
- 모든 함수에 무분별하게 적용 (오버헤드 발생)
- 매우 빠른 함수 (<0.01초)에는 불필요
- 단순 getter/setter에는 사용하지 않음

## 📊 프로덕션 모니터링

프로덕션 환경에서는:

1. **로그 수집**: CloudWatch, ELK Stack 등 사용
2. **알림 설정**: 10초 이상 걸리는 작업에 알림
3. **대시보드**: Grafana 등으로 실시간 모니터링
4. **정기 리포트**: 주간 성능 리포트 자동 생성

## 🚀 성능 최적화 체크리스트

- [ ] 모든 느린 작업(>1초) 로깅 확인
- [ ] 매우 느린 작업(>10초) 원인 파악
- [ ] 캐싱 적용 검토
- [ ] 데이터베이스 쿼리 최적화
- [ ] 불필요한 데이터 로딩 제거
- [ ] 병렬 처리 가능 여부 검토

## 📚 추가 리소스

- [Python logging 공식 문서](https://docs.python.org/3/library/logging.html)
- [성능 프로파일링 가이드](https://docs.python.org/3/library/profile.html)
- [Streamlit 성능 최적화](https://docs.streamlit.io/library/advanced-features/caching)
