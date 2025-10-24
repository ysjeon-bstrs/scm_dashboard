# 🐛 Bug Fixes - v9 Module

## 수정된 버그 (2개)

---

## Bug #1: Zero Chunk Size Overridden by Default

### 📋 설명

**위치**: `scm_dashboard_v9/ui/kpi/cards.py:272`

**문제**: 
`render_sku_summary_cards()` 함수에서 `chunk_size=0`을 명시적으로 전달해도 CONFIG 기본값으로 덮어씌워지는 버그.

```python
# 🐛 Before (Buggy)
chunk_size = chunk_size or CONFIG.ui.kpi_card_chunk_size
# chunk_size=0 → Falsy → CONFIG 값으로 대체됨
```

**원인**:
- Python의 `or` 연산자는 `0`을 falsy 값으로 취급
- `chunk_size=0`이 명시적으로 전달되어도 `False`로 평가되어 우측 값으로 대체됨

**영향**:
- 호출자가 의도적으로 `chunk_size=0`을 설정할 수 없음
- 특정 UI 레이아웃에서 문제 발생 가능

### ✅ 수정

```python
# ✅ After (Fixed)
if chunk_size is None:
    chunk_size = CONFIG.ui.kpi_card_chunk_size
# None만 기본값으로 대체, 0은 유지됨
```

**동작**:
- `chunk_size=None` → CONFIG 기본값 사용 ✅
- `chunk_size=0` → 0 유지 ✅
- `chunk_size=3` → 3 유지 ✅

### 🧪 테스트

**파일**: `tests/test_v9_kpi_cards.py`

```python
def test_chunk_size_zero_preserved():
    """chunk_size=0을 명시적으로 전달하면 유지되어야 함"""
    result = render_sku_summary_cards(
        ...,
        chunk_size=0,  # ← 명시적으로 0 전달
    )
    # 함수가 정상 실행되어야 함 (ZeroDivisionError 없음)

def test_chunk_size_none_uses_config():
    """chunk_size=None일 때는 CONFIG 기본값 사용"""
    result = render_sku_summary_cards(...)  # chunk_size 없음
    # CONFIG 기본값 사용

def test_chunk_size_positive_preserved():
    """양수는 그대로 유지되어야 함"""
    result = render_sku_summary_cards(..., chunk_size=3)
    # 3 유지
```

---

## Bug #2: Stock Quantities Not Always Float

### 📋 설명

**위치**: `scm_dashboard_v9/domain/normalization.py:355`

**문제**:
`normalize_snapshot()` 함수가 `stock_qty`를 항상 `float`로 반환해야 하지만, 소스 데이터가 정수만 포함하고 NaN이 없을 때 `int64`로 유지되는 버그.

```python
# 🐛 Before (Buggy)
out["stock_qty"] = pd.to_numeric(out.get("stock_qty"), errors="coerce").fillna(0.0)
# 정수만 있으면 int64로 유지됨
```

**원인**:
- Pandas의 `to_numeric()`은 입력에 따라 타입 추론
- 정수만 있으면 `int64` 유지
- `.fillna(0.0)`은 NaN이 있을 때만 `float64`로 변환

**영향**:
- 테스트 실패 (`assert dtype == float64`)
- 하류 코드에서 float 연산 기대하지만 int 받음
- 나눗셈 등에서 정수 나눗셈 발생 가능 (Python 2 스타일)

**예시**:
```python
# 입력: [100, 200, 300] (모두 정수, NaN 없음)
result = pd.to_numeric([100, 200, 300]).fillna(0.0)
print(result.dtype)  # int64 ❌ (예상: float64)

# 입력: [100, 200, None] (NaN 있음)
result = pd.to_numeric([100, 200, None]).fillna(0.0)
print(result.dtype)  # float64 ✅
```

### ✅ 수정

```python
# ✅ After (Fixed)
out["stock_qty"] = (
    pd.to_numeric(out.get("stock_qty"), errors="coerce")
    .fillna(0.0)
    .astype(float)  # ← 명시적으로 float 변환
)
```

**적용 컬럼**:
- ✅ `stock_qty` (필수)
- ✅ `stock_available` (선택적, Amazon FBA)
- ✅ `stock_expected` (선택적, Amazon FBA)
- ✅ `stock_processing` (선택적, Amazon FBA)
- ✅ `pending_fc` (선택적, Amazon FBA)

**적용 안 함**:
- `sales_qty` - 이미 `.astype(int)` 명시적 변환됨 ✅

### 🧪 테스트

**파일**: `tests/test_v9_domain.py`

```python
def test_normalize_snapshot_stock_qty_always_float():
    """stock_qty는 항상 float (Bug Fix)"""
    raw_integers = pd.DataFrame({
        "stock_qty": [100, 200],  # 정수만
    })
    
    result = normalize_snapshot(raw_integers)
    
    # stock_qty가 float64여야 함
    assert result["stock_qty"].dtype == "float64"
    assert result.iloc[0]["stock_qty"] == 100.0  # float

def test_normalize_snapshot_fba_columns_always_float():
    """Amazon FBA 컬럼도 항상 float (Bug Fix)"""
    raw = pd.DataFrame({
        "stock_available": [80],   # 정수
        "stock_expected": [20],    # 정수
        "stock_processing": [10],  # 정수
        "pending_fc": [5],         # 정수
    })
    
    result = normalize_snapshot(raw)
    
    # 모든 재고 컬럼이 float여야 함
    assert result["stock_available"].dtype == "float64"
    assert result["stock_expected"].dtype == "float64"
    assert result["stock_processing"].dtype == "float64"
    assert result["pending_fc"].dtype == "float64"
```

---

## 📊 영향 분석

### Bug #1 영향

| 항목 | Before | After |
|------|--------|-------|
| `chunk_size=None` | CONFIG 기본값 | CONFIG 기본값 ✅ |
| `chunk_size=0` | CONFIG 기본값 ❌ | 0 유지 ✅ |
| `chunk_size=3` | 3 유지 | 3 유지 ✅ |

**영향받는 코드**:
- `scm_dashboard_v9/ui/kpi/cards.py:272` (1곳)

### Bug #2 영향

| 항목 | Before | After |
|------|--------|-------|
| 정수만 있는 stock_qty | `int64` ❌ | `float64` ✅ |
| NaN 포함 stock_qty | `float64` | `float64` ✅ |
| FBA 컬럼들 | `int64/float64` 혼재 ❌ | 항상 `float64` ✅ |

**영향받는 코드**:
- `scm_dashboard_v9/domain/normalization.py:355` (5개 컬럼)
- 모든 하류 소비자 (타임라인 빌더, 예측 모듈 등)

---

## ✅ 검증 체크리스트

### Bug #1: Chunk Size
- [x] `chunk_size=0` → 0 유지
- [x] `chunk_size=None` → CONFIG 사용
- [x] `chunk_size=3` → 3 유지
- [x] 테스트 추가 (`test_v9_kpi_cards.py`)

### Bug #2: Float Conversion
- [x] `stock_qty` 항상 float
- [x] `stock_available` 항상 float
- [x] `stock_expected` 항상 float
- [x] `stock_processing` 항상 float
- [x] `pending_fc` 항상 float
- [x] 테스트 추가 (`test_v9_domain.py`)

---

## 📝 변경 파일

```bash
# 수정된 파일
scm_dashboard_v9/ui/kpi/cards.py           |  4 ++--
scm_dashboard_v9/domain/normalization.py   | 10 +++++-----

# 신규 테스트
tests/test_v9_kpi_cards.py                 | 125 ++++++++++++
tests/test_v9_domain.py                    |  50 ++++++

4 files changed, 182 insertions(+), 7 deletions(-)
```

---

## 🎯 결론

두 버그 모두 **경계 조건(edge case)** 처리 문제:

1. **Falsy 값 처리**: `0`과 `None`을 구분해야 함
2. **타입 추론**: Pandas 타입 추론에 의존하지 말고 명시적 변환

**교훈**:
- ✅ 기본값 처리 시 `is None` 명시적 체크
- ✅ 타입 보장이 필요한 곳에 `.astype()` 명시
- ✅ 경계 조건 테스트 케이스 작성

**영향**: 낮음 (로컬 수정, API 변경 없음)
**위험**: 없음 (하위 호환성 유지)
**테스트**: 5개 신규 테스트 추가

✨ **모든 버그가 수정되고 테스트로 검증되었습니다!**
