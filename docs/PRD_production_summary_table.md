# PRD: 생산중 테이블 UI 통일 리디자인

## 1. 목적

현재 생산 진행 현황(Manufacturing WIP Status) 테이블은 컬럼 수가 많고 타임라인/상태가 한눈에 들어오지 않음.

인바운드 요약 테이블과 동일한 스타일로, **"10일 내 생산 완료 예정인 WIP만 추려서 보여주는 요약 테이블"** 을 추가하고자 함.

상세 확인이 필요할 때는 기존 WIP 원본 테이블을 그대로 아래에 유지한다.

## 2. 대상 위치

`v9_app.py` 마지막 부분의 15단계:

```python
# 15단계: 입고 예정 및 WIP 테이블
render_inbound_and_wip_tables(
    moves=data.moves,
    snapshot=snapshot_df,
    selected_centers=selected_centers,
    selected_skus=selected_skus,
    start=start_ts,
    end=end_ts,
    lag_days=lag_days,
    today=today_norm,
)
```

이 함수 내부에서 "생산중 WIP 요약 테이블"을 추가 구현한다
(필요시 `build_production_summary_table` / `render_production_summary_table` 같은 헬퍼로 분리).

## 3. 데이터 전제 (WIP 원본)

생산중 데이터는 현재 "생산 진행 현황(Manufacturing WIP Status)" 테이블에 사용 중인 DF를 기준으로 한다.

### 주요 컬럼

| 컬럼명 | 의미 |
|--------|------|
| `display_date` | 기준일 |
| `days_to_completion` | 생산 완료까지 남은 일수? (미사용 가능) |
| `resource_code` | SKU 코드 (예: BA00022) |
| `resource_name` | SKU 한글명 |
| `qty_ea` | 수량 (예: 12,000) |
| `pred_inbound_date` | 생산 완료 후 입고 예정일 (예상 완료일로 사용) |
| `lot` | Lot 번호 |
| `global_b2c` | 생산량 중 B2C 배정 수량 |
| `global_b2b` | 생산량 중 B2B 배정 수량 |

SKU 한글명 매핑은 이미 `build_resource_name_map` / `resource_name_map` 으로 제공되고 있으므로, 가능하면 이를 재사용.

## 4. 요약 테이블 요구사항

### 4-1. 필터링 (10일 내 생산완료건만)

`pred_inbound_date` 를 기준으로 "오늘 기준 10일 이내 완료 예정"만 표시.

**조건식 (pandas 기준):**

```python
today = today_norm  # v9_app에서 넘겨받은 normalized today
df["pred_inbound_date"] = pd.to_datetime(df["pred_inbound_date"], errors="coerce")
mask = (df["pred_inbound_date"].notna()) & \
       (df["pred_inbound_date"] >= today) & \
       (df["pred_inbound_date"] <= today + pd.Timedelta(days=10))
df_soon = df[mask]
```

이 요약 테이블에 포함되지 않는 나머지 생산중 데이터는 기존 "상세 WIP 테이블"에서 그대로 확인한다.

### 4-2. 노출 컬럼 (새 구성)

요약판에서는 아래 **5개 컬럼**만 사용:

1. **제품명(SKU)**
   - `"{resource_name} ({resource_code})"` 형태로 병기
   - 예: `비타민일루미네이팅세럼[30ml/-] (BA00022)`

2. **수량** (`qty_ea`)
   - 천단위 콤마, `12,000ea` 포맷

3. **예상완료일** (`pred_inbound_date`)
   - 날짜 형태 `YYYY-MM-DD`
   - 인바운드 ETA와 비슷한 색상 규칙 적용:
     - 오늘 ≤ 날짜 ≤ 오늘+5일 → 🟢 초록 (곧 완료)
     - 날짜 > 오늘+5일 → ⚪ 회색

4. **B2C** (`global_b2c`)
   - B2C 배정 수량

5. **B2B** (`global_b2b`)
   - B2B 배정 수량

**컬럼 예시:**

| 제품명(SKU) | 수량 | 예상 완료일 | B2C | B2B |
|------------|------|------------|-----|-----|

### 4-3. 정렬/정책

- **기본 정렬**: `pred_inbound_date` 오름차순 (완료 임박건이 위로)
- `qty_ea` / `B2C`/`B2B` 는 그대로 표시만, 추가 집계는 없음.
- `pred_inbound_date` 가 NaT인 건은 요약 테이블에서 제외(상세 테이블만).

## 5. UI 스타일 요구사항

상단 요약 인바운드 테이블과 동일한 스타일 재사용:
- `st.dataframe` + pandas `Styler`
- 행 패딩, 폰트 사이즈, 헤더 볼드, 보더 색 등 동일 규칙.

### 섹션 제목

**생산중 요약 섹션 제목 예:**
```python
st.subheader("🛠️ 생산 진행 현황 (요약)")
```

**기존 WIP 상세 섹션 제목 수정:**
```python
st.subheader("🛠️ 생산 진행 현황 (상세 · 전체 WIP)")
```

### 색상 팔레트

인바운드 ETA와 동일 규칙으로 설정 (함수 재사용 가능):

```python
def leadtime_color(target_date, today):
    if pd.isna(target_date): return "orange"
    delta = (target_date - today).days
    if delta < 0: return "red"
    if delta <= 5: return "green"
    return "gray"
```

**실제 색상 코드:**
- `red`: `#ef4444` (지연/과거)
- `green`: `#22c55e` (5일 이내)
- `gray`: `#9ca3af` (6일 이후)
- `orange`: `#f59e0b` (미확인)

## 6. 구현 단계 요약

### 6-1. 데이터 준비
- 기존 WIP DataFrame에서 생산중 행을 가져오는 부분 재사용.
- `pred_inbound_date`, `resource_name`, `resource_code`, `qty_ea`, `global_b2c`, `global_b2b` 활용.

### 6-2. 요약 DataFrame 생성 함수 추가
- 예: `build_production_summary_table(wip_df, today_norm) -> pd.DataFrame`
- 위 필터/컬럼/색상 로직 구현.

### 6-3. UI 렌더 함수 추가
- 예: `render_production_summary_table(df_summary)`
- 스타일은 인바운드 요약용 `render_inbound_table` 과 동일 컨벤션 사용.

### 6-4. `render_inbound_and_wip_tables` 안에서 호출
- 상단에 `render_production_summary_table` 호출 추가 (10일 이내 건수 존재할 때만).
- 그 아래 기존 WIP 상세 테이블 렌더 호출 유지.

### 6-5. 주석(선택)
테이블 하단에 간단한 설명 문구:
```python
st.caption("※ 10일 내 생산 완료 예정인 품목만 표시됩니다. 이후 내역은 아래 WIP 테이블을 참고하세요.")
```

## 7. 비기능 요구사항

- 10일 필터·색상 계산은 pandas 벡터 연산으로 처리하여 **1만행 기준 100ms 이내**.
- `pred_inbound_date` 파싱 오류/결측은 NaT로 처리하고, 요약 테이블에서는 제외.
- 기존 WIP 상세 테이블 기능에는 영향 없음 (컬럼/정렬 로직 그대로 유지).

## 8. 완료 기준 (Acceptance Criteria)

✅ 대시보드에서 **"생산 진행 현황(요약)"** 섹션에 10일 내 생산완료건만 표시되는 테이블이 보인다.

✅ 요약 테이블 컬럼이 구성된다:
   - `[제품명(SKU), 수량, 예상완료일, B2C, B2B]`

✅ **예상완료일** 컬럼이 인바운드 ETA와 비슷한 색상 규칙으로 표시된다(5일 이내 초록).

✅ 기존 **"생산 진행 현황(상세 · 전체 WIP)"** 테이블은 그대로 유지되며, 모든 생산중 데이터가 포함된다.

✅ `pred_inbound_date`가 NaT/결측인 행은 요약 테이블에 포함되지 않는다(상세에서만 보인다).

## 9. 파일 구조

```
scm_dashboard_v9/
├── ui/
│   ├── tables.py  # render_inbound_and_wip_tables, build_production_summary_table, render_production_summary_table 추가
│   └── inbound_table.py  # 참고용 (스타일 재사용)
├── docs/
│   └── PRD_production_summary_table.md  # 본 문서
└── v9_app.py  # 15단계에서 호출
```

## 10. 구현 체크리스트

- [ ] `build_production_summary_table()` 함수 구현
  - [ ] 10일 필터링 로직
  - [ ] 컬럼 구성 (제품명, 수량, 예상완료일, B2C, B2B)
  - [ ] 색상 계산 (`completion_color`)
  - [ ] 정렬 (pred_inbound_date 오름차순)

- [ ] `render_production_summary_table()` 함수 구현
  - [ ] Pandas Styler 적용
  - [ ] 인바운드 테이블과 동일 스타일
  - [ ] 예상완료일 색상 적용
  - [ ] 제품명 볼드

- [ ] `render_inbound_and_wip_tables()` 수정
  - [ ] 요약 테이블 섹션 추가
  - [ ] 기존 WIP 상세 섹션 제목 수정
  - [ ] 10일 이내 건수 체크 후 조건부 렌더링

- [ ] 테스트
  - [ ] 10일 필터 정확성 확인
  - [ ] 색상 표시 확인
  - [ ] 성능 체크 (1만행 100ms)
  - [ ] 결측치 처리 확인

---

**작성일**: 2025-01-13
**버전**: 1.0
**작성자**: Claude (AI Assistant)
