# PRD: 인바운드 예상 입고일 컬럼 확장

## 1. 배경 및 목적
- Google Sheets `leadtime_cal` 탭에는 출발창고 × 도착창고 × 운송수단 조합별 평균 리드타임이 정리되어 있음.
- 현재 Streamlit 인바운드 요약 테이블에는 ETA만 노출되고, 입고 예상일은 제공하지 않음.
- 운영 팀은 각 경로의 출발→입고 평균 리드타임을 반영한 예상 입고일을 파악하고자 함.

## 2. 목표
1. Google Sheets에서 `leadtime_cal` 탭을 로드하고 평균 출발→입고 리드타임을 애플리케이션에서 사용할 수 있도록 한다.
2. 인바운드 요약 테이블에 `예상 입고일` 컬럼을 추가하여, `출발일 + 평균 리드타임`으로 계산된 결과를 노출한다.
3. 테이블 하단에 "※ 예상 입고일은 해당 경로 출발→입고 평균 리드타임 기준입니다." 주석을 표기한다.

## 3. 주요 지표
- `leadtime_cal` 로드 성공 여부 및 캐시 적용(ttl=3600).
- 리드타임 맵 생성 수(조합 수)와 누락 조합 대비 NaT 처리율.
- 인바운드 요약 테이블에서 `예상 입고일` 컬럼 노출 여부 및 YYYY-MM-DD 포맷.

## 4. 요구 사항
- `LoadedData` 객체에 `leadtime_cal: pd.DataFrame` 필드를 추가하고 Google Sheets/Excel 로더 모두에서 값 설정.
- `scm_dashboard_v9/data_sources/gsheet.py`에 `load_leadtime_cal` 함수를 추가하여 컬럼명 영문화 및 숫자 변환.
- 리드타임 맵 빌더(`build_lt_inbound_map`)와 예상 입고일 계산 유틸리티를 도메인 계층에 구현.
- `v9_app.py` 인바운드 테이블 생성 흐름에서 평균 리드타임을 조회하여 새 컬럼을 계산하고 전달.
- `build_inbound_table` 및 `render_inbound_table`에 `expected_inbound_date` 처리를 추가하고 기존 기능과 호환 유지.
- 신규 유닛 테스트: 리드타임 맵/예상 입고일 계산 3~5건 검증, 인바운드 테이블에 예상 입고일 컬럼 포함 확인.

## 5. 데이터 흐름
1. `load_from_gsheet_api` → `leadtime_cal` 원본 DataFrame 로드.
2. `load_leadtime_cal` → 컬럼 리네임 및 평균 리드타임 수치 변환.
3. `build_lt_inbound_map` → `(from_center, to_center, carrier_mode)` 키 기반 딕셔너리 생성.
4. 인바운드 원본 데이터(`inbound_raw`)에 리드타임 매칭 → `expected_inbound_date` 계산.
5. `build_inbound_table` → 인보이스 단위로 최소 예상 입고일 집계 후 문자열 변환.
6. `render_inbound_table` → 새로운 컬럼 노출 및 주석 추가.

## 6. 비기능 요구 사항
- Streamlit 캐싱(`@st.cache_data`) 활용으로 API 호출 최소화.
- 리드타임 조합 미존재 또는 데이터 결측 시 `예상 입고일`은 공란 처리.
- 기존 ETA 색상 규칙 유지, `예상 입고일` 컬럼은 색상 미적용.

## 7. 범위 외 (Out of Scope)
- 중앙값 기반 리드타임 사용 전환.
- 리드타임 데이터 정제(이상치 처리, 보간 등) 개선.
- Streamlit UI 구조/레이아웃 전면 개편.

## 8. 완료 기준 (Acceptance Criteria)
- PR 머지 후 대시보드에서 `예상 입고일` 컬럼이 정상 노출되고, 테스트 케이스가 통과한다.
- 리드타임 데이터 미존재 시에도 앱이 오류 없이 동작하며 컬럼은 빈 값으로 표시된다.
- 테이블 하단에 요구된 주석 문구가 출력된다.
