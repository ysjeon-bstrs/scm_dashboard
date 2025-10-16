# V6 마이그레이션 체크리스트 (v5 → v6)

- 브랜치 전략
  - [ ] 기능 브랜치 `v6/refactor-modularization` 생성
  - [ ] `v5_main.py`는 버그 수정 외 변경 동결

- 앱 엔트리/UI
  - [ ] `scm_dashboard_v6/app/main.py` 신규 생성 (얇은 엔트리)
  - [ ] 사이드바/입력 컨트롤을 `ui/controls.py`로 이동
  - [ ] Streamlit 차트 옵션 `use_container_width` → `width='stretch'`로 치환

- 기능(Features) 분리
  - [ ] `features/timeline.py`: 타임라인 빌드 + 소비 반영 + 스텝 차트 렌더
  - [ ] `features/amazon.py`: 아마존 컨텍스트 생성 + 판매/재고 차트 렌더
  - [ ] `features/inventory_view.py`: 인벤토리 피벗 + CSV 다운로드 + LOT 상세

- 차트 모듈 분할
  - [ ] v5 `ui/charts.py` → `ui/charts/amazon.py`, `ui/charts/step.py`로 분리
  - [ ] 공통 유틸(색상/가드/보조)은 `ui/charts/utils.py`로 이동

- 데이터 + 도메인 정리
  - [ ] `data/loaders.py`(gsheet/excel), `data/cache.py`, `data/config.py` 정리
  - [ ] `domain/centers.py`(센터 정규화), `domain/time.py`(기간/날짜 유틸) 신설

- 파이프라인
  - [ ] `pipeline/orchestration.py`에서 단일 API `build_timeline_bundle` 제공

- 예측(Forecast)
  - [ ] `forecast/context.py`(AmazonForecastContext) 분리
  - [ ] `forecast/sales_from_inventory.py`(재고→판매 변환) 분리
  - [ ] `forecast/consumption.py`는 순수 계산 모듈로 유지

- 어댑터
  - [ ] `adapters/v4.py`에서 v4 호출(kpi, 재고자산, processing, CENTER_COL) 래핑

- 테스트
  - [ ] 템플릿 `tests/test_*_feature_template.py`를 실제 테스트로 교체
  - [ ] 품절 시점 클리핑, 인벤토리→판매 변환 등 단위 테스트 추가

- 사용 중단/경계 정리
  - [ ] 교차 모듈 프라이빗 호출 제거(예: UI에서 `_timeline_inventory_matrix` 직접 호출 금지)
  - [ ] 어댑터 외부에서 v4 직접 import 금지

- 문서 & CI
  - [ ] README에 v6 구조/실행 방법 갱신
  - [ ] CI에서 v6 패키지 테스트 포함하도록 갱신
