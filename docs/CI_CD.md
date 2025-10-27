# CI/CD 파이프라인 가이드

SCM Dashboard v9는 GitHub Actions를 사용한 자동화된 CI/CD 파이프라인을 제공합니다.

## 🔄 자동화된 워크플로우

### 1. 테스트 워크플로우 (test.yml)

**트리거 조건:**
- `main`, `master`, `develop` 브랜치에 push
- `claude/**` 브랜치에 push
- PR이 위 브랜치들로 생성될 때

**실행 내용:**
- Python 3.9, 3.10, 3.11에서 테스트 실행
- 모든 v9 테스트 실행
- 테스트 결과 리포트 생성
- 실패 시 상세 로그 제공

**사용 방법:**
```bash
# 로컬에서 동일한 테스트 실행
python -m pytest tests/test_v9*.py -v --tb=short
```

### 2. 코드 품질 워크플로우 (code-quality.yml)

**트리거 조건:**
- 테스트 워크플로우와 동일

**실행 내용:**

#### Lint 작업
- **flake8**: Python 구문 오류 및 스타일 체크
- **black**: 코드 포맷팅 체크
- **isort**: import 정렬 체크
- **mypy**: 타입 힌트 검증

#### Security 작업
- **bandit**: 보안 취약점 스캔
- 보안 리포트 자동 생성

## 📋 로컬 개발 환경 설정

### 1. 개발 의존성 설치

```bash
# 프로덕션 의존성
pip install -r requirements.txt

# 개발 의존성
pip install -r requirements-dev.txt
```

### 2. 로컬 코드 품질 체크

```bash
# 전체 체크 실행
./scripts/check-quality.sh

# 또는 개별 실행
flake8 scm_dashboard_v9
black --check scm_dashboard_v9
isort --check-only scm_dashboard_v9
mypy scm_dashboard_v9 --ignore-missing-imports
```

### 3. 자동 포맷팅

```bash
# black으로 자동 포맷팅
black scm_dashboard_v9

# isort로 import 정렬
isort scm_dashboard_v9
```

## 🚦 PR 체크리스트

Pull Request를 만들기 전에 확인:

- [ ] 모든 테스트 통과 (`pytest`)
- [ ] 코드 포맷팅 적용 (`black`)
- [ ] Import 정렬 완료 (`isort`)
- [ ] Flake8 오류 없음
- [ ] 타입 힌트 추가 (새 함수)
- [ ] 문서화 완료 (docstring)
- [ ] 성능 영향 검토

## 📊 CI 상태 확인

### GitHub Actions UI
1. 레포지토리의 "Actions" 탭 방문
2. 최근 워크플로우 실행 확인
3. 실패한 작업 클릭하여 로그 확인

### 커맨드 라인
```bash
# GitHub CLI 사용
gh run list --limit 10
gh run view <run-id>
```

## ⚠️ 일반적인 CI 실패 원인

### 1. 테스트 실패
```bash
# 로컬에서 재현
python -m pytest tests/test_v9_domain.py::test_name -v

# 디버그 모드
python -m pytest tests/test_v9_domain.py -v --pdb
```

### 2. Flake8 오류
```bash
# 오류 확인
flake8 scm_dashboard_v9 --show-source --statistics

# 자동 수정 가능한 것들
autopep8 --in-place --aggressive scm_dashboard_v9/*.py
```

### 3. Black 포맷 불일치
```bash
# 차이점 확인
black --check --diff scm_dashboard_v9

# 자동 수정
black scm_dashboard_v9
```

### 4. Import 정렬 문제
```bash
# 차이점 확인
isort --check-only --diff scm_dashboard_v9

# 자동 수정
isort scm_dashboard_v9
```

## 🔧 CI 설정 커스터마이징

### Python 버전 변경

`.github/workflows/test.yml` 수정:
```yaml
strategy:
  matrix:
    python-version: ["3.9", "3.10", "3.11", "3.12"]  # 3.12 추가
```

### 테스트 범위 변경

```yaml
- name: Run tests with pytest
  run: |
    # 특정 테스트만 실행
    python -m pytest tests/test_v9_domain.py tests/test_v9_timeline.py -v

    # 커버리지 포함
    python -m pytest tests/test_v9*.py --cov=scm_dashboard_v9 --cov-report=html
```

### 추가 체크 추가

`.github/workflows/code-quality.yml`에:
```yaml
- name: Check docstrings
  run: |
    pip install pydocstyle
    pydocstyle scm_dashboard_v9
```

## 📈 CI 성능 최적화

### 1. 캐싱 활용
현재 pip 캐시가 활성화되어 있어 의존성 설치 시간이 단축됩니다.

### 2. 병렬 실행
Matrix strategy로 여러 Python 버전을 동시 테스트합니다.

### 3. 선택적 실행
```yaml
# 특정 파일 변경 시만 실행
on:
  push:
    paths:
      - 'scm_dashboard_v9/**'
      - 'tests/**'
      - 'requirements.txt'
```

## 🔒 시크릿 관리

GitHub Secrets에 저장해야 할 것들:
- Google Sheets API 키
- 데이터베이스 접속 정보
- 외부 서비스 토큰

**설정 방법:**
1. GitHub 레포지토리 → Settings → Secrets
2. "New repository secret" 클릭
3. 이름과 값 입력
4. 워크플로우에서 사용:
   ```yaml
   env:
     GSHEET_ID: ${{ secrets.GSHEET_ID }}
   ```

## 🚀 배포 자동화 (향후)

프로덕션 배포를 자동화하려면:

```yaml
name: Deploy to Production

on:
  push:
    branches: [ main ]
    tags:
      - 'v*'

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Deploy to server
        run: |
          # 서버 배포 스크립트 실행
          ssh user@server 'cd /app && git pull && systemctl restart app'
```

## 📚 추가 리소스

- [GitHub Actions 공식 문서](https://docs.github.com/en/actions)
- [pytest 공식 문서](https://docs.pytest.org/)
- [black 공식 문서](https://black.readthedocs.io/)
- [flake8 공식 문서](https://flake8.pycqa.org/)
