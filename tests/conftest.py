import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pytest_configure(config):
    """pytest 초기화 시점에 실행되어 테스트 수집 전에 환경을 준비합니다.

    이 훅은 테스트 모듈이 import되기 전에 실행되므로,
    config.py가 로드될 때 GSHEET_ID가 이미 설정되어 있습니다.
    """
    # 테스트 환경 설정: 필수 환경변수가 없으면 테스트용 값 설정
    if not os.getenv("GSHEET_ID"):
        os.environ["GSHEET_ID"] = "test-sheet-id-for-pytest"
