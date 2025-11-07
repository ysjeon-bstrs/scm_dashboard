import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """테스트 환경 설정: 필수 환경변수가 없으면 테스트용 값 설정"""
    if not os.getenv("GSHEET_ID"):
        os.environ["GSHEET_ID"] = "test-sheet-id-for-pytest"
