"""
데이터 소스 추상화 계층

이 모듈은 다양한 데이터 소스(Excel, Google Sheets 등)로부터
데이터를 로드하는 기능을 제공합니다.
"""

from .excel import LoadedData, load_from_excel_uploader
from .gsheet import load_from_gsheet
from .loader import Loader, StaticFrameLoader
from .session import ensure_data

__all__ = [
    # 로더 프로토콜
    "Loader",
    "StaticFrameLoader",
    # 데이터 모델
    "LoadedData",
    # 로더 함수
    "load_from_excel_uploader",
    "load_from_gsheet",
    "ensure_data",
]
