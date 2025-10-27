"""
도메인 예외 → UI 에러 메시지 어댑터

이 모듈은 도메인 계층에서 발생하는 예외를 잡아서
Streamlit 사용자 친화적인 에러 메시지로 변환합니다.

이를 통해 도메인 계층은 Streamlit에 의존하지 않으면서도
UI에서 적절한 에러 메시지를 표시할 수 있습니다.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

import streamlit as st

from scm_dashboard_v9.domain.exceptions import (
    DataLoadError,
    FilterError,
    TimelineError,
    ValidationError,
)


@contextmanager
def handle_domain_errors() -> Generator[None, None, None]:
    """
    도메인 예외를 잡아서 Streamlit 에러 메시지로 변환하는 컨텍스트 매니저.

    도메인 계층에서 발생하는 예외를 잡아서 사용자에게
    친화적인 에러 메시지를 표시합니다.

    Yields:
        None

    Examples:
        >>> from scm_dashboard_v9.domain import validate_timeline_inputs
        >>> with handle_domain_errors():
        ...     validate_timeline_inputs(snapshot, moves, start, end)

    Notes:
        - ValidationError: 데이터 검증 실패
        - DataLoadError: 데이터 로드 실패
        - FilterError: 필터링 작업 실패
        - TimelineError: 타임라인 빌드 실패
    """
    try:
        yield

    except ValidationError as e:
        # 검증 실패: 빨간색 에러 메시지
        st.error(f"❌ 데이터 검증 실패: {str(e)}")

    except DataLoadError as e:
        # 데이터 로드 실패: 빨간색 에러 메시지
        st.error(f"❌ 데이터 로드 실패: {str(e)}")

    except FilterError as e:
        # 필터링 실패: 노란색 경고 메시지
        st.warning(f"⚠️ 필터링 작업 실패: {str(e)}")

    except TimelineError as e:
        # 타임라인 빌드 실패: 빨간색 에러 메시지
        st.error(f"❌ 타임라인 생성 실패: {str(e)}")

    except Exception as e:
        # 예상치 못한 예외: 상세 정보와 함께 표시
        st.error(f"❌ 예상치 못한 오류가 발생했습니다: {type(e).__name__}: {str(e)}")
        st.exception(e)
