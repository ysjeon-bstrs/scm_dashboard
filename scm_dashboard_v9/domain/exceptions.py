"""
도메인 계층 예외 정의

이 모듈은 SCM 대시보드 도메인 계층에서 발생할 수 있는
모든 예외를 정의합니다. UI 계층은 이 예외들을 잡아서
사용자 친화적인 에러 메시지로 변환합니다.
"""

from __future__ import annotations


class DomainError(Exception):
    """
    도메인 계층의 기본 예외 클래스.

    모든 도메인 예외는 이 클래스를 상속합니다.
    """

    pass


class ValidationError(DomainError):
    """
    데이터 검증 실패 시 발생하는 예외.

    입력 데이터가 비즈니스 규칙을 만족하지 않을 때 발생합니다.
    예: 필수 컬럼 누락, 날짜 범위 오류 등
    """

    pass


class DataLoadError(DomainError):
    """
    데이터 로드 실패 시 발생하는 예외.

    Excel, Google Sheets 등에서 데이터를 불러올 때
    오류가 발생한 경우 사용합니다.
    """

    pass


class FilterError(DomainError):
    """
    필터링 작업 실패 시 발생하는 예외.

    센터/SKU 필터링, 날짜 범위 계산 등에서
    오류가 발생한 경우 사용합니다.
    """

    pass


class TimelineError(DomainError):
    """
    타임라인 빌드 실패 시 발생하는 예외.

    타임라인 생성 과정에서 데이터 부족이나
    설정 오류가 발생한 경우 사용합니다.
    """

    pass
