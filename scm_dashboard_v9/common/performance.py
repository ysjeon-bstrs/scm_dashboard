"""
성능 모니터링 유틸리티

함수 실행 시간 측정 및 성능 메트릭 수집을 위한 데코레이터와 유틸리티를 제공합니다.
"""

from __future__ import annotations

import functools
import logging
import time
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

# Type variable for generic decorator
F = TypeVar("F", bound=Callable[..., Any])


def measure_time(func: F) -> F:
    """
    함수 실행 시간을 측정하고 로깅하는 데코레이터.

    실행 시간이 1초 이상이면 WARNING, 10초 이상이면 ERROR 레벨로 로깅합니다.

    Args:
        func: 실행 시간을 측정할 함수

    Returns:
        래핑된 함수

    Examples:
        >>> @measure_time
        ... def slow_function():
        ...     time.sleep(2)
        ...     return "done"
        >>> result = slow_function()
        WARNING - slow_function took 2.00s
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.perf_counter()

        try:
            result = func(*args, **kwargs)
            return result
        finally:
            elapsed = time.perf_counter() - start_time

            # 실행 시간에 따라 로그 레벨 조정
            if elapsed >= 10.0:
                logger.error(
                    f"⚠️  SLOW: {func.__name__} took {elapsed:.2f}s " f"(threshold: 10s)"
                )
            elif elapsed >= 1.0:
                logger.warning(
                    f"⏱️  {func.__name__} took {elapsed:.2f}s " f"(threshold: 1s)"
                )
            else:
                logger.info(f"✓ {func.__name__} completed in {elapsed:.2f}s")

    return wrapper  # type: ignore[return-value]


def measure_time_context(operation_name: str) -> PerformanceContext:
    """
    컨텍스트 매니저를 사용한 코드 블록 성능 측정.

    Args:
        operation_name: 측정할 작업의 이름

    Returns:
        PerformanceContext 인스턴스

    Examples:
        >>> with measure_time_context("data loading"):
        ...     df = pd.read_csv("large_file.csv")
        INFO - data loading completed in 2.34s
    """
    return PerformanceContext(operation_name)


class PerformanceContext:
    """
    코드 블록의 실행 시간을 측정하는 컨텍스트 매니저.

    Attributes:
        operation_name: 측정할 작업의 이름
        start_time: 시작 시간 (초)
        elapsed: 경과 시간 (초)
    """

    def __init__(self, operation_name: str) -> None:
        self.operation_name = operation_name
        self.start_time: float = 0.0
        self.elapsed: float = 0.0

    def __enter__(self) -> PerformanceContext:
        self.start_time = time.perf_counter()
        logger.debug(f"Starting: {self.operation_name}")
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.elapsed = time.perf_counter() - self.start_time

        if exc_type is not None:
            logger.error(f"❌ {self.operation_name} failed after {self.elapsed:.2f}s")
        elif self.elapsed >= 10.0:
            logger.error(
                f"⚠️  SLOW: {self.operation_name} took {self.elapsed:.2f}s "
                f"(threshold: 10s)"
            )
        elif self.elapsed >= 1.0:
            logger.warning(
                f"⏱️  {self.operation_name} took {self.elapsed:.2f}s " f"(threshold: 1s)"
            )
        else:
            logger.info(f"✓ {self.operation_name} completed in {self.elapsed:.2f}s")


class PerformanceMetrics:
    """
    성능 메트릭 수집 및 통계 계산.

    여러 번 실행되는 함수의 평균/최소/최대 실행 시간을 추적합니다.

    Examples:
        >>> metrics = PerformanceMetrics()
        >>> for i in range(10):
        ...     with metrics.track("my_operation"):
        ...         do_work()
        >>> print(metrics.get_stats("my_operation"))
        {'count': 10, 'avg': 0.5, 'min': 0.3, 'max': 0.8}
    """

    def __init__(self) -> None:
        self._metrics: dict[str, list[float]] = {}

    def track(self, operation_name: str) -> PerformanceContext:
        """
        작업 실행 시간을 추적하는 컨텍스트 매니저.

        Args:
            operation_name: 작업 이름

        Returns:
            PerformanceContext 인스턴스
        """
        ctx = PerformanceContext(operation_name)

        class TrackingContext:
            def __enter__(self_ctx) -> PerformanceContext:
                return ctx.__enter__()

            def __exit__(self_ctx, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
                ctx.__exit__(exc_type, exc_val, exc_tb)
                if exc_type is None:  # 성공한 경우만 기록
                    if operation_name not in self._metrics:
                        self._metrics[operation_name] = []
                    self._metrics[operation_name].append(ctx.elapsed)

        return TrackingContext()  # type: ignore[return-value]

    def get_stats(self, operation_name: str) -> dict[str, float]:
        """
        특정 작업의 통계를 반환합니다.

        Args:
            operation_name: 작업 이름

        Returns:
            count, avg, min, max를 포함한 딕셔너리
        """
        if operation_name not in self._metrics:
            return {"count": 0, "avg": 0.0, "min": 0.0, "max": 0.0}

        times = self._metrics[operation_name]
        return {
            "count": len(times),
            "avg": sum(times) / len(times),
            "min": min(times),
            "max": max(times),
        }

    def get_all_stats(self) -> dict[str, dict[str, float]]:
        """
        모든 작업의 통계를 반환합니다.

        Returns:
            작업명을 키로 하는 통계 딕셔너리
        """
        return {name: self.get_stats(name) for name in self._metrics.keys()}

    def reset(self) -> None:
        """모든 메트릭을 초기화합니다."""
        self._metrics.clear()
        logger.debug("Performance metrics reset")


# 글로벌 메트릭 인스턴스 (선택적 사용)
global_metrics = PerformanceMetrics()
