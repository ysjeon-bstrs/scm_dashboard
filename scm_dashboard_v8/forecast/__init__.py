"""예측 관련 래퍼.

v8 계층에서는 예측 로직을 직접 수정하지 않고 v5에서 검증된 함수를 그대로 노출한다.
"""

from scm_dashboard_v5.forecast import (
    apply_consumption_with_events as apply_consumption_with_events,
    build_amazon_forecast_context as build_amazon_forecast_context,
)

__all__ = ["apply_consumption_with_events", "build_amazon_forecast_context"]
