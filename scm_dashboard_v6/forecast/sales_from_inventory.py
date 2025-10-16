"""
인벤토리 궤적에서 미래 판매를 유도하는 보조 모듈 (v6)

- 초기에는 v5의 변환 로직을 간략히 래핑/위임한다.
"""

from __future__ import annotations

from typing import Sequence
import pandas as pd

from scm_dashboard_v5.ui.charts import _sales_forecast_from_inventory_projection as _v5_project


def derive_future_sales_from_inventory(
    inv_actual: pd.DataFrame,
    inv_forecast: pd.DataFrame,
    *,
    centers: Sequence[str],
    skus: Sequence[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    today: pd.Timestamp,
) -> pd.DataFrame:
    """v5의 재고→판매 변환을 호출하여 미래 판매를 계산한다.

    - 향후 이 모듈을 순수 계산으로 재구성하여 v5에 의존하지 않도록 전환한다.
    """

    return _v5_project(
        inv_actual,
        inv_forecast,
        centers=centers,
        skus=skus,
        start=start,
        end=end,
        today=today,
    )


