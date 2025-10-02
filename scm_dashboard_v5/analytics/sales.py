"""Sales adapters re-exporting the existing v4 series helpers."""

from __future__ import annotations

import pandas as pd

from scm_dashboard_v4 import sales as v4_sales


def prepare_amazon_sales_series(sales: pd.DataFrame) -> pd.DataFrame:
    return v4_sales.prepare_amazon_sales_series(sales)
