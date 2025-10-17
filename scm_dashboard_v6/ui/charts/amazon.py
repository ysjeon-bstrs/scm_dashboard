from __future__ import annotations

from typing import Any

import pandas as pd
from scm_dashboard_v5.ui.charts import render_amazon_sales_vs_inventory as _v5_render


def render_amazon_sales_vs_inventory(ctx: Any, **kwargs: Any) -> None:
    """Thin wrapper around v5 renderer (keeps behavior)."""

    try:
        _v5_render(ctx, **kwargs)
    except AttributeError:
        # Guard: moves.event_date path on some environments
        try:
            setattr(
                ctx,
                "moves",
                pd.DataFrame(columns=["event_date", "to_center", "resource_code", "qty_ea"]),
            )
        except Exception:
            pass
        _v5_render(ctx, **kwargs)


