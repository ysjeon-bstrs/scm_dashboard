"""Forecasting facade for SCM dashboard v5."""

from .consumption import apply_consumption_with_events, estimate_daily_consumption

__all__ = ["apply_consumption_with_events", "estimate_daily_consumption"]
