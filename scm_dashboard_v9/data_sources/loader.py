"""Unified data loading interfaces for files and Google Sheets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd


class Loader(Protocol):
    """Simple protocol describing a load operation that returns a DataFrame."""

    def load(self) -> pd.DataFrame:  # pragma: no cover - interface definition
        ...


@dataclass(frozen=True)
class StaticFrameLoader:
    """Loader implementation that simply returns an in-memory DataFrame."""

    frame: pd.DataFrame

    def load(self) -> pd.DataFrame:
        return self.frame.copy()
