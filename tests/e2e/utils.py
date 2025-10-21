"""Utility helpers for golden data comparisons."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

GOLDEN_DIR = Path(__file__).resolve().parents[1] / "golden"


def ensure_golden_dir() -> Path:
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    return GOLDEN_DIR


def load_golden_csv(filename: str) -> pd.DataFrame:
    path = GOLDEN_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Golden file missing: {path}")
    return pd.read_csv(path)


def normalise_dates(frame: pd.DataFrame, *, columns: Iterable[str] = ("date",)) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        if column in out.columns:
            out[column] = pd.to_datetime(out[column], errors="coerce").dt.normalize()
    return out


def normalise_dataframe(
    frame: pd.DataFrame,
    *,
    sort_by: Sequence[str] | None = None,
    round_columns: Sequence[str] | None = ("stock_qty", "sales_ea"),
    decimals: int = 4,
) -> pd.DataFrame:
    """Return a copy of *frame* sorted and rounded for tolerant comparisons."""

    out = normalise_dates(frame)
    for column in round_columns or []:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
            out[column] = np.round(out[column].fillna(0.0), decimals)
    if sort_by:
        present = [col for col in sort_by if col in out.columns]
        out = out.sort_values(present).reset_index(drop=True)
    else:
        out = out.reset_index(drop=True)
    return out


def assert_frame_equivalent(actual: pd.DataFrame, expected: pd.DataFrame, *, sort_by: Sequence[str]) -> None:
    actual_norm = normalise_dataframe(actual, sort_by=sort_by)
    expected_norm = normalise_dataframe(expected, sort_by=sort_by)
    pd.testing.assert_frame_equal(actual_norm, expected_norm, check_dtype=False, check_exact=False, atol=1e-6)

