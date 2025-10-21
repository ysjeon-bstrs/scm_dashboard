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
    int_columns: Sequence[str] | None = (),
    decimals: int = 4,
) -> pd.DataFrame:
    """Return a copy of *frame* sorted and rounded for tolerant comparisons."""

    out = normalise_dates(frame)

    for column in round_columns or []:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
            out[column] = np.round(out[column].fillna(0.0), decimals)

    for column in int_columns or []:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0).astype(int)

    if sort_by:
        present = [col for col in sort_by if col in out.columns]
        if present:
            out = out.sort_values(present)

    out = out.reset_index(drop=True)

    # Ensure a deterministic column order with sort keys leading.
    if sort_by:
        leading = [col for col in sort_by if col in out.columns]
        trailing = [col for col in out.columns if col not in leading]
        out = out[leading + trailing]

    return out


def _row_identifier(
    actual_row: pd.Series,
    expected_row: pd.Series,
    identifier_columns: Sequence[str] | None,
) -> str:
    if not identifier_columns:
        return f"index={actual_row.name}"

    parts: list[str] = []
    for column in identifier_columns:
        value = actual_row.get(column)
        if pd.isna(value):
            value = expected_row.get(column)
        parts.append(f"{column}={value!r}")
    return ", ".join(parts)


def _log_differences(
    label: str,
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    *,
    identifier_columns: Sequence[str] | None,
    max_diffs: int = 20,
) -> str:
    combined_columns = sorted(set(actual.columns) | set(expected.columns))
    actual_aligned = actual.reindex(columns=combined_columns)
    expected_aligned = expected.reindex(columns=combined_columns)

    mask = actual_aligned.ne(expected_aligned) & ~(
        actual_aligned.isna() & expected_aligned.isna()
    )

    rows, cols = np.where(mask.to_numpy(dtype=bool))
    if len(rows) == 0:
        shape_msg = (
            f"[{label}] DataFrames differ by shape or column set: "
            f"actual{actual.shape} vs expected{expected.shape}; "
            f"columns actual={list(actual.columns)}, expected={list(expected.columns)}"
        )
        return shape_msg

    messages: list[str] = []
    col_names = list(mask.columns)
    for idx, col_idx in zip(rows[:max_diffs], cols[:max_diffs]):
        column = col_names[col_idx]
        actual_row = actual_aligned.iloc[idx]
        expected_row = expected_aligned.iloc[idx]
        context = _row_identifier(actual_row, expected_row, identifier_columns)
        messages.append(
            (
                f"[{label}] row {idx} ({context}) column '{column}': "
                f"actual={actual_row[column]!r}, expected={expected_row[column]!r}"
            )
        )

    if len(rows) > max_diffs:
        messages.append(
            f"[{label}] ... truncated {len(rows) - max_diffs} additional differences"
        )

    return "\n".join(messages)


def assert_frame_equivalent(
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    *,
    sort_by: Sequence[str],
    label: str = "frame",
    row_identifier: Sequence[str] | None = None,
) -> None:
    actual_norm = normalise_dataframe(actual, sort_by=sort_by)
    expected_norm = normalise_dataframe(expected, sort_by=sort_by)
    try:
        pd.testing.assert_frame_equal(
            actual_norm,
            expected_norm,
            check_dtype=False,
            check_exact=False,
            atol=1e-6,
        )
    except AssertionError as err:
        diff_report = _log_differences(
            label,
            actual_norm,
            expected_norm,
            identifier_columns=row_identifier or sort_by,
        )
        raise AssertionError(f"{label} mismatch:\n{diff_report}") from err

