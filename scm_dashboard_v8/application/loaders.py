from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd

from scm_dashboard_v4.processing import merge_wip_as_moves

from ..infrastructure import excel, gsheet
from ..infrastructure.models import DashboardSourceData


class LoadError(RuntimeError):
    """Raised when a data source cannot be loaded successfully."""


@dataclass
class DashboardLoadResult:
    moves: pd.DataFrame
    snapshot: pd.DataFrame
    wip: pd.DataFrame
    snapshot_raw: Optional[pd.DataFrame]
    wip_warning: Optional[str] = None


def _validate_source_data(data: DashboardSourceData, *, source_name: str) -> None:
    if data.moves.empty or data.snapshot.empty:
        raise LoadError(f"{source_name}에서 데이터를 불러올 수 없습니다. 권한을 확인해주세요.")


def _merge_wip(data: DashboardSourceData) -> DashboardLoadResult:
    try:
        moves_with_wip = merge_wip_as_moves(data.moves, data.wip)
        warning: Optional[str] = None
    except Exception as exc:
        moves_with_wip = data.moves
        warning = f"WIP 불러오기 실패: {exc}"
    return DashboardLoadResult(
        moves=moves_with_wip,
        snapshot=data.snapshot,
        wip=data.wip,
        snapshot_raw=data.snapshot_raw,
        wip_warning=warning,
    )


def load_dashboard_from_gsheet() -> DashboardLoadResult:
    """Retrieve dashboard data from Google Sheets and normalize it."""

    try:
        source = gsheet.load_dashboard_data()
    except Exception as exc:
        raise LoadError(f"Google Sheets 데이터를 불러오는 중 오류가 발생했습니다: {exc}") from exc

    _validate_source_data(source, source_name="Google Sheets")
    return _merge_wip(source)


def load_dashboard_from_excel(file: Any) -> DashboardLoadResult:
    """Retrieve dashboard data from an uploaded Excel file."""

    try:
        source = excel.load_dashboard_data(file)
    except Exception as exc:
        raise LoadError(f"엑셀 데이터를 불러오는 중 오류가 발생했습니다: {exc}") from exc

    _validate_source_data(source, source_name="엑셀")
    return _merge_wip(source)
