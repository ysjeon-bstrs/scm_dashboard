"""v6 애플리케이션 계층에서 v4 데이터 로더를 감싸는 전달 래퍼."""

from __future__ import annotations

from typing import Optional, Tuple

import pandas as pd

from scm_dashboard_v4.loaders import (
    load_from_excel as _load_from_excel,
    load_from_gsheet_api as _load_from_gsheet_api,
    load_snapshot_raw as _load_snapshot_raw,
)

__all__ = [
    "load_from_excel",
    "load_from_gsheet_api",
    "load_snapshot_raw",
]


def load_from_gsheet_api() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Google Sheets 원천 데이터를 v4와 동일한 구조로 반환한다.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]
        순서대로 이동(MOVE) 데이터, 정제된 스냅샷, 입고 예정 내역 데이터프레임.
        각 데이터프레임의 컬럼 구성은 v4 로더가 유지 관리하며 v6에서는 수정하지 않는다.
    """

    # Google Sheets API 호출, 인증, 캐싱 로직은 v4 구현에 위임하여 중복을 피한다.
    # v5/v6 파이프라인은 MOVE, SNAPSHOT, INCOMING 프레임의 열 이름을 그대로 기대한다.
    return _load_from_gsheet_api()


def load_from_excel(
    file,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """사용자가 업로드한 엑셀 파일에서 v4와 동일한 데이터프레임 묶음을 추출한다.

    Parameters
    ----------
    file : BinaryIO | bytes | streamlit.UploadedFile
        Streamlit 업로드 컴포넌트에서 전달된 파일 객체 또는 원시 바이트.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame, Optional[pandas.DataFrame], Optional[pandas.DataFrame]]
        이동 데이터, 정제 스냅샷, 입고 예정, snapshot_raw 순으로 구성된 튜플.
        일부 시트가 없을 경우 뒤의 요소는 ``None``이 되며, 예외 상황은 v4에서 처리된다.
    """

    # v4 로더는 시트 유효성 검사, Streamlit 오류 처리, openpyxl 기반 파싱을 담당한다.
    # v6 래퍼에서는 반환 순서와 데이터프레임 스키마가 바뀌지 않도록 그대로 전달한다.
    return _load_from_excel(file)


def load_snapshot_raw() -> pd.DataFrame:
    """캐시된 snapshot_raw 시트 데이터를 v4 로직을 통해 조회한다.

    Returns
    -------
    pandas.DataFrame
        최신 snapshot_raw 시트 데이터. 캐시나 웹앱 훅이 실패하면 빈 데이터프레임을 반환한다.
    """

    # Streamlit 세션 캐시와 웹앱 훅을 이용한 백필 백업 전략은 v4 구현에 의존한다.
    # v6에서는 DataFrame 스키마와 업데이트 타이밍을 보장하기 위해 추가 가공을 하지 않는다.
    return _load_snapshot_raw()
