"""
필터 옵션 추출 및 날짜 범위 계산

이 모듈은 데이터프레임으로부터 센터/SKU 옵션을 추출하고,
대시보드의 날짜 범위 슬라이더를 위한 경계값을 계산합니다.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import pandas as pd

from center_alias import normalize_center_value


# ========================================
# 날짜 정규화 헬퍼
# ========================================


def safe_to_datetime(
    series_or_value,
    *,
    normalize: bool = True,
    errors: str = "coerce",
) -> pd.Series | pd.Timestamp:
    """
    날짜를 안전하게 datetime으로 변환하고 선택적으로 정규화합니다.

    Args:
        series_or_value: Series 또는 단일 값
        normalize: True이면 시간을 00:00:00으로 정규화
        errors: pd.to_datetime의 errors 파라미터 (기본: 'coerce')

    Returns:
        변환된 datetime Series 또는 Timestamp

    Examples:
        >>> safe_to_datetime("2024-01-15 12:30:00")
        Timestamp('2024-01-15 00:00:00')
        >>> safe_to_datetime(df["date"], normalize=False)
        <Series of datetimes>
    """
    result = pd.to_datetime(series_or_value, errors=errors)
    if normalize:
        if isinstance(result, pd.Series):
            return result.dt.normalize()
        elif isinstance(result, pd.Timestamp) and not pd.isna(result):
            return result.normalize()
    return result


def normalize_timestamp(value) -> Optional[pd.Timestamp]:
    """
    단일 값을 정규화된 Timestamp로 변환합니다.

    Args:
        value: 변환할 날짜 값

    Returns:
        정규화된 Timestamp. 변환 실패 시 None.

    Examples:
        >>> normalize_timestamp("2024-01-15")
        Timestamp('2024-01-15 00:00:00')
        >>> normalize_timestamp(None)
        None
    """
    if pd.isna(value):
        return None
    try:
        ts = pd.to_datetime(value, errors="coerce")
        return ts.normalize() if not pd.isna(ts) else None
    except Exception:
        return None


# ========================================
# DataFrame 필터 헬퍼
# ========================================


def filter_by_centers(
    df: pd.DataFrame,
    centers: str | Sequence[str],
    *,
    center_col: str = "center",
) -> pd.DataFrame:
    """
    DataFrame을 센터 목록으로 필터링합니다.

    Args:
        df: 필터링할 DataFrame
        centers: 단일 센터명 또는 센터 목록
        center_col: 센터 컬럼명 (기본: 'center')

    Returns:
        필터링된 DataFrame (복사본)

    Examples:
        >>> filtered = filter_by_centers(df, ["태광KR", "AMZUS"])
        >>> filtered = filter_by_centers(df, "태광KR")
    """
    if df.empty or center_col not in df.columns:
        return df.copy()

    centers_list = [centers] if isinstance(centers, str) else list(centers)
    if not centers_list:
        return df.copy()

    return df[df[center_col].isin(centers_list)].copy()


def filter_by_skus(
    df: pd.DataFrame,
    skus: str | Sequence[str],
    *,
    sku_col: str = "resource_code",
) -> pd.DataFrame:
    """
    DataFrame을 SKU 목록으로 필터링합니다.

    Args:
        df: 필터링할 DataFrame
        skus: 단일 SKU 또는 SKU 목록
        sku_col: SKU 컬럼명 (기본: 'resource_code')

    Returns:
        필터링된 DataFrame (복사본)

    Examples:
        >>> filtered = filter_by_skus(df, ["BA00021", "BA00022"])
        >>> filtered = filter_by_skus(df, "BA00021")
    """
    if df.empty or sku_col not in df.columns:
        return df.copy()

    skus_list = [skus] if isinstance(skus, str) else list(skus)
    if not skus_list:
        return df.copy()

    return df[df[sku_col].isin(skus_list)].copy()


def filter_by_centers_and_skus(
    df: pd.DataFrame,
    *,
    centers: Optional[str | Sequence[str]] = None,
    skus: Optional[str | Sequence[str]] = None,
    center_col: str = "center",
    sku_col: str = "resource_code",
) -> pd.DataFrame:
    """
    DataFrame을 센터와 SKU 목록으로 동시에 필터링합니다.

    Args:
        df: 필터링할 DataFrame
        centers: 센터 목록 (None이면 필터링 안 함)
        skus: SKU 목록 (None이면 필터링 안 함)
        center_col: 센터 컬럼명
        sku_col: SKU 컬럼명

    Returns:
        필터링된 DataFrame (복사본)

    Examples:
        >>> filtered = filter_by_centers_and_skus(
        ...     df,
        ...     centers=["태광KR", "AMZUS"],
        ...     skus=["BA00021", "BA00022"]
        ... )
    """
    result = df.copy()

    if centers is not None:
        result = filter_by_centers(result, centers, center_col=center_col)

    if skus is not None:
        result = filter_by_skus(result, skus, sku_col=sku_col)

    return result


# ========================================
# 검증 헬퍼
# ========================================


def is_empty_or_none(df: Optional[pd.DataFrame]) -> bool:
    """
    DataFrame이 None이거나 비어있는지 확인합니다.

    Args:
        df: 확인할 DataFrame

    Returns:
        None이거나 비어있으면 True

    Examples:
        >>> is_empty_or_none(None)
        True
        >>> is_empty_or_none(pd.DataFrame())
        True
        >>> is_empty_or_none(pd.DataFrame({"a": [1]}))
        False
    """
    return df is None or df.empty


def ensure_list(value: Optional[str | Sequence[str]]) -> list[str]:
    """
    단일 값 또는 Sequence를 리스트로 변환합니다.

    Args:
        value: 단일 값, Sequence, 또는 None

    Returns:
        리스트. None이면 빈 리스트.

    Examples:
        >>> ensure_list("BA00021")
        ['BA00021']
        >>> ensure_list(["BA00021", "BA00022"])
        ['BA00021', 'BA00022']
        >>> ensure_list(None)
        []
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


# ========================================
# 기존 함수들
# ========================================


def norm_center(x: str) -> Optional[str]:
    """
    센터명을 정규화합니다.

    Args:
        x: 원본 센터명

    Returns:
        정규화된 센터명. 유효하지 않으면 None.

    Examples:
        >>> norm_center("amz us")
        'AMZUS'
        >>> norm_center("태광 kr")
        '태광KR'
    """
    return normalize_center_value(x)


def extract_center_and_sku_options(
    moves: pd.DataFrame, snapshot: pd.DataFrame
) -> Tuple[list[str], list[str]]:
    """
    이동 원장과 스냅샷 데이터로부터 선택 가능한 센터와 SKU 목록을 추출합니다.

    센터는 다음 위치에서 수집됩니다:
    - snapshot의 'center' 컬럼
    - moves의 'from_center' 및 'to_center' 컬럼

    SKU는 snapshot의 'resource_code'를 우선하고,
    없으면 moves의 'resource_code'를 사용합니다.

    Args:
        moves: 이동 원장 데이터프레임
        snapshot: 스냅샷 데이터프레임

    Returns:
        (centers, skus) 튜플:
        - centers: 정렬된 센터명 목록
        - skus: 정렬된 SKU 코드 목록

    Examples:
        >>> centers, skus = extract_center_and_sku_options(moves_df, snapshot_df)
        >>> centers
        ['AMZUS', '상해CN', '태광KR']
        >>> skus
        ['BA00021', 'BA00022', 'BA00023']
    """
    # ========================================
    # 1단계: 스냅샷에서 센터 추출
    # ========================================
    snap_centers_src = snapshot.get("center")

    if snap_centers_src is None:
        snap_centers = pd.Series(dtype=str)
    else:
        snap_centers = snap_centers_src.dropna().astype(str).str.strip()

    # ========================================
    # 2단계: 이동 원장에서 센터 추출 (출발지 + 목적지)
    # ========================================
    move_centers = (
        pd.concat(
            [
                moves.get("from_center", pd.Series(dtype=object)),
                moves.get("to_center", pd.Series(dtype=object)),
            ],
            ignore_index=True,
        )
        .dropna()
        .astype(str)
        .str.strip()
    )

    # ========================================
    # 3단계: 모든 센터를 결합하고 정규화
    # ========================================
    all_candidates = pd.concat([snap_centers, move_centers], ignore_index=True).dropna()

    # 센터명 정규화 및 중복 제거
    centers = sorted({c for c in (norm_center(value) for value in all_candidates) if c})

    # ========================================
    # 4단계: SKU 추출 (snapshot 우선, 없으면 moves)
    # ========================================
    skus = sorted(snapshot["resource_code"].dropna().astype(str).unique().tolist())

    if not skus:
        skus = sorted(moves["resource_code"].dropna().astype(str).unique().tolist())

    return centers, skus


def calculate_move_date_bounds(
    moves: pd.DataFrame,
) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """
    이동 원장에서 사용 가능한 가장 이른 날짜와 가장 늦은 날짜를 반환합니다.

    다음 날짜 컬럼들을 검사합니다:
    - onboard_date: 출발일
    - arrival_date: 도착일
    - inbound_date: 입고일
    - event_date: 이벤트일 (WIP 완료일 등)

    Args:
        moves: 이동 원장 데이터프레임

    Returns:
        (min_date, max_date) 튜플:
        - min_date: 가장 이른 날짜. 없으면 None.
        - max_date: 가장 늦은 날짜. 없으면 None.

    Examples:
        >>> min_dt, max_dt = calculate_move_date_bounds(moves_df)
        >>> min_dt
        Timestamp('2024-01-01 00:00:00')
        >>> max_dt
        Timestamp('2024-02-28 00:00:00')
    """
    # ========================================
    # 1단계: 날짜 컬럼 찾기
    # ========================================
    date_columns = [
        col
        for col in ("onboard_date", "arrival_date", "inbound_date", "event_date")
        if col in moves.columns
    ]

    if not date_columns:
        return None, None

    # ========================================
    # 2단계: 각 날짜 컬럼을 datetime64로 변환하고 정규화
    # ========================================
    normalized_dates = []
    for col in date_columns:
        series = pd.to_datetime(moves[col], errors="coerce").dropna()
        if not series.empty:
            normalized_dates.append(series.dt.normalize())

    if not normalized_dates:
        return None, None

    # ========================================
    # 3단계: 모든 날짜를 결합하여 min/max 계산
    # ========================================
    combined = pd.concat(normalized_dates, ignore_index=True)

    if combined.empty:
        return None, None

    min_dt = combined.min()
    max_dt = combined.max()

    if pd.isna(min_dt) or pd.isna(max_dt):
        return None, None

    return pd.Timestamp(min_dt).normalize(), pd.Timestamp(max_dt).normalize()


def calculate_date_bounds(
    *,
    today: pd.Timestamp,
    snapshot_df: pd.DataFrame,
    moves_df: pd.DataFrame,
    base_past_days: int,
    base_future_days: int,
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    대시보드 기간 슬라이더를 위한 선택 가능한 날짜 범위를 계산합니다.

    날짜 범위는 다음 요소들을 고려하여 결정됩니다:
    1. 기본 범위: today ± base_past_days/base_future_days
    2. 스냅샷 데이터의 날짜 범위
    3. 이동 원장의 날짜 범위

    최종 범위는 위 세 가지의 교집합으로 결정됩니다.

    Args:
        today: 오늘 날짜 (기준일)
        snapshot_df: 스냅샷 데이터프레임
        moves_df: 이동 원장 데이터프레임
        base_past_days: 기본 과거 범위 (일)
        base_future_days: 기본 미래 범위 (일)

    Returns:
        (bound_min, bound_max) 튜플:
        - bound_min: 선택 가능한 최소 날짜
        - bound_max: 선택 가능한 최대 날짜

    Examples:
        >>> bounds = calculate_date_bounds(
        ...     today=pd.Timestamp("2024-01-15"),
        ...     snapshot_df=snapshot,
        ...     moves_df=moves,
        ...     base_past_days=42,
        ...     base_future_days=42
        ... )
        >>> bounds
        (Timestamp('2023-12-04 00:00:00'), Timestamp('2024-02-26 00:00:00'))
    """
    # ========================================
    # 1단계: 기본 날짜 범위 계산
    # ========================================
    normalized_today = pd.Timestamp(today).normalize()
    base_min = (normalized_today - pd.Timedelta(days=base_past_days)).normalize()
    base_max = (normalized_today + pd.Timedelta(days=base_future_days)).normalize()

    # ========================================
    # 2단계: 후보 경계값 초기화
    # ========================================
    bound_min_candidates = [base_min]
    bound_max_candidates = [base_max]

    # ========================================
    # 3단계: 스냅샷 데이터의 날짜 범위 추가
    # ========================================
    snap_dates = pd.to_datetime(snapshot_df.get("date"), errors="coerce").dropna()

    if not snap_dates.empty:
        bound_min_candidates.append(snap_dates.min().normalize())
        bound_max_candidates.append(snap_dates.max().normalize())

    # ========================================
    # 4단계: 이동 원장의 날짜 범위 추가
    # ========================================
    move_min, move_max = calculate_move_date_bounds(moves_df)

    if move_min is not None:
        bound_min_candidates.append(move_min)
    if move_max is not None:
        bound_max_candidates.append(move_max)

    # ========================================
    # 5단계: 동적 범위 계산 (데이터 기반)
    # ========================================
    dynamic_min = min(bound_min_candidates)
    dynamic_max = max(bound_max_candidates)

    # ========================================
    # 6단계: 기본 범위 내로 제한
    # ========================================
    # 데이터 범위가 너무 넓어지지 않도록 기본 범위 내로 클램핑
    bound_min = max(dynamic_min, base_min)
    bound_max = min(dynamic_max, base_max)

    # ========================================
    # 7단계: 범위 유효성 검증
    # ========================================
    # min이 max보다 크면 max로 조정
    if bound_min > bound_max:
        bound_min = bound_max

    return bound_min.normalize(), bound_max.normalize()
