"""
데이터 정규화 유틸리티

이 모듈은 SCM 대시보드에서 사용하는 데이터의 타입과 형식을
표준화하는 함수들을 제공합니다. 모든 날짜는 자정(00:00:00)으로
정규화되며, 문자열/숫자 컬럼은 일관된 타입으로 변환됩니다.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import pandas as pd

from center_alias import normalize_center_value

# 기본적으로 정규화할 날짜 컬럼 목록
DATE_COLUMNS = (
    "onboard_date",
    "arrival_date",
    "eta_date",
    "inbound_date",
    "event_date",
)


# Common column aliases observed in upstream move ledgers. The lists should
# contain lowercase, whitespace-trimmed values so that we can perform a case
# insensitive lookup without allocating extra strings for every comparison.
MOVE_COLUMN_ALIASES: dict[str, Sequence[str]] = {
    "resource_code": (
        "resource_code",
        "resource code",
        "상품코드",
        "sku",
        "sku code",
        "product_code",
    ),
    "qty_ea": (
        "qty_ea",
        "qty",
        "quantity",
        "total_quantity",
        "수량",
        "수량(ea)",
        "ea",
    ),
    "carrier_mode": (
        "carrier_mode",
        "carrier mode",
        "carrier",
        "운송방법",
        "운송수단",
    ),
    "from_center": (
        "from_center",
        "from center",
        "출발창고",
        "출발센터",
    ),
    "to_center": (
        "to_center",
        "to center",
        "도착창고",
        "목적지",
    ),
    "invoice_no": (
        "invoice_no",
        "invoice no",
        "invoice_no",
        "인보이스 번호",
        "주문번호",
        "lot",
        "lot no",
    ),
    "onboard_date": (
        "onboard_date",
        "onboard",
        "depart_date",
        "depart date",
        "배정일",
        "출발일",
        "h",
    ),
    "arrival_date": (
        "arrival_date",
        "arrival",
        "eta",
        "eta_date",
        "도착일",
    ),
    "inbound_date": (
        "inbound_date",
        "입고일",
        "입고완료일",
    ),
    "event_date": (
        "event_date",
        "event",
    ),
}


def normalize_dates(
    frame: pd.DataFrame, *, columns: Iterable[str] = DATE_COLUMNS
) -> pd.DataFrame:
    """
    지정된 날짜 컬럼들을 datetime64 타입으로 변환하고 자정으로 정규화합니다.

    날짜 정규화는 시간 부분을 00:00:00으로 만들어 일 단위 집계를
    정확하게 수행할 수 있도록 합니다.

    Args:
        frame: 원본 데이터프레임
        columns: 정규화할 날짜 컬럼명 목록 (기본값: DATE_COLUMNS)

    Returns:
        날짜 컬럼이 정규화된 데이터프레임 복사본.
        원본은 변경되지 않습니다.

    Examples:
        >>> df = pd.DataFrame({
        ...     "onboard_date": ["2024-01-01 14:30", "2024-01-02 09:15"],
        ...     "qty": [100, 200]
        ... })
        >>> normalized = normalize_dates(df)
        >>> normalized["onboard_date"]
        0   2024-01-01 00:00:00
        1   2024-01-02 00:00:00
    """
    out = frame.copy()

    # ========================================
    # 각 날짜 컬럼을 순회하며 정규화
    # ========================================
    for col in columns:
        if col in out.columns:
            # datetime64로 변환 (변환 실패 시 NaT)
            # dt.normalize()로 시간 부분을 00:00:00으로 설정
            out[col] = pd.to_datetime(out[col], errors="coerce").dt.normalize()

    return out


def _normalise_center_series(series: pd.Series) -> pd.Series:
    """Return a normalised centre series using ``normalize_center_value``."""

    values = series.copy()
    if values.empty:
        return pd.Series("", index=series.index, dtype=object)

    def _normalise(value: object) -> str:
        if pd.isna(value):  # type: ignore[arg-type]
            return ""
        text = str(value).strip()
        if not text:
            return ""
        lowered = text.lower()
        if lowered in {"nan", "none", "null", "<na>"}:
            return ""
        normalised = normalize_center_value(text)
        return normalised or text

    return values.map(_normalise).fillna("")


def _build_column_lookup(columns: Iterable[object]) -> dict[str, str]:
    """Create a mapping of normalised column names to the original labels."""

    lookup: dict[str, str] = {}
    for col in columns:
        name = str(col).strip()
        if not name:
            continue
        lookup[name.casefold()] = name
    return lookup


def _find_column(lookup: dict[str, str], aliases: Sequence[str]) -> Optional[str]:
    """Return the first matching column for *aliases* using *lookup*."""

    for alias in aliases:
        key = str(alias).strip().casefold()
        if not key:
            continue
        found = lookup.get(key)
        if found is not None:
            return found

    # Fallback to partial matches so inputs like ``depart date`` match
    # ``depart_date`` and vice versa.
    for alias in aliases:
        key = str(alias).strip().casefold()
        if not key:
            continue
        for candidate_key, original in lookup.items():
            if key in candidate_key or candidate_key in key:
                return original
    return None


def normalize_moves(frame: pd.DataFrame) -> pd.DataFrame:
    """
    이동 원장 데이터의 핵심 컬럼들을 예측 가능한 타입으로 변환합니다.

    정규화되는 컬럼:
    - carrier_mode: 운송 방식 (문자열, 대문자 변환)
    - resource_code: SKU 코드 (문자열)
    - from_center: 출발 센터 (문자열)
    - to_center: 목적지 센터 (문자열)
    - qty_ea: 수량 (숫자, NaN은 0으로 대체)
    - 날짜 컬럼들: datetime64로 정규화

    Args:
        frame: 원본 이동 원장 데이터프레임

    Returns:
        정규화된 이동 원장 데이터프레임

    Examples:
        >>> raw_moves = pd.read_csv("moves.csv")
        >>> normalized_moves = normalize_moves(raw_moves)
    """
    # ========================================
    # 1단계: 날짜 컬럼 정규화
    # ========================================
    column_lookup = _build_column_lookup(frame.columns)
    rename_map: dict[str, str] = {}
    consumed: set[str] = set()

    for canonical, aliases in MOVE_COLUMN_ALIASES.items():
        match = _find_column(column_lookup, aliases)
        if match is None or match in consumed or match == canonical:
            continue
        rename_map[match] = canonical
        consumed.add(match)

    out = normalize_dates(frame.rename(columns=rename_map))

    # ========================================
    # 2단계: carrier_mode (운송 방식) 정규화
    # ========================================
    # 컬럼이 없으면 빈 문자열로 채운 시리즈 생성
    carrier_src = (
        out["carrier_mode"]
        if "carrier_mode" in out.columns
        else pd.Series("", index=out.index)
    )
    # 문자열로 변환하고 대문자/공백 정리 (예: "sea" -> "SEA")
    out["carrier_mode"] = carrier_src.astype(str).str.strip().str.upper()

    # ========================================
    # 3단계: resource_code (SKU) 정규화
    # ========================================
    resource_src = (
        out["resource_code"]
        if "resource_code" in out.columns
        else pd.Series("", index=out.index)
    )
    out["resource_code"] = resource_src.astype(str).str.strip()

    # ========================================
    # 4단계: from_center (출발 센터) 정규화
    # ========================================
    from_src = (
        out["from_center"]
        if "from_center" in out.columns
        else pd.Series("", index=out.index)
    )
    out["from_center"] = _normalise_center_series(from_src)

    # ========================================
    # 5단계: to_center (목적지 센터) 정규화
    # ========================================
    to_src = (
        out["to_center"]
        if "to_center" in out.columns
        else pd.Series("", index=out.index)
    )
    out["to_center"] = _normalise_center_series(to_src)

    # ========================================
    # 6단계: qty_ea (수량) 정규화
    # ========================================
    qty_src = (
        out["qty_ea"] if "qty_ea" in out.columns else pd.Series(0, index=out.index)
    )
    # 숫자로 변환 (변환 실패 시 NaN -> 0으로 대체)
    out["qty_ea"] = (
        pd.to_numeric(qty_src.astype(str).str.replace(",", ""), errors="coerce")
        .fillna(0)
        .astype(int)
    )

    return out


def normalize_snapshot(frame: pd.DataFrame) -> pd.DataFrame:
    """
    스냅샷 데이터의 스키마를 표준화하여 하류 소비자가 사용할 수 있도록 합니다.

    표준 스키마:
    - date: 스냅샷 날짜 (datetime64, 자정으로 정규화)
    - center: 센터명 (문자열)
    - resource_code: SKU 코드 (문자열)
    - stock_qty: 재고 수량 (숫자)

    선택적 컬럼 (있으면 포함):
    - sales_qty: 판매 수량 (숫자)
    - resource_name: 품명 (문자열)
    - stock_available: 사용가능 재고 (숫자, Amazon FBA)
    - stock_expected: 입고예정 재고 (숫자, Amazon FBA)
    - stock_processing: 입고처리중 재고 (숫자, Amazon FBA)
    - stock_readytoship: 입고등록 재고 (숫자, Amazon FBA)
    - pending_fc: FC 재배치중 재고 (숫자, Amazon FBA)
    - snap_time: 스냅샷 시간 (datetime64, Amazon FBA)
    - selling_speed: 판매속도 (숫자, SHOPEE)
    - coverage_days: 커버일수 (숫자, SHOPEE)

    Args:
        frame: 원본 스냅샷 데이터프레임

    Returns:
        표준 스키마로 정규화된 스냅샷 데이터프레임.
        기본 컬럼 + 선택적 컬럼 포함.

    Raises:
        KeyError: 'date' 또는 'snapshot_date' 컬럼이 없을 경우

    Examples:
        >>> raw_snapshot = pd.read_csv("snapshot.csv")
        >>> normalized = normalize_snapshot(raw_snapshot)
        >>> normalized.columns.tolist()
        ['date', 'center', 'resource_code', 'stock_qty', 'sales_qty', ...]
    """
    out = frame.copy()

    cols_lower = {str(col).strip().lower(): col for col in out.columns}

    def _pick_column(keys: Iterable[str]) -> Optional[str]:
        for key in keys:
            if key in cols_lower:
                return cols_lower[key]
        return None

    date_col = _pick_column(["date", "snapshot_date"])
    center_col = _pick_column(["center", "센터", "창고", "warehouse"])
    resource_col = _pick_column(
        ["resource_code", "resource_cc", "sku", "상품코드", "product_code"]
    )
    stock_col = _pick_column(["stock_qty", "qty", "수량", "재고", "quantity"])
    sales_col = _pick_column(
        ["sales_qty", "sale_qty", "판매량", "출고수량", "출고", "출고 수량", "sales_ea"]
    )
    name_col = _pick_column(["resource_name", "품명", "상품명", "product_name"])

    # Amazon FBA 스냅샷 전용 컬럼 (선택적)
    available_col = _pick_column(["stock_available", "available_qty", "사용가능재고"])
    expected_col = _pick_column(["stock_expected", "expected_qty", "입고예정"])
    processing_col = _pick_column(["stock_processing", "processing_qty", "입고처리중"])
    readytoship_col = _pick_column(["stock_readytoship", "readytoship_qty", "입고등록"])
    snap_time_col = _pick_column(["snap_time", "snapshot_time", "snapshot_datetime"])
    pending_fc_col = _pick_column(
        ["pending_fc", "pending_fc_qty", "fc_pending", "pending at fc"]
    )  # 신규 컬럼

    # SHOPEE 스냅샷 전용 컬럼 (선택적)
    selling_speed_col = _pick_column(["selling_speed", "sales_speed", "판매속도"])
    coverage_days_col = _pick_column(["coverage_days", "cover_days", "커버일수"])

    if not date_col or not center_col or not resource_col or not stock_col:
        raise KeyError(
            "snapshot frame must include date/center/resource_code/stock_qty columns"
        )

    rename_map = {
        date_col: "date",
        center_col: "center",
        resource_col: "resource_code",
        stock_col: "stock_qty",
    }
    if sales_col:
        rename_map[sales_col] = "sales_qty"
    if name_col:
        rename_map[name_col] = "resource_name"

    # Amazon FBA 컬럼 매핑 (선택적)
    if available_col:
        rename_map[available_col] = "stock_available"
    if expected_col:
        rename_map[expected_col] = "stock_expected"
    if processing_col:
        rename_map[processing_col] = "stock_processing"
    if readytoship_col:
        rename_map[readytoship_col] = "stock_readytoship"
    if snap_time_col:
        rename_map[snap_time_col] = "snap_time"
    if pending_fc_col:
        rename_map[pending_fc_col] = "pending_fc"

    # SHOPEE 컬럼 매핑 (선택적)
    if selling_speed_col:
        rename_map[selling_speed_col] = "selling_speed"
    if coverage_days_col:
        rename_map[coverage_days_col] = "coverage_days"

    out = out.rename(columns=rename_map)

    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out["center"] = _normalise_center_series(
        out.get("center", pd.Series("", index=out.index))
    )
    out["resource_code"] = (
        out.get("resource_code", pd.Series("", index=out.index)).astype(str).str.strip()
    )
    out["stock_qty"] = (
        pd.to_numeric(out.get("stock_qty"), errors="coerce").fillna(0.0).astype(float)
    )

    if "sales_qty" in out.columns:
        out["sales_qty"] = (
            pd.to_numeric(out.get("sales_qty"), errors="coerce").fillna(0).astype(int)
        )
    else:
        out["sales_qty"] = 0

    if "resource_name" in out.columns:
        out["resource_name"] = (
            out.get("resource_name", pd.Series("", index=out.index))
            .astype(str)
            .str.strip()
            .replace({"nan": "", "None": ""})
        )

    # Amazon FBA 컬럼 타입 변환 (선택적) - 모두 float 보장
    if "stock_available" in out.columns:
        out["stock_available"] = (
            pd.to_numeric(out.get("stock_available"), errors="coerce")
            .fillna(0.0)
            .astype(float)
        )
    if "stock_expected" in out.columns:
        out["stock_expected"] = (
            pd.to_numeric(out.get("stock_expected"), errors="coerce")
            .fillna(0.0)
            .astype(float)
        )
    if "stock_processing" in out.columns:
        out["stock_processing"] = (
            pd.to_numeric(out.get("stock_processing"), errors="coerce")
            .fillna(0.0)
            .astype(float)
        )
    if "stock_readytoship" in out.columns:
        out["stock_readytoship"] = (
            pd.to_numeric(out.get("stock_readytoship"), errors="coerce")
            .fillna(0.0)
            .astype(float)
        )
    if "pending_fc" in out.columns:
        out["pending_fc"] = (
            pd.to_numeric(out.get("pending_fc"), errors="coerce")
            .fillna(0.0)
            .astype(float)
        )
    if "snap_time" in out.columns:
        if "center" in out.columns:
            for center in out["center"].unique():
                center_mask = out["center"] == center
                out.loc[center_mask, "snap_time"] = pd.to_datetime(
                    out.loc[center_mask, "snap_time"], errors="coerce"
                )
            # 센터별 변환 후 전체 컬럼을 datetime64 타입으로 확정
            # (object dtype에 Timestamp 객체들이 있는 상태를 datetime64로 변환)
            out["snap_time"] = pd.to_datetime(out["snap_time"], errors="coerce")
        else:
            # center 컬럼이 없으면 전체 변환
            out["snap_time"] = pd.to_datetime(out.get("snap_time"), errors="coerce")

    # SHOPEE 컬럼 타입 변환 (선택적)
    if "selling_speed" in out.columns:
        out["selling_speed"] = (
            pd.to_numeric(out.get("selling_speed"), errors="coerce")
            .fillna(0.0)
            .astype(float)
        )
    if "coverage_days" in out.columns:
        out["coverage_days"] = (
            pd.to_numeric(out.get("coverage_days"), errors="coerce")
            .fillna(0.0)
            .astype(float)
        )

    out = out.dropna(subset=["date"])
    out = out[(out["center"] != "") & (out["resource_code"] != "")]

    # 기본 컬럼
    columns = ["date", "center", "resource_code", "stock_qty"]
    # 선택적 컬럼 추가
    for optional_col in [
        "sales_qty",
        "resource_name",
        "stock_available",
        "stock_expected",
        "stock_processing",
        "stock_readytoship",
        "pending_fc",
        "snap_time",
        "selling_speed",
        "coverage_days",
    ]:
        if optional_col in out.columns:
            columns.append(optional_col)

    return out[columns].reset_index(drop=True)
