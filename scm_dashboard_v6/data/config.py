"""
데이터 설정 (v6)

- 시트명/컬럼명 등 I/O 구성 정보를 한 곳에서 관리한다.
- 환경별 상이 값을 덮어쓸 수 있도록 키 기반 접근을 제공한다.
"""

from __future__ import annotations

SHEETS = {
    "snapshot_refined": "snapshot_refined",
    "moves": "moves",
    "incoming": "incoming",
    "snapshot_raw": "snapshot_raw",
}

COLUMNS = {
    "date": ["date", "snapshot_date", "스냅샷일자"],
    "center": ["center", "센터", "창고", "창고명"],
    "resource_code": ["resource_code", "sku", "상품코드"],
    "stock_qty": ["stock_qty", "qty", "quantity"],
}
