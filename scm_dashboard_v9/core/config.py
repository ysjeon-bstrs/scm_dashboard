"""Configuration and constants for the SCM dashboard v9.

센터별 재고 컬럼 매핑, Google Sheets ID 등 전역 설정을 제공합니다.
"""

from __future__ import annotations

# Google Sheets 문서 ID
GSHEET_ID = "1RYjKW2UDJ2kWJLAqQH26eqx2-r9Xb0_qE_hfwu9WIj8"

# 센터별 snapshot_raw에서 사용하는 재고 컬럼명 매핑
CENTER_COL = {
    "태광KR": "stock2",
    "AMZUS": "fba_available_stock",
    "품고KR": "poomgo_v2_available_stock",
    "SBSPH": "shopee_ph_available_stock",
    "SBSSG": "shopee_sg_available_stock",
    "SBSMY": "shopee_my_available_stock",
    "AcrossBUS": "acrossb_available_stock",
    "어크로스비US": "acrossb_available_stock",
}
