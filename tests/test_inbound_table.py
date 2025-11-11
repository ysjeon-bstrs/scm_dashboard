"""
입고 예정 테이블 모듈 테스트

테스트 범위:
- 인보이스별 그룹핑 및 집계
- 대표 SKU 선정 (수량 최다, 동률 시 사전순)
- SKU 요약 생성
- ETA 색상 규칙 적용
- 경로 생성
- 성능 테스트 (1만 행 ≤ 200ms)
"""

from datetime import datetime, timedelta

import pandas as pd
import pytest

from scm_dashboard_v9.ui.inbound_table import build_inbound_table


class TestBuildInboundTable:
    """build_inbound_table 함수 테스트"""

    @pytest.fixture
    def sample_raw_data(self):
        """테스트용 샘플 원본 데이터"""
        today = pd.Timestamp.today().normalize()

        return pd.DataFrame(
            {
                "invoice_no": ["INV001", "INV001", "INV001", "INV002", "INV003"],
                "from_country": ["KR", "KR", "KR", "US", "CN"],
                "to_country": ["US", "US", "US", "KR", "KR"],
                "to_center": ["AMZUS", "AMZUS", "AMZUS", "태광KR", "태광KR"],
                "resource_code": ["BA00021", "BA00022", "BA00023", "BA00024", "BA00025"],
                "resource_name": ["제품A", "제품B", "제품C", "제품D", "제품E"],
                "qty_ea": [1000, 500, 300, 200, 150],
                "carrier_mode": ["특송", "특송", "특송", "해운", "항공"],
                "onboard_date": [
                    today - timedelta(days=10),
                    today - timedelta(days=10),
                    today - timedelta(days=10),
                    today - timedelta(days=5),
                    today - timedelta(days=1),
                ],
                "pred_inbound_date": [
                    today - timedelta(days=1),  # 과거 (red)
                    today - timedelta(days=1),
                    today - timedelta(days=1),
                    today + timedelta(days=3),  # 5일 이내 (green)
                    "not_defined",  # 미확인 (orange)
                ],
            }
        )

    @pytest.fixture
    def sample_sku_color_map(self):
        """테스트용 SKU 색상 매핑"""
        return {
            "BA00021": "#4E79A7",
            "BA00022": "#F28E2B",
            "BA00023": "#E15759",
            "BA00024": "#76B7B2",
            "BA00025": "#59A14F",
        }

    def test_empty_dataframe(self, sample_sku_color_map):
        """빈 데이터프레임 처리 테스트"""
        empty_df = pd.DataFrame()
        result = build_inbound_table(empty_df, sample_sku_color_map)
        assert result.empty

    def test_invoice_grouping(self, sample_raw_data, sample_sku_color_map):
        """인보이스별 그룹핑 테스트"""
        result = build_inbound_table(sample_raw_data, sample_sku_color_map)

        # 3개의 인보이스 → 3개의 행
        assert len(result) == 3

        # 인보이스 번호 확인
        assert set(result["invoice_no"]) == {"INV001", "INV002", "INV003"}

    def test_representative_sku_selection(self, sample_raw_data, sample_sku_color_map):
        """대표 SKU 선정 테스트 (수량 최다)"""
        result = build_inbound_table(sample_raw_data, sample_sku_color_map)

        # INV001의 대표 SKU는 BA00021 (수량 1000ea)
        inv001_row = result[result["invoice_no"] == "INV001"].iloc[0]
        assert inv001_row["_rep_sku"] == "BA00021"
        assert "BA00021: 1,000ea" in inv001_row["sku_summary"]

    def test_representative_sku_tie_breaking(self, sample_sku_color_map):
        """동일 수량 시 사전순 선정 테스트"""
        # 동일 수량 데이터
        data = pd.DataFrame(
            {
                "invoice_no": ["INV999", "INV999", "INV999"],
                "from_country": ["KR", "KR", "KR"],
                "to_country": ["US", "US", "US"],
                "to_center": ["AMZUS", "AMZUS", "AMZUS"],
                "resource_code": ["BA00033", "BA00022", "BA00011"],  # 사전순: BA00011
                "resource_name": ["제품X", "제품Y", "제품Z"],
                "qty_ea": [500, 500, 500],  # 모두 동일
                "carrier_mode": ["특송", "특송", "특송"],
                "onboard_date": ["2025-01-15", "2025-01-15", "2025-01-15"],
                "pred_inbound_date": ["2025-01-20", "2025-01-20", "2025-01-20"],
            }
        )

        result = build_inbound_table(data, sample_sku_color_map)

        # 사전순으로 BA00011 선정
        assert result.iloc[0]["_rep_sku"] == "BA00011"

    def test_sku_summary_format(self, sample_raw_data, sample_sku_color_map):
        """SKU 요약 포맷 테스트"""
        result = build_inbound_table(sample_raw_data, sample_sku_color_map)

        # INV001: BA00021이 대표, 총 3종
        inv001_row = result[result["invoice_no"] == "INV001"].iloc[0]
        assert inv001_row["sku_summary"] == "BA00021: 1,000ea 외 2종"

        # INV002: 단일 SKU (외 N종 없음)
        inv002_row = result[result["invoice_no"] == "INV002"].iloc[0]
        assert inv002_row["sku_summary"] == "BA00024: 200ea"

    def test_eta_color_red_past(self, sample_raw_data, sample_sku_color_map):
        """ETA 색상 규칙 - 과거 (빨강) 테스트"""
        result = build_inbound_table(sample_raw_data, sample_sku_color_map)

        # INV001: 과거 날짜 → red
        inv001_row = result[result["invoice_no"] == "INV001"].iloc[0]
        assert inv001_row["eta_color"] == "red"

    def test_eta_color_green_within_5days(self, sample_raw_data, sample_sku_color_map):
        """ETA 색상 규칙 - 5일 이내 (초록) 테스트"""
        result = build_inbound_table(sample_raw_data, sample_sku_color_map)

        # INV002: 3일 후 → green
        inv002_row = result[result["invoice_no"] == "INV002"].iloc[0]
        assert inv002_row["eta_color"] == "green"

    def test_eta_color_orange_undefined(self, sample_raw_data, sample_sku_color_map):
        """ETA 색상 규칙 - 미확인 (주황) 테스트"""
        result = build_inbound_table(sample_raw_data, sample_sku_color_map)

        # INV003: not_defined → orange
        inv003_row = result[result["invoice_no"] == "INV003"].iloc[0]
        assert inv003_row["eta_color"] == "orange"
        assert inv003_row["eta_text"] == "미확인"

    def test_eta_color_gray_after_5days(self, sample_sku_color_map):
        """ETA 색상 규칙 - 6일 이후 (회색) 테스트"""
        today = pd.Timestamp.today().normalize()

        # 7일 후 데이터
        data = pd.DataFrame(
            {
                "invoice_no": ["INV888"],
                "from_country": ["KR"],
                "to_country": ["US"],
                "to_center": ["AMZUS"],
                "resource_code": ["BA00099"],
                "resource_name": ["제품Z"],
                "qty_ea": [100],
                "carrier_mode": ["해운"],
                "onboard_date": [today - timedelta(days=20)],
                "pred_inbound_date": [today + timedelta(days=7)],  # 7일 후
            }
        )

        result = build_inbound_table(data, sample_sku_color_map)
        assert result.iloc[0]["eta_color"] == "gray"

    def test_route_format(self, sample_raw_data, sample_sku_color_map):
        """경로 포맷 테스트"""
        result = build_inbound_table(sample_raw_data, sample_sku_color_map)

        # INV001: KR → US
        inv001_row = result[result["invoice_no"] == "INV001"].iloc[0]
        assert inv001_row["route"] == "KR → US"

        # INV002: US → KR
        inv002_row = result[result["invoice_no"] == "INV002"].iloc[0]
        assert inv002_row["route"] == "US → KR"

    def test_onboard_date_format(self, sample_raw_data, sample_sku_color_map):
        """출발일 포맷 테스트"""
        result = build_inbound_table(sample_raw_data, sample_sku_color_map)

        # 모든 행에 YYYY-MM-DD 포맷 확인
        for _, row in result.iterrows():
            onboard = row["onboard_date"]
            if onboard:  # 빈 문자열이 아니면
                # YYYY-MM-DD 형식 검증
                try:
                    datetime.strptime(onboard, "%Y-%m-%d")
                    valid = True
                except ValueError:
                    valid = False
                assert valid, f"Invalid date format: {onboard}"

    def test_sku_color_html(self, sample_raw_data, sample_sku_color_map):
        """SKU 색상 HTML 적용 테스트"""
        result = build_inbound_table(sample_raw_data, sample_sku_color_map)

        # INV001의 HTML에 색상 코드 포함 확인
        inv001_row = result[result["invoice_no"] == "INV001"].iloc[0]
        html = inv001_row["sku_summary_html"]

        # BA00021의 색상 (#4E79A7) 포함 확인
        assert "#4E79A7" in html or "#4e79a7" in html.lower()
        assert "<span" in html
        assert "BA00021: 1,000ea 외 2종" in html

    def test_sorting_by_onboard_date(self, sample_sku_color_map):
        """출발일 최신순 정렬 테스트"""
        today = pd.Timestamp.today().normalize()

        # 날짜가 다른 데이터
        data = pd.DataFrame(
            {
                "invoice_no": ["INV_OLD", "INV_NEW", "INV_MID"],
                "from_country": ["KR", "KR", "KR"],
                "to_country": ["US", "US", "US"],
                "to_center": ["AMZUS", "AMZUS", "AMZUS"],
                "resource_code": ["BA00001", "BA00002", "BA00003"],
                "resource_name": ["A", "B", "C"],
                "qty_ea": [100, 200, 300],
                "carrier_mode": ["특송", "특송", "특송"],
                "onboard_date": [
                    today - timedelta(days=10),  # 오래됨
                    today - timedelta(days=1),  # 최신
                    today - timedelta(days=5),  # 중간
                ],
                "pred_inbound_date": [
                    today + timedelta(days=5),
                    today + timedelta(days=5),
                    today + timedelta(days=5),
                ],
            }
        )

        result = build_inbound_table(data, sample_sku_color_map)

        # 최신순: INV_NEW → INV_MID → INV_OLD
        assert result.iloc[0]["invoice_no"] == "INV_NEW"
        assert result.iloc[1]["invoice_no"] == "INV_MID"
        assert result.iloc[2]["invoice_no"] == "INV_OLD"

    def test_missing_columns_handled(self, sample_sku_color_map):
        """필수 컬럼 누락 처리 테스트"""
        # 최소한의 컬럼만 포함
        minimal_data = pd.DataFrame(
            {
                "invoice_no": ["INV001"],
                "resource_code": ["BA00001"],
                "qty_ea": [100],
            }
        )

        # 에러 없이 처리되어야 함
        result = build_inbound_table(minimal_data, sample_sku_color_map)
        assert len(result) == 1

    def test_performance_10k_rows(self, sample_sku_color_map):
        """성능 테스트: 1만 행 ≤ 200ms"""
        import time

        today = pd.Timestamp.today().normalize()

        # 1만 행 생성 (1000개 인보이스 × 10개 SKU)
        rows = []
        for i in range(1000):
            for j in range(10):
                rows.append(
                    {
                        "invoice_no": f"INV{i:05d}",
                        "from_country": "KR",
                        "to_country": "US",
                        "to_center": "AMZUS",
                        "resource_code": f"BA{j:05d}",
                        "resource_name": f"제품{j}",
                        "qty_ea": 100 * (j + 1),
                        "carrier_mode": "특송",
                        "onboard_date": today - timedelta(days=i % 30),
                        "pred_inbound_date": today + timedelta(days=(i % 20)),
                    }
                )

        large_data = pd.DataFrame(rows)
        assert len(large_data) == 10000

        # 성능 측정
        start = time.perf_counter()
        result = build_inbound_table(large_data, sample_sku_color_map)
        elapsed_ms = (time.perf_counter() - start) * 1000

        # 검증
        assert len(result) == 1000  # 1000개 인보이스
        assert elapsed_ms < 200, f"Performance test failed: {elapsed_ms:.2f}ms > 200ms"

        print(f"✓ Performance test passed: {elapsed_ms:.2f}ms for 10,000 rows")

    def test_internal_columns_exist(self, sample_raw_data, sample_sku_color_map):
        """내부용 컬럼 존재 테스트"""
        result = build_inbound_table(sample_raw_data, sample_sku_color_map)

        # 내부용 컬럼 확인
        assert "_rep_sku" in result.columns
        assert "_to_center" in result.columns
        assert "_total_qty" in result.columns

    def test_total_qty_calculation(self, sample_raw_data, sample_sku_color_map):
        """총 수량 계산 테스트"""
        result = build_inbound_table(sample_raw_data, sample_sku_color_map)

        # INV001: 1000 + 500 + 300 = 1800
        inv001_row = result[result["invoice_no"] == "INV001"].iloc[0]
        assert inv001_row["_total_qty"] == 1800

        # INV002: 200
        inv002_row = result[result["invoice_no"] == "INV002"].iloc[0]
        assert inv002_row["_total_qty"] == 200


class TestEdgeCases:
    """엣지 케이스 테스트"""

    def test_null_dates_handled(self):
        """날짜 결측값 처리 테스트"""
        data = pd.DataFrame(
            {
                "invoice_no": ["INV001"],
                "from_country": ["KR"],
                "to_country": ["US"],
                "to_center": ["AMZUS"],
                "resource_code": ["BA00001"],
                "resource_name": ["제품A"],
                "qty_ea": [100],
                "carrier_mode": ["특송"],
                "onboard_date": [None],  # 결측
                "pred_inbound_date": [None],  # 결측
            }
        )

        sku_map = {"BA00001": "#4E79A7"}
        result = build_inbound_table(data, sku_map)

        assert len(result) == 1
        assert result.iloc[0]["onboard_date"] == ""  # 빈 문자열
        assert result.iloc[0]["eta_text"] == "미확인"
        assert result.iloc[0]["eta_color"] == "orange"

    def test_invalid_date_strings(self):
        """잘못된 날짜 문자열 처리 테스트"""
        data = pd.DataFrame(
            {
                "invoice_no": ["INV001"],
                "from_country": ["KR"],
                "to_country": ["US"],
                "to_center": ["AMZUS"],
                "resource_code": ["BA00001"],
                "resource_name": ["제품A"],
                "qty_ea": [100],
                "carrier_mode": ["특송"],
                "onboard_date": ["invalid_date"],  # 잘못된 날짜
                "pred_inbound_date": ["also_invalid"],  # 잘못된 날짜
            }
        )

        sku_map = {"BA00001": "#4E79A7"}
        result = build_inbound_table(data, sku_map)

        # 에러 없이 처리되어야 함
        assert len(result) == 1
        assert result.iloc[0]["eta_text"] == "미확인"

    def test_zero_quantity_handling(self):
        """수량 0 처리 테스트"""
        data = pd.DataFrame(
            {
                "invoice_no": ["INV001", "INV001"],
                "from_country": ["KR", "KR"],
                "to_country": ["US", "US"],
                "to_center": ["AMZUS", "AMZUS"],
                "resource_code": ["BA00001", "BA00002"],
                "resource_name": ["제품A", "제품B"],
                "qty_ea": [0, 0],  # 모두 0
                "carrier_mode": ["특송", "특송"],
                "onboard_date": ["2025-01-15", "2025-01-15"],
                "pred_inbound_date": ["2025-01-20", "2025-01-20"],
            }
        )

        sku_map = {"BA00001": "#4E79A7", "BA00002": "#F28E2B"}
        result = build_inbound_table(data, sku_map)

        # 수량 0이어도 처리되어야 함 (사전순으로 BA00001 선정)
        assert len(result) == 1
        assert result.iloc[0]["_rep_sku"] == "BA00001"

    def test_sku_not_in_color_map(self):
        """SKU 색상 맵에 없는 경우 테스트"""
        data = pd.DataFrame(
            {
                "invoice_no": ["INV001"],
                "from_country": ["KR"],
                "to_country": ["US"],
                "to_center": ["AMZUS"],
                "resource_code": ["BA99999"],  # 맵에 없음
                "resource_name": ["제품X"],
                "qty_ea": [100],
                "carrier_mode": ["특송"],
                "onboard_date": ["2025-01-15"],
                "pred_inbound_date": ["2025-01-20"],
            }
        )

        sku_map = {}  # 빈 맵
        result = build_inbound_table(data, sku_map)

        # 기본 회색 (#6b7280) 적용되어야 함
        html = result.iloc[0]["sku_summary_html"]
        assert "#6b7280" in html.lower()
