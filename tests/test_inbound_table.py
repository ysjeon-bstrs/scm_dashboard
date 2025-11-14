"""
입고 예정 테이블 모듈 테스트

테스트 범위:
- SKU별 개별 행 생성
- 제품명 + SKU 코드 HTML 생성 (제품명 검정, 코드만 색상)
- ETA 색상 규칙 적용
- 경로 생성
- 수량 포맷팅
- 경과일수 계산 (오늘 - 출발일)
- 평균 리드타임 조회 (leadtime_map 기반)
- 출발일 기준 정렬 (오래된 것부터)
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
                "from_center": ["태광KR", "태광KR", "태광KR", "AMZUS", "CN_WH"],
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

    def test_each_sku_is_separate_row(self, sample_raw_data, sample_sku_color_map):
        """각 SKU가 별도 행으로 생성되는지 테스트"""
        result = build_inbound_table(sample_raw_data, sample_sku_color_map)

        # 5개의 SKU → 5개의 행
        assert len(result) == 5

        # 주문번호 확인
        assert result["주문번호"].tolist().count("INV001") == 3
        assert result["주문번호"].tolist().count("INV002") == 1
        assert result["주문번호"].tolist().count("INV003") == 1

    def test_columns_exist(self, sample_raw_data, sample_sku_color_map):
        """필수 컬럼 존재 테스트"""
        result = build_inbound_table(sample_raw_data, sample_sku_color_map)

        expected_cols = [
            "주문번호",
            "경로",
            "제품(SKU)",
            "수량",
            "운송모드",
            "출발일",
            "예상 도착일",
            "경과일수",
            "평균 리드타임(일)",
            "eta_color",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_product_html_format_with_name(self, sample_sku_color_map):
        """제품명이 있을 때 HTML 포맷 테스트"""
        data = pd.DataFrame(
            {
                "invoice_no": ["INV001"],
                "from_country": ["KR"],
                "to_country": ["US"],
                "to_center": ["AMZUS"],
                "resource_code": ["BA00021"],
                "resource_name": ["비타민세럼"],
                "qty_ea": [1000],
                "carrier_mode": ["특송"],
                "onboard_date": ["2025-01-15"],
                "pred_inbound_date": ["2025-01-20"],
            }
        )

        result = build_inbound_table(data, sample_sku_color_map)
        product_html = result.iloc[0]["제품(SKU)"]

        # "비타민세럼 (<span style='color:#4E79A7'>BA00021</span>)" 형태
        assert "비타민세럼" in product_html
        assert "BA00021" in product_html
        assert "<span" in product_html
        assert "style=" in product_html
        assert "#4E79A7" in product_html or "#4e79a7" in product_html.lower()

    def test_product_html_format_without_name(self, sample_sku_color_map):
        """제품명이 없을 때 HTML 포맷 테스트"""
        data = pd.DataFrame(
            {
                "invoice_no": ["INV001"],
                "from_country": ["KR"],
                "to_country": ["US"],
                "to_center": ["AMZUS"],
                "resource_code": ["BA00021"],
                "resource_name": [""],  # 빈 이름
                "qty_ea": [1000],
                "carrier_mode": ["특송"],
                "onboard_date": ["2025-01-15"],
                "pred_inbound_date": ["2025-01-20"],
            }
        )

        result = build_inbound_table(data, sample_sku_color_map)
        product_html = result.iloc[0]["제품(SKU)"]

        # "<span style='color:#4E79A7'>BA00021</span>" 형태
        assert "BA00021" in product_html
        assert "<span" in product_html
        assert "#4E79A7" in product_html or "#4e79a7" in product_html.lower()

    def test_quantity_is_integer(self, sample_raw_data, sample_sku_color_map):
        """수량이 정수형인지 테스트"""
        result = build_inbound_table(sample_raw_data, sample_sku_color_map)

        for qty in result["수량"]:
            assert isinstance(qty, int) or isinstance(qty, pd.Int64Dtype)

    def test_route_format_with_center(self, sample_raw_data, sample_sku_color_map):
        """센터가 있을 때 경로 포맷 테스트"""
        result = build_inbound_table(sample_raw_data, sample_sku_color_map)

        # "KR → US (AMZUS)" 형태
        inv001_rows = result[result["주문번호"] == "INV001"]
        assert any("KR → US" in route and "AMZUS" in route for route in inv001_rows["경로"])

    def test_eta_color_red_past(self, sample_raw_data, sample_sku_color_map):
        """ETA 색상 규칙 - 과거 (빨강) 테스트"""
        result = build_inbound_table(sample_raw_data, sample_sku_color_map)

        # INV001 행들: 과거 날짜 → red
        inv001_rows = result[result["주문번호"] == "INV001"]
        assert all(row["eta_color"] == "red" for _, row in inv001_rows.iterrows())

    def test_eta_color_green_within_5days(self, sample_raw_data, sample_sku_color_map):
        """ETA 색상 규칙 - 5일 이내 (초록) 테스트"""
        result = build_inbound_table(sample_raw_data, sample_sku_color_map)

        # INV002: 3일 후 → green
        inv002_row = result[result["주문번호"] == "INV002"].iloc[0]
        assert inv002_row["eta_color"] == "green"

    def test_eta_color_orange_undefined(self, sample_raw_data, sample_sku_color_map):
        """ETA 색상 규칙 - 미확인 (주황) 테스트"""
        result = build_inbound_table(sample_raw_data, sample_sku_color_map)

        # INV003: not_defined → orange
        inv003_row = result[result["주문번호"] == "INV003"].iloc[0]
        assert inv003_row["eta_color"] == "orange"
        assert inv003_row["예상 도착일"] == "미확인"

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

    def test_onboard_date_format(self, sample_raw_data, sample_sku_color_map):
        """출발일 포맷 테스트"""
        result = build_inbound_table(sample_raw_data, sample_sku_color_map)

        # 모든 행에 YYYY-MM-DD 포맷 확인
        for _, row in result.iterrows():
            onboard = row["출발일"]
            if onboard:  # 빈 문자열이 아니면
                # YYYY-MM-DD 형식 검증
                try:
                    datetime.strptime(onboard, "%Y-%m-%d")
                    valid = True
                except ValueError:
                    valid = False
                assert valid, f"Invalid date format: {onboard}"

    def test_sorting_by_onboard_date_oldest_first(self, sample_sku_color_map):
        """출발일 오름차순 정렬 테스트 (오래된 것부터)"""
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
                    today - timedelta(days=10),  # 가장 오래됨
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

        # 오름차순 (오래된 것부터): INV_OLD → INV_MID → INV_NEW
        assert result.iloc[0]["주문번호"] == "INV_OLD"
        assert result.iloc[1]["주문번호"] == "INV_MID"
        assert result.iloc[2]["주문번호"] == "INV_NEW"

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

    def test_default_sku_color(self, sample_raw_data):
        """SKU 색상 맵에 없는 경우 기본 색상 테스트"""
        # 빈 색상 맵
        result = build_inbound_table(sample_raw_data, {})

        # 기본 빨강 (#b91c1c) 적용되어야 함
        product_html = result.iloc[0]["제품(SKU)"]
        assert "#b91c1c" in product_html.lower()

    def test_elapsed_days_calculation(self, sample_sku_color_map):
        """경과일수 계산 테스트"""
        today = pd.Timestamp.today().normalize()

        data = pd.DataFrame(
            {
                "invoice_no": ["INV001", "INV002", "INV003"],
                "from_country": ["KR", "KR", "KR"],
                "to_country": ["US", "US", "US"],
                "to_center": ["AMZUS", "AMZUS", "AMZUS"],
                "resource_code": ["BA00001", "BA00002", "BA00003"],
                "resource_name": ["제품A", "제품B", "제품C"],
                "qty_ea": [100, 200, 300],
                "carrier_mode": ["특송", "특송", "특송"],
                "onboard_date": [
                    today - timedelta(days=10),  # 10일 전
                    today - timedelta(days=5),  # 5일 전
                    today,  # 오늘
                ],
                "pred_inbound_date": [
                    today + timedelta(days=5),
                    today + timedelta(days=5),
                    today + timedelta(days=5),
                ],
            }
        )

        result = build_inbound_table(data, sample_sku_color_map)

        # 경과일수 확인
        assert result.iloc[0]["경과일수"] == 10
        assert result.iloc[1]["경과일수"] == 5
        assert result.iloc[2]["경과일수"] == 0

    def test_elapsed_days_with_null_onboard(self, sample_sku_color_map):
        """출발일이 없을 때 경과일수 테스트"""
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
                "pred_inbound_date": ["2025-01-20"],
            }
        )

        result = build_inbound_table(data, sample_sku_color_map)
        assert result.iloc[0]["경과일수"] == "-"

    def test_average_leadtime_with_map(self, sample_sku_color_map):
        """leadtime_map이 제공되었을 때 평균 리드타임 조회 테스트"""
        data = pd.DataFrame(
            {
                "invoice_no": ["INV001"],
                "from_country": ["KR"],
                "from_center": ["태광KR"],
                "to_country": ["US"],
                "to_center": ["AMZUS"],
                "resource_code": ["BA00001"],
                "resource_name": ["제품A"],
                "qty_ea": [1000],
                "carrier_mode": ["특송"],
                "onboard_date": ["2025-01-15"],
                "pred_inbound_date": ["2025-01-20"],
            }
        )

        leadtime_map = {
            ("태광KR", "AMZUS", "특송"): 7.2,
        }

        result = build_inbound_table(data, sample_sku_color_map, leadtime_map)
        assert result.iloc[0]["평균 리드타임(일)"] == "7.2"

    def test_average_leadtime_without_map(self, sample_sku_color_map):
        """leadtime_map이 없을 때 평균 리드타임 테스트"""
        data = pd.DataFrame(
            {
                "invoice_no": ["INV001"],
                "from_country": ["KR"],
                "to_country": ["US"],
                "to_center": ["AMZUS"],
                "resource_code": ["BA00001"],
                "resource_name": ["제품A"],
                "qty_ea": [1000],
                "carrier_mode": ["특송"],
                "onboard_date": ["2025-01-15"],
                "pred_inbound_date": ["2025-01-20"],
            }
        )

        result = build_inbound_table(data, sample_sku_color_map, None)
        assert result.iloc[0]["평균 리드타임(일)"] == "-"

    def test_average_leadtime_no_match(self, sample_sku_color_map):
        """leadtime_map에 매칭되는 경로가 없을 때 테스트"""
        data = pd.DataFrame(
            {
                "invoice_no": ["INV001"],
                "from_country": ["KR"],
                "from_center": ["태광KR"],
                "to_country": ["US"],
                "to_center": ["AMZUS"],
                "resource_code": ["BA00001"],
                "resource_name": ["제품A"],
                "qty_ea": [1000],
                "carrier_mode": ["특송"],
                "onboard_date": ["2025-01-15"],
                "pred_inbound_date": ["2025-01-20"],
            }
        )

        # 다른 경로만 있는 leadtime_map
        leadtime_map = {
            ("CN", "AMZUS", "해운"): 25.5,
        }

        result = build_inbound_table(data, sample_sku_color_map, leadtime_map)
        assert result.iloc[0]["평균 리드타임(일)"] == "-"

    def test_performance_10k_rows(self, sample_sku_color_map):
        """성능 테스트: 1만 행 ≤ 500ms (groupby 제거로 더 빠를 것으로 예상)"""
        import time

        today = pd.Timestamp.today().normalize()

        # 1만 행 생성
        rows = []
        for i in range(10000):
            rows.append(
                {
                    "invoice_no": f"INV{i % 100:05d}",  # 100개의 서로 다른 인보이스
                    "from_country": "KR",
                    "to_country": "US",
                    "to_center": "AMZUS",
                    "resource_code": f"BA{i % 50:05d}",  # 50개의 서로 다른 SKU
                    "resource_name": f"제품{i % 50}",
                    "qty_ea": 100 * ((i % 10) + 1),
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
        assert len(result) == 10000  # 각 SKU가 별도 행
        assert elapsed_ms < 500, f"Performance test failed: {elapsed_ms:.2f}ms > 500ms"

        print(f"✓ Performance test passed: {elapsed_ms:.2f}ms for 10,000 rows")


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
        assert result.iloc[0]["출발일"] == ""  # 빈 문자열
        assert result.iloc[0]["예상 도착일"] == "미확인"
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
        assert result.iloc[0]["예상 도착일"] == "미확인"

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

        # 수량 0이어도 두 행 생성
        assert len(result) == 2
        assert result.iloc[0]["수량"] == 0
        assert result.iloc[1]["수량"] == 0

    def test_none_sku_color_map(self):
        """sku_color_map이 None일 때 테스트"""
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
                "onboard_date": ["2025-01-15"],
                "pred_inbound_date": ["2025-01-20"],
            }
        )

        # None 전달
        result = build_inbound_table(data, None)

        # 기본 색상 적용되어야 함
        assert len(result) == 1
        product_html = result.iloc[0]["제품(SKU)"]
        assert "<span" in product_html
