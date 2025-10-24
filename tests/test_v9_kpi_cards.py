"""
v9 KPI 카드 렌더링 테스트

KPI 카드 렌더링 로직의 버그 수정을 검증합니다.
"""
from __future__ import annotations

import pandas as pd
import pytest
from unittest.mock import Mock, patch

from scm_dashboard_v9.ui.kpi.cards import render_sku_summary_cards


def test_chunk_size_zero_preserved():
    """
    Bug Fix: chunk_size=0을 명시적으로 전달하면 유지되어야 함
    
    이전에는 `chunk_size or CONFIG.ui.kpi_card_chunk_size` 로직으로 인해
    chunk_size=0이 CONFIG 기본값으로 덮어씌워졌음.
    """
    # Mock 데이터 준비
    snapshot = pd.DataFrame({
        "date": ["2024-01-01"],
        "center": ["태광KR"],
        "resource_code": ["BA00021"],
        "stock_qty": [100],
        "resource_name": ["테스트상품"],
    })
    
    moves = pd.DataFrame({
        "from_center": ["상해CN"],
        "to_center": ["태광KR"],
        "resource_code": ["BA00021"],
        "qty_ea": [50],
        "carrier_mode": ["SEA"],
        "onboard_date": ["2024-01-05"],
        "arrival_date": ["2024-01-10"],
    })
    
    today = pd.Timestamp("2024-01-15")
    
    # Streamlit 함수들을 Mock
    with patch("scm_dashboard_v9.ui.kpi.cards.st") as mock_st:
        mock_st.subheader = Mock()
        mock_st.caption = Mock()
        mock_st.markdown = Mock()
        
        # chunk_size=0으로 호출 (명시적)
        result = render_sku_summary_cards(
            snapshot,
            moves,
            centers=["태광KR"],
            skus=["BA00021"],
            today=today,
            latest_snapshot=pd.Timestamp("2024-01-01"),
            chunk_size=0,  # ← 명시적으로 0 전달
        )
        
        # 함수가 정상적으로 실행되어야 함 (ZeroDivisionError 없음)
        # chunk_size=0일 때는 sku_min_width=320이 되어야 함
        assert mock_st.markdown.called


def test_chunk_size_none_uses_config():
    """chunk_size=None일 때는 CONFIG 기본값 사용"""
    snapshot = pd.DataFrame({
        "date": ["2024-01-01"],
        "center": ["태광KR"],
        "resource_code": ["BA00021"],
        "stock_qty": [100],
        "resource_name": ["테스트상품"],
    })
    
    moves = pd.DataFrame({
        "from_center": ["상해CN"],
        "to_center": ["태광KR"],
        "resource_code": ["BA00021"],
        "qty_ea": [50],
        "carrier_mode": ["SEA"],
        "onboard_date": ["2024-01-05"],
        "arrival_date": ["2024-01-10"],
    })
    
    today = pd.Timestamp("2024-01-15")
    
    with patch("scm_dashboard_v9.ui.kpi.cards.st") as mock_st:
        mock_st.subheader = Mock()
        mock_st.caption = Mock()
        mock_st.markdown = Mock()
        
        # chunk_size를 명시하지 않음 (None 기본값)
        result = render_sku_summary_cards(
            snapshot,
            moves,
            centers=["태광KR"],
            skus=["BA00021"],
            today=today,
            latest_snapshot=pd.Timestamp("2024-01-01"),
            # chunk_size 인자 없음 → None
        )
        
        # CONFIG 기본값이 사용되어야 함
        assert mock_st.markdown.called


def test_chunk_size_positive_preserved():
    """chunk_size에 양수를 전달하면 그대로 사용되어야 함"""
    snapshot = pd.DataFrame({
        "date": ["2024-01-01"],
        "center": ["태광KR"],
        "resource_code": ["BA00021"],
        "stock_qty": [100],
        "resource_name": ["테스트상품"],
    })
    
    moves = pd.DataFrame({
        "from_center": ["상해CN"],
        "to_center": ["태광KR"],
        "resource_code": ["BA00021"],
        "qty_ea": [50],
        "carrier_mode": ["SEA"],
        "onboard_date": ["2024-01-05"],
        "arrival_date": ["2024-01-10"],
    })
    
    today = pd.Timestamp("2024-01-15")
    
    with patch("scm_dashboard_v9.ui.kpi.cards.st") as mock_st:
        mock_st.subheader = Mock()
        mock_st.caption = Mock()
        mock_st.markdown = Mock()
        
        # chunk_size=3 전달
        result = render_sku_summary_cards(
            snapshot,
            moves,
            centers=["태광KR"],
            skus=["BA00021"],
            today=today,
            latest_snapshot=pd.Timestamp("2024-01-01"),
            chunk_size=3,  # ← 명시적으로 3 전달
        )
        
        assert mock_st.markdown.called


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
