"""리드타임 맵 및 예상 입고일 계산 테스트"""

from datetime import datetime

import pandas as pd

from scm_dashboard_v9.domain.inbound import (
    assign_expected_inbound_dates,
    build_lt_inbound_map,
)


def test_build_map_and_assign_expected_inbound_dates():
    leadtime_df = pd.DataFrame(
        {
            "from_center": ["태광KR", "태광KR", "CJ서부US"],
            "to_center": ["AMZUS", "SBSBPH", "SBSMY"],
            "carrier_mode": ["특송", "택배", "SEND"],
            "avg_lt_depart_to_inbound": [7.5, 5, 12],
        }
    )

    leadtime_map = build_lt_inbound_map(leadtime_df)

    assert leadtime_map[("태광KR", "AMZUS", "특송")] == 7.5
    assert leadtime_map[("CJ서부US", "SBSMY", "SEND")] == 12.0

    moves = pd.DataFrame(
        {
            "from_center": ["태광KR", "태광KR", "CJ서부US", "태광KR"],
            "to_center": ["AMZUS", "SBSBPH", "SBSMY", "AMZUS"],
            "carrier_mode": ["특송", "택배", "SEND", "그레이"],
            "onboard_date": [
                datetime(2025, 1, 3),
                datetime(2025, 1, 10),
                datetime(2025, 2, 1),
                datetime(2025, 3, 5),
            ],
        }
    )

    enriched = assign_expected_inbound_dates(moves, leadtime_map)

    expected_dates = enriched["expected_inbound_date"].dt.strftime("%Y-%m-%d").tolist()

    assert expected_dates[0] == "2025-01-10"
    assert expected_dates[1] == "2025-01-15"
    assert expected_dates[2] == "2025-02-13"
    assert pd.isna(enriched.loc[3, "expected_inbound_date"])
