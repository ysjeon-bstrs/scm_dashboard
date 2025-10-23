import pandas as pd

from scm_dashboard_v9.domain.normalization import normalize_moves


def test_normalize_moves_accepts_column_aliases():
    raw = pd.DataFrame(
        {
            "RESOURCE_CODE": ["BA00021"],
            "Quantity": ["1,200"],
            "TO_CENTER": ["AMZ US"],
            "FROM CENTER": ["태광KR"],
            "Carrier Mode": ["sea"],
            "depart_date": ["2024-05-01"],
            "ETA": ["2024-05-10"],
            "입고일": ["2024-05-12"],
        }
    )

    normalized = normalize_moves(raw)

    assert normalized.loc[0, "resource_code"] == "BA00021"
    assert normalized.loc[0, "qty_ea"] == 1200
    assert normalized.loc[0, "carrier_mode"] == "SEA"
    assert normalized.loc[0, "from_center"] == "태광KR"
    assert normalized.loc[0, "to_center"] == "AMZUS"
    assert normalized.loc[0, "onboard_date"] == pd.Timestamp("2024-05-01")
    assert normalized.loc[0, "arrival_date"] == pd.Timestamp("2024-05-10")
    assert normalized.loc[0, "inbound_date"] == pd.Timestamp("2024-05-12")
