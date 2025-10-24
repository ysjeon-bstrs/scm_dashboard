import pandas as pd

from center_alias import normalize_center_series, normalize_center_value


def test_normalize_center_value_handles_amazon_variants() -> None:
    variants = ["Amazon US", "AmazonUS", "AMZ US", "AMZ-US", "AMZ_US", "amazonus", "아마존US"]
    for value in variants:
        assert normalize_center_value(value) == "AMZUS"


def test_normalize_center_value_handles_across_b_variants() -> None:
    variants = ["AcrossBUS", "Across B US", "Across-B US", "Across_B US", "어크로스비US", "acrossbus"]
    for value in variants:
        assert normalize_center_value(value) == "AcrossBUS"


def test_normalize_center_series_applies_alias_to_series() -> None:
    series = pd.Series(["Amazon US", "AMZ-US", "Across B US", "태광KR", pd.NA])
    normalized = normalize_center_series(series)
    assert normalized.tolist()[:3] == ["AMZUS", "AMZUS", "AcrossBUS"]
    assert normalized.iloc[3] == "태광KR"
    assert normalized.iloc[4] == ""


def test_normalize_center_value_returns_none_for_ignored_tokens() -> None:
    assert normalize_center_value("In-Transit") is None
    assert normalize_center_value("WIP") is None
