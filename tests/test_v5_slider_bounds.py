import pandas as pd

from v5_main import _calculate_date_bounds


def _empty_snapshot() -> pd.DataFrame:
    return pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]")})


def test_calculate_date_bounds_defaults_to_base_window():
    today = pd.Timestamp("2024-04-10")
    snapshot = _empty_snapshot()
    moves = pd.DataFrame()

    bound_min, bound_max = _calculate_date_bounds(
        today=today,
        snapshot_df=snapshot,
        moves_df=moves,
        base_past_days=42,
        base_future_days=42,
    )

    assert bound_min == pd.Timestamp("2024-02-28")
    assert bound_max == pd.Timestamp("2024-05-22")


def test_calculate_date_bounds_clamps_snapshot_history_to_base_window():
    today = pd.Timestamp("2024-04-10")
    history_start = today - pd.Timedelta(days=60)
    snapshot = pd.DataFrame({"date": pd.date_range(history_start, periods=5, freq="7D")})
    moves = pd.DataFrame()

    bound_min, bound_max = _calculate_date_bounds(
        today=today,
        snapshot_df=snapshot,
        moves_df=moves,
        base_past_days=42,
        base_future_days=42,
    )

    assert bound_min == pd.Timestamp("2024-02-28")
    assert bound_max == pd.Timestamp("2024-05-22")


def test_calculate_date_bounds_clamps_future_moves_to_base_window():
    today = pd.Timestamp("2024-04-10")
    snapshot = _empty_snapshot()
    future_date = today + pd.Timedelta(days=90)
    moves = pd.DataFrame({"arrival_date": [future_date]})

    bound_min, bound_max = _calculate_date_bounds(
        today=today,
        snapshot_df=snapshot,
        moves_df=moves,
        base_past_days=42,
        base_future_days=42,
    )

    assert bound_min == pd.Timestamp("2024-02-28")
    assert bound_max == pd.Timestamp("2024-05-22")
