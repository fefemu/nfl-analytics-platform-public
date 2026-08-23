import pandas as pd

from src.modeling.backtest_external_units_totals_candidates import (
    CANDIDATE_FEATURES,
    SUMMARY_COLUMNS,
    backtest_unit_totals_candidates,
)


def create_totals_data() -> pd.DataFrame:
    rows = []
    all_features = {feature for values in CANDIDATE_FEATURES.values() for feature in values}
    for season in range(2018, 2025):
        for index in range(24):
            row = {
                "game_id": f"{season}_{index}",
                "season": season,
                "target_total_points": 40.0 + index,
            }
            row.update({feature: float((index % 7) + 1) for feature in all_features})
            rows.append(row)
    return pd.DataFrame(rows)


def test_totals_backtest_returns_every_candidate() -> None:
    summary, predictions = backtest_unit_totals_candidates(create_totals_data())

    assert tuple(summary.columns) == SUMMARY_COLUMNS
    assert set(summary["candidate_name"]) == set(CANDIDATE_FEATURES)
    assert len(predictions) == 4 * 24 * len(CANDIDATE_FEATURES)


def test_totals_backtest_uses_chronological_validation() -> None:
    _, predictions = backtest_unit_totals_candidates(create_totals_data())

    assert set(predictions["validation_season"]) == {2021, 2022, 2023, 2024}
    assert predictions.groupby(["candidate_name", "game_id"]).size().max() == 1
