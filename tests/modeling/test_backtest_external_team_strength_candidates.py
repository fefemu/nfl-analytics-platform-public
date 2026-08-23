import pandas as pd

from src.modeling.backtest_external_team_strength_candidates import (
    CANDIDATE_FEATURES,
    SUMMARY_COLUMNS,
    backtest_team_strength_candidates,
)


def create_source_data() -> pd.DataFrame:
    rows = []
    game_number = 0
    for season in range(2018, 2025):
        for index in range(40):
            game_number += 1
            strength = (index - 19.5) / 4.0
            rows.append(
                {
                    "game_id": f"{season}_{game_number}",
                    "season": season,
                    "game_date": pd.Timestamp(season, 9, 1)
                    + pd.Timedelta(days=index),
                    "target_home_win": int(strength + (index % 3) > 0),
                    "external_nfelo_rating_difference": strength,
                    "external_nfelo_qb_adjustment_difference": index % 4,
                    "unit_offense_rating_difference": strength * 0.4,
                    "unit_defense_rating_difference": strength * 0.2,
                    "win_total_elo_difference": strength * 10.0,
                }
            )
    return pd.DataFrame(rows)


def test_backtest_returns_every_candidate() -> None:
    summary, seasons = backtest_team_strength_candidates(
        create_source_data()
    )

    assert tuple(summary.columns) == SUMMARY_COLUMNS
    assert set(summary["candidate_name"]) == set(CANDIDATE_FEATURES)
    assert set(seasons["validation_season"]) == {2021, 2022, 2023, 2024}


def test_backtest_never_uses_validation_season_for_training() -> None:
    _, seasons = backtest_team_strength_candidates(create_source_data())

    first_fold = seasons.loc[seasons["validation_season"] == 2021]
    assert set(first_fold["training_game_count"]) == {120}
    assert set(first_fold["validation_game_count"]) == {40}


def test_summary_is_sorted_by_brier_score() -> None:
    summary, _ = backtest_team_strength_candidates(create_source_data())

    assert summary["brier_score"].is_monotonic_increasing
