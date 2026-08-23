"""Tests for totals fallback candidate evaluation."""

import numpy as np
import pandas as pd
import pytest

from src.modeling.evaluate_totals_fallback_candidates import (
    FALLBACK_CORE_FEATURES,
    RESULT_COLUMNS,
    create_totals_fallback_features,
    evaluate_totals_fallback_candidates,
    prepare_common_fallback_sample,
)
from src.modeling.backtest_totals_fallback_candidates import (
    FOLD_RESULT_COLUMNS,
    SUMMARY_RESULT_COLUMNS,
    create_backtest_candidates,
    evaluate_totals_fallback_expanding_window,
)


def create_development_data() -> pd.DataFrame:
    """Create synthetic fallback development games."""

    rows: list[dict[str, object]] = []

    for index in range(80):
        split_name = (
            "train"
            if index < 60
            else "validation"
        )

        home_elo = float(
            1400 + index * 3
        )

        away_elo = float(
            1550 - index
        )

        is_indoor = (
            index % 4 == 0
        )

        league_average = (
            44.0 + index * 0.04
        )

        target_total = (
            10.0
            + 0.010
            * (
                home_elo
                + away_elo
            )
            + 0.75 * league_average
            + (
                1.5
                if is_indoor
                else 0.0
            )
        )

        rows.append(
            {
                "game_id": f"game_{index}",
                "season": (
                    2022
                    if split_name == "train"
                    else 2023
                ),
                "split_name": split_name,
                "target_total_points": target_total,
                "home_elo_rating": home_elo,
                "away_elo_rating": away_elo,
                "is_indoor": is_indoor,
                "league_average_total_last_64": (
                    league_average
                ),
            }
        )

    incomplete = {
        **rows[-1],
        "game_id": "incomplete_game",
        "league_average_total_last_64": np.nan,
    }

    holdout = {
        **rows[-1],
        "game_id": "holdout_game",
        "split_name": "holdout",
    }

    rows.extend(
        [
            incomplete,
            holdout,
        ]
    )

    return pd.DataFrame(rows)


def test_create_elo_sum_feature() -> None:
    """Create a team-order-invariant Elo sum."""

    features = create_totals_fallback_features(
        create_development_data().iloc[[0]]
    )

    assert features.iloc[0][
        "elo_rating_sum"
    ] == pytest.approx(2950.0)


def test_prepare_common_sample() -> None:
    """Keep complete train and validation games."""

    sample = prepare_common_fallback_sample(
        create_development_data()
    )

    assert len(sample) == 80

    assert set(
        sample["split_name"]
    ) == {
        "train",
        "validation",
    }

    assert sample[
        list(FALLBACK_CORE_FEATURES)
    ].notna().all().all()


def test_evaluate_expected_candidates() -> None:
    """Return every fallback candidate and baseline."""

    results = evaluate_totals_fallback_candidates(
        create_development_data()
    )

    assert tuple(
        results.columns
    ) == RESULT_COLUMNS

    assert set(
        results["candidate_name"]
    ) == {
        "constant_train_mean",
        "ridge_league_64",
        "ridge_league_64_indoor",
        "ridge_league_64_elo",
        "ridge_league_64_indoor_elo",
    }

    assert set(
        results["train_game_count"]
    ) == {
        60,
    }

    assert set(
        results["validation_game_count"]
    ) == {
        20,
    }


def test_signal_model_beats_constant() -> None:
    """Recover the synthetic fallback signal."""

    results = evaluate_totals_fallback_candidates(
        create_development_data()
    ).set_index(
        "candidate_name"
    )

    assert (
        results.loc[
            "ridge_league_64_indoor_elo",
            "validation_mae",
        ]
        < results.loc[
            "constant_train_mean",
            "validation_mae",
        ]
    )


def test_team_order_does_not_change_elo_sum(
) -> None:
    """Preserve the Elo aggregate when teams swap."""

    original = create_development_data().iloc[
        [0]
    ].copy()

    swapped = original.copy()

    swapped.loc[
        swapped.index[0],
        "home_elo_rating",
    ] = original.iloc[0]["away_elo_rating"]

    swapped.loc[
        swapped.index[0],
        "away_elo_rating",
    ] = original.iloc[0]["home_elo_rating"]

    original_features = (
        create_totals_fallback_features(
            original
        )
    )

    swapped_features = (
        create_totals_fallback_features(
            swapped
        )
    )

    assert original_features.iloc[0][
        "elo_rating_sum"
    ] == pytest.approx(
        swapped_features.iloc[0][
            "elo_rating_sum"
        ]
    )


def test_missing_source_column_is_rejected() -> None:
    """Reject an incomplete fallback schema."""

    data = create_development_data().drop(
        columns=[
            "home_elo_rating",
        ]
    )

    with pytest.raises(
        ValueError,
        match="missing columns",
    ):
        create_totals_fallback_features(
            data
        )


def create_backtest_data() -> pd.DataFrame:
    """Create chronological fallback games."""

    rows: list[dict[str, object]] = []

    for season in range(2018, 2025):
        for game_index in range(12):
            home_elo = float(
                1400 + game_index * 15
            )

            away_elo = float(
                1580 - game_index * 8
            )

            is_indoor = (
                game_index % 4 == 0
            )

            league_average = (
                44.0
                + 0.35 * (season - 2018)
            )

            target_total = (
                10.0
                + 0.010
                * (
                    home_elo
                    + away_elo
                )
                + 0.75 * league_average
                + (
                    1.5
                    if is_indoor
                    else 0.0
                )
            )

            rows.append(
                {
                    "game_id": (
                        f"{season}_{game_index}"
                    ),
                    "season": season,
                    "split_name": (
                        "train"
                        if season <= 2022
                        else "validation"
                    ),
                    "target_total_points": (
                        target_total
                    ),
                    "home_elo_rating": home_elo,
                    "away_elo_rating": away_elo,
                    "is_indoor": is_indoor,
                    "league_average_total_last_64": (
                        league_average
                    ),
                }
            )

    return pd.DataFrame(rows)


def test_backtest_candidate_grid() -> None:
    """Create every fallback and alpha setting."""

    candidates = create_backtest_candidates(
        alpha_grid=(
            0.0,
            1.0,
        )
    )

    assert len(candidates) == 4

    assert {
        candidate.candidate_name
        for candidate in candidates
    } == {
        "ridge_league_64_elo",
        "ridge_league_64_indoor_elo",
    }


def test_backtest_uses_only_earlier_seasons(
) -> None:
    """Prevent future seasons from training."""

    _, folds = (
        evaluate_totals_fallback_expanding_window(
            development_data=create_backtest_data(),
            validation_seasons=(
                2021,
                2022,
                2023,
                2024,
            ),
            alpha_grid=(
                1.0,
            ),
        )
    )

    assert tuple(
        folds.columns
    ) == FOLD_RESULT_COLUMNS

    assert (
        folds["train_last_season"]
        < folds["validation_season"]
    ).all()


def test_backtest_returns_expected_rows() -> None:
    """Return one summary row per setting."""

    summary, folds = (
        evaluate_totals_fallback_expanding_window(
            development_data=create_backtest_data(),
            validation_seasons=(
                2022,
                2023,
                2024,
            ),
            alpha_grid=(
                0.0,
                1.0,
            ),
        )
    )

    assert tuple(
        summary.columns
    ) == SUMMARY_RESULT_COLUMNS

    assert len(summary) == 5
    assert len(folds) == 15

    assert set(
        summary["validation_game_count"]
    ) == {
        36,
    }


def test_backtest_signal_beats_constant() -> None:
    """Recover the chronological fallback signal."""

    summary, _ = (
        evaluate_totals_fallback_expanding_window(
            development_data=create_backtest_data(),
            validation_seasons=(
                2021,
                2022,
                2023,
                2024,
            ),
            alpha_grid=(
                0.0,
                1.0,
            ),
        )
    )

    constant_mae = summary.loc[
        summary["candidate_name"]
        == "constant_train_mean",
        "pooled_validation_mae",
    ].iloc[0]

    best_model_mae = summary.loc[
        summary["candidate_name"]
        != "constant_train_mean",
        "pooled_validation_mae",
    ].min()

    assert best_model_mae < constant_mae


def test_backtest_rejects_invalid_alpha() -> None:
    """Reject duplicate and negative alpha grids."""

    with pytest.raises(
        ValueError,
        match="unique",
    ):
        create_backtest_candidates(
            alpha_grid=(
                1.0,
                1.0,
            )
        )

    with pytest.raises(
        ValueError,
        match="negative",
    ):
        create_backtest_candidates(
            alpha_grid=(
                -1.0,
            )
        )