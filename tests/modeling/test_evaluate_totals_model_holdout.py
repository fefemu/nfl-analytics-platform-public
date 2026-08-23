"""Tests for final totals holdout evaluation."""

import numpy as np
import pandas as pd
import pytest

from src.modeling.evaluate_totals_model_holdout import (
    FINAL_RIDGE_ALPHA,
    LOCKED_TOTALS_FEATURES,
    RESULT_COLUMNS,
    evaluate_locked_totals_holdout,
    prepare_totals_holdout_sample,
)


def create_holdout_data() -> pd.DataFrame:
    """Create development and holdout totals games."""

    rows: list[dict[str, object]] = []

    for season in range(2018, 2026):
        for game_index in range(20):
            feature_values = {
                feature_name: float(
                    1
                    + feature_index
                    + game_index % 5
                )
                for feature_index, feature_name
                in enumerate(
                    LOCKED_TOTALS_FEATURES
                )
            }

            target_total = (
                25.0
                + 0.5
                * feature_values[
                    "offensive_epa_sum_last_4"
                ]
                + 0.8
                * feature_values[
                    (
                        "defensive_epa_allowed_"
                        "sum_last_4"
                    )
                ]
                + 0.4
                * feature_values[
                    "listed_qb_rating_sum"
                ]
                + 0.7
                * feature_values[
                    "league_average_total_last_64"
                ]
                + 0.1 * (season - 2018)
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
                        else (
                            "validation"
                            if season <= 2024
                            else "holdout"
                        )
                    ),
                    "both_short_windows_complete": True,
                    "target_total_points": (
                        target_total
                    ),
                    **feature_values,
                }
            )

    return pd.DataFrame(rows)


def test_prepare_separates_development_and_holdout(
) -> None:
    """Keep holdout outside model training."""

    development, holdout = (
        prepare_totals_holdout_sample(
            create_holdout_data()
        )
    )

    assert set(
        development["split_name"]
    ) == {
        "train",
        "validation",
    }

    assert set(
        holdout["split_name"]
    ) == {
        "holdout",
    }

    assert development["season"].max() == 2024
    assert holdout["season"].min() == 2025


def test_locked_model_beats_constant() -> None:
    """Recover the synthetic totals signal."""

    results = evaluate_locked_totals_holdout(
        create_holdout_data()
    ).set_index(
        "candidate_name"
    )

    model_name = (
        "ridge_epa_weather_qb_league_64_locked"
    )

    assert (
        results.loc[
            model_name,
            "holdout_mae",
        ]
        < results.loc[
            "constant_development_mean",
            "holdout_mae",
        ]
    )

    assert (
        results.loc[
            model_name,
            "mae_improvement_percent",
        ]
        > 0.0
    )


def test_holdout_result_schema_and_counts() -> None:
    """Return the locked two-row comparison."""

    results = evaluate_locked_totals_holdout(
        create_holdout_data()
    )

    assert tuple(
        results.columns
    ) == RESULT_COLUMNS

    assert len(results) == 2

    assert set(
        results["training_game_count"]
    ) == {
        140,
    }

    assert set(
        results["holdout_game_count"]
    ) == {
        20,
    }


def test_incomplete_games_are_removed() -> None:
    """Require complete target and locked features."""

    data = create_holdout_data()

    incomplete_development = {
        **data.iloc[0].to_dict(),
        "game_id": "incomplete_development",
        "listed_qb_rating_sum": np.nan,
    }

    incomplete_holdout = {
        **data.iloc[-1].to_dict(),
        "game_id": "incomplete_holdout",
        "both_short_windows_complete": False,
    }

    data = pd.concat(
        [
            data,
            pd.DataFrame(
                [
                    incomplete_development,
                    incomplete_holdout,
                ]
            ),
        ],
        ignore_index=True,
    )

    development, holdout = (
        prepare_totals_holdout_sample(data)
    )

    assert len(development) == 140
    assert len(holdout) == 20


def test_final_alpha_is_locked() -> None:
    """Reject post-validation alpha changes."""

    assert FINAL_RIDGE_ALPHA == 100.0

    with pytest.raises(
        ValueError,
        match="locked",
    ):
        evaluate_locked_totals_holdout(
            data=create_holdout_data(),
            ridge_alpha=10.0,
        )


def test_overlapping_seasons_are_rejected() -> None:
    """Require strictly future holdout seasons."""

    data = create_holdout_data()

    data.loc[
        data["split_name"] == "holdout",
        "season",
    ] = 2024

    with pytest.raises(
        ValueError,
        match="must precede",
    ):
        prepare_totals_holdout_sample(
            data
        )