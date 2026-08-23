"""Tests for final spread holdout evaluation."""

import numpy as np
import pandas as pd
import pytest

from src.modeling.evaluate_spread_model_holdout import (
    FINAL_RIDGE_ALPHA,
    RESULT_COLUMNS,
    evaluate_locked_spread_holdout,
    prepare_spread_holdout_sample,
)


def create_holdout_data() -> pd.DataFrame:
    """Create development and untouched holdout games."""

    rows: list[dict[str, object]] = []

    for season in range(2018, 2026):
        for game_index in range(20):
            elo_difference = float(
                -120 + game_index * 12
            )

            qb_difference = float(
                -5 + game_index * 0.5
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
                    "target_point_differential": (
                        0.04 * elo_difference
                        + 1.0 * qb_difference
                        + (
                            0.5
                            if game_index % 2 == 0
                            else -0.5
                        )
                    ),
                    "elo_rating_difference": (
                        elo_difference
                    ),
                    "listed_qb_rating_difference": (
                        qb_difference
                    ),
                }
            )

    return pd.DataFrame(rows)


def test_prepare_separates_development_and_holdout(
) -> None:
    """Keep holdout games outside model training."""

    development, holdout = (
        prepare_spread_holdout_sample(
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
    """Recover the synthetic spread signal."""

    results = evaluate_locked_spread_holdout(
        create_holdout_data()
    ).set_index(
        "candidate_name"
    )

    assert (
        results.loc[
            "ridge_elo_qb_locked",
            "holdout_mae",
        ]
        < results.loc[
            "constant_development_mean",
            "holdout_mae",
        ]
    )

    assert (
        results.loc[
            "ridge_elo_qb_locked",
            "mae_improvement_percent",
        ]
        > 0.0
    )


def test_holdout_result_schema_and_counts() -> None:
    """Return the locked two-row comparison."""

    results = evaluate_locked_spread_holdout(
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
    """Require complete target, Elo and QB values."""

    data = create_holdout_data()

    incomplete_development = {
        **data.iloc[0].to_dict(),
        "game_id": "incomplete_development",
        "listed_qb_rating_difference": np.nan,
    }

    incomplete_holdout = {
        **data.iloc[-1].to_dict(),
        "game_id": "incomplete_holdout",
        "elo_rating_difference": np.nan,
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
        prepare_spread_holdout_sample(data)
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
        evaluate_locked_spread_holdout(
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
        prepare_spread_holdout_sample(
            data
        )