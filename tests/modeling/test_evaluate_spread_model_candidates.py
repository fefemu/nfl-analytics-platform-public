"""Tests for spread model candidate evaluation."""

import numpy as np
import pandas as pd
import pytest

from src.modeling.evaluate_spread_model_candidates import (
    RESULT_COLUMNS,
    SPREAD_CORE_FEATURES,
    calculate_regression_metrics,
    create_ridge_pipeline,
    evaluate_spread_model_candidates,
    prepare_common_spread_sample,
)


def create_development_data() -> pd.DataFrame:
    """Create train, validation and incomplete games."""

    rows: list[dict[str, object]] = []

    for index in range(30):
        split_name = (
            "train"
            if index < 20
            else "validation"
        )

        elo_difference = float(
            -120 + index * 10
        )
        qb_difference = float(
            -5 + index * 0.4
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
                "has_complete_injury_data": True,
                "target_point_differential": (
                    0.05 * elo_difference
                    + 1.2 * qb_difference
                    + (
                        1.0
                        if index % 2 == 0
                        else -1.0
                    )
                ),
                "elo_rating_difference": (
                    elo_difference
                ),
                "listed_qb_rating_difference": (
                    qb_difference
                ),
                "offense_injury_burden_difference": (
                    -0.2
                    if index % 2 == 0
                    else 0.2
                ),
                "defense_injury_burden_difference": (
                    -0.1
                    if index % 3 == 0
                    else 0.1
                ),
                "special_teams_injury_burden_difference": (
                    0.0
                ),
            }
        )

    rows.append(
        {
            **rows[-1],
            "game_id": "incomplete_game",
            "listed_qb_rating_difference": np.nan,
        }
    )

    rows.append(
        {
            **rows[-1],
            "game_id": "holdout_game",
            "split_name": "holdout",
        }
    )

    return pd.DataFrame(rows)


def test_prepare_common_sample(
) -> None:
    """Use identical complete train and validation games."""

    sample = prepare_common_spread_sample(
        create_development_data()
    )

    assert len(sample) == 30

    assert set(
        sample["split_name"]
    ) == {
        "train",
        "validation",
    }

    assert sample[
        list(SPREAD_CORE_FEATURES)
    ].notna().all().all()


def test_evaluate_candidates_on_identical_sample(
) -> None:
    """Compare every candidate on the same games."""

    results = (
        evaluate_spread_model_candidates(
            create_development_data()
        )
    )

    assert tuple(
        results.columns
    ) == RESULT_COLUMNS

    assert set(
        results["candidate_name"]
    ) == {
        "constant_train_mean",
        "ridge_elo",
        "ridge_elo_qb",
        "ridge_elo_qb_injury",
    }

    assert set(
        results["train_game_count"]
    ) == {
        20,
    }

    assert set(
        results["validation_game_count"]
    ) == {
        10,
    }


def test_regression_candidate_beats_constant(
) -> None:
    """Recover signal from leakage-safe features."""

    results = (
        evaluate_spread_model_candidates(
            create_development_data()
        )
    ).set_index(
        "candidate_name"
    )

    assert (
        results.loc[
            "ridge_elo_qb",
            "validation_mae",
        ]
        < results.loc[
            "constant_train_mean",
            "validation_mae",
        ]
    )


def test_calculate_regression_metrics(
) -> None:
    """Calculate MAE, RMSE, bias and R-squared."""

    metrics = calculate_regression_metrics(
        actual_margin=pd.Series(
            [
                3.0,
                -1.0,
            ]
        ),
        predicted_margin=np.array(
            [
                2.0,
                1.0,
            ]
        ),
    )

    assert metrics[
        "validation_mae"
    ] == pytest.approx(1.5)

    assert metrics[
        "validation_rmse"
    ] == pytest.approx(
        np.sqrt(2.5)
    )

    assert metrics[
        "validation_bias"
    ] == pytest.approx(0.5)


def test_ridge_rejects_negative_alpha(
) -> None:
    """Reject invalid regularization."""

    with pytest.raises(
        ValueError,
        match="must not be negative",
    ):
        create_ridge_pipeline(
            ridge_alpha=-1.0
        )