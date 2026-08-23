"""Tests for external injury candidate evaluation."""

import pandas as pd
import pytest

from src.modeling.evaluate_injury_model_candidates import (
    evaluate_injury_candidates,
    evaluation_to_row,
)
from src.modeling.train_logistic_baseline import (
    ModelEvaluation,
)


def create_development_data() -> pd.DataFrame:
    """Create deterministic train and validation games."""

    rows: list[dict[str, object]] = []

    specifications = (
        ("train_1", 2019, "train", 1, 80.0, 5.0),
        ("train_2", 2019, "train", 0, -70.0, -4.0),
        ("train_3", 2020, "train", 1, 60.0, 3.0),
        ("train_4", 2020, "train", 0, -55.0, -2.0),
        ("train_5", 2021, "train", 1, 40.0, 2.0),
        ("train_6", 2021, "train", 0, -35.0, -1.0),
        ("validation_2023_1", 2023, "validation", 1, 50.0, 2.5),
        ("validation_2023_2", 2023, "validation", 0, -45.0, -2.5),
        ("validation_2024_1", 2024, "validation", 1, 30.0, 1.5),
        ("validation_2024_2", 2024, "validation", 0, -25.0, -1.5),
    )

    for (
        game_id,
        season,
        split_name,
        target,
        elo_difference,
        qb_difference,
    ) in specifications:
        direction = (
            1.0
            if target == 1
            else -1.0
        )

        rows.append(
            {
                "game_id": game_id,
                "season": season,
                "game_date": (
                    f"{season}-09-10"
                ),
                "split_name": split_name,
                "target_home_win": target,
                "elo_home_win_probability": (
                    0.65
                    if target == 1
                    else 0.35
                ),
                "has_complete_injury_data": True,
                "elo_rating_difference": (
                    elo_difference
                ),
                "listed_qb_rating_difference": (
                    qb_difference
                ),
                "non_qb_injury_burden_difference": (
                    -0.30 * direction
                ),
                "offense_injury_burden_difference": (
                    -0.15 * direction
                ),
                "defense_injury_burden_difference": (
                    -0.10 * direction
                ),
                "special_teams_injury_burden_difference": (
                    -0.05 * direction
                ),
                "out_player_count_difference": (
                    -1.0 * direction
                ),
                "doubtful_player_count_difference": 0.0,
                "questionable_player_count_difference": (
                    -1.0 * direction
                ),
                "starter_out_count_difference": (
                    -1.0 * direction
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def test_evaluation_to_row_calculates_elo_improvement(
) -> None:
    """Calculate Brier and log-loss improvement."""

    evaluation = ModelEvaluation(
        game_count=10,
        accuracy=0.70,
        brier_score=0.20,
        log_loss=0.60,
    )

    elo_evaluation = ModelEvaluation(
        game_count=10,
        accuracy=0.60,
        brier_score=0.23,
        log_loss=0.65,
    )

    row = evaluation_to_row(
        model_name="candidate",
        evaluation=evaluation,
        elo_evaluation=elo_evaluation,
    )

    assert row[
        "brier_improvement_vs_elo"
    ] == pytest.approx(
        0.03
    )

    assert row[
        "log_loss_improvement_vs_elo"
    ] == pytest.approx(
        0.05
    )


def test_evaluate_injury_candidates_compares_models(
) -> None:
    """Evaluate Elo and both logistic candidates."""

    development_data = (
        create_development_data()
    )

    (
        overall_results,
        season_results,
    ) = evaluate_injury_candidates(
        development_data
    )

    assert set(
        overall_results["model_name"]
    ) == {
        "elo",
        "logistic_elo_plus_qb",
        "logistic_elo_qb_unit_burdens",
    }

    assert set(
        overall_results["game_count"]
    ) == {
        4,
    }

    assert set(
        season_results["season"]
    ) == {
        2023,
        2024,
    }

    assert set(
        season_results["game_count"]
    ) == {
        2,
    }


def test_evaluate_injury_candidates_rejects_no_validation(
) -> None:
    """Reject development data without validation games."""

    development_data = (
        create_development_data()
    )

    development_data = development_data.loc[
        development_data["split_name"]
        == "train"
    ].copy()

    with pytest.raises(
        RuntimeError,
        match="No injury validation games",
    ):
        evaluate_injury_candidates(
            development_data
        )