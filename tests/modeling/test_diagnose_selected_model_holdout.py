"""Tests for selected-model holdout diagnostics."""

import pandas as pd

from src.modeling.diagnose_selected_model_holdout import (
    assign_season_phase,
    build_disagreement_diagnostics,
    build_feature_drift_table,
    build_phase_diagnostics,
    create_holdout_diagnostic_predictions,
)
from src.modeling.evaluate_selected_model_holdout import (
    train_frozen_selected_model,
)
from src.modeling.selected_model import (
    SELECTED_MODEL,
)
from src.modeling.train_logistic_baseline import (
    HOLDOUT_SPLIT,
    TARGET_COLUMN,
    TRAIN_SPLIT,
    VALIDATION_SPLIT,
)


def create_diagnostic_frame() -> pd.DataFrame:
    """Create deterministic diagnostic test data."""

    rows = []

    split_names = (
        [TRAIN_SPLIT] * 20
        + [VALIDATION_SPLIT] * 10
        + [HOLDOUT_SPLIT] * 12
    )

    holdout_weeks = (
        1,
        3,
        5,
        6,
        7,
        9,
        11,
        12,
        13,
        15,
        18,
        19,
    )

    for index, split_name in enumerate(split_names):
        target = index % 2
        direction = 1.0 if target else -1.0

        if split_name == HOLDOUT_SPLIT:
            holdout_index = index - 30
            week = holdout_weeks[holdout_index]
            game_type = (
                "POST"
                if holdout_index == 11
                else "REG"
            )
        else:
            week = index % 18 + 1
            game_type = "REG"

        rows.append(
            {
                "game_id": f"game_{index}",
                "season": (
                    2025
                    if split_name == HOLDOUT_SPLIT
                    else 2024
                ),
                "game_type": game_type,
                "week": week,
                "game_date": pd.Timestamp(
                    "2024-01-01"
                )
                + pd.Timedelta(days=index),
                "split_name": split_name,
                TARGET_COLUMN: target,
                "elo_home_win_probability": (
                    0.65 if target else 0.35
                ),
                "elo_rating_difference": (
                    direction * 50.0
                ),
                "listed_qb_rating_difference": (
                    direction * 5.0
                ),
                "post_bye_difference": (
                    1 if index % 5 == 0 else 0
                ),
            }
        )

    return pd.DataFrame(rows)


def test_assign_season_phase() -> None:
    """Assign regular-season phases and postseason."""

    assert (
        assign_season_phase("REG", 4)
        == "early_regular_season"
    )
    assert (
        assign_season_phase("REG", 9)
        == "middle_regular_season"
    )
    assert (
        assign_season_phase("REG", 16)
        == "late_regular_season"
    )
    assert (
        assign_season_phase("POST", 19)
        == "postseason"
    )


def test_create_holdout_diagnostic_predictions() -> None:
    """Create predictions for holdout games only."""

    evaluation_data = create_diagnostic_frame()

    model = train_frozen_selected_model(
        evaluation_data
    )

    predictions = (
        create_holdout_diagnostic_predictions(
            model=model,
            evaluation_data=evaluation_data,
        )
    )

    assert len(predictions) == 12
    assert set(predictions["split_name"]) == {
        HOLDOUT_SPLIT
    }
    assert (
        predictions[
            "logistic_home_win_probability"
        ].between(0.0, 1.0).all()
    )


def test_build_phase_diagnostics() -> None:
    """Evaluate every populated holdout season phase."""

    evaluation_data = create_diagnostic_frame()

    model = train_frozen_selected_model(
        evaluation_data
    )

    predictions = (
        create_holdout_diagnostic_predictions(
            model=model,
            evaluation_data=evaluation_data,
        )
    )

    diagnostics = build_phase_diagnostics(
        predictions
    )

    assert diagnostics["game_count"].sum() == 12
    assert set(diagnostics["season_phase"]) == {
        "early_regular_season",
        "middle_regular_season",
        "late_regular_season",
        "postseason",
    }


def test_build_feature_drift_table() -> None:
    """Measure drift for every selected model feature."""

    evaluation_data = create_diagnostic_frame()

    drift_table = build_feature_drift_table(
        evaluation_data
    )

    assert set(drift_table["feature_name"]) == set(
        SELECTED_MODEL.feature_columns
    )
    assert len(drift_table) == len(
        SELECTED_MODEL.feature_columns
    )
    assert (
        drift_table["development_game_count"] > 0
    ).all()
    assert (
        drift_table["holdout_game_count"] > 0
    ).all()


def test_build_disagreement_diagnostics() -> None:
    """Summarize logistic changes relative to Elo."""

    evaluation_data = create_diagnostic_frame()

    model = train_frozen_selected_model(
        evaluation_data
    )

    predictions = (
        create_holdout_diagnostic_predictions(
            model=model,
            evaluation_data=evaluation_data,
        )
    )

    diagnostics = build_disagreement_diagnostics(
        predictions
    )

    assert "all_holdout" in set(
        diagnostics["diagnostic_group"]
    )

    overall = diagnostics.loc[
        diagnostics["diagnostic_group"]
        == "all_holdout"
    ].iloc[0]

    assert overall["game_count"] == 12
    assert (
        overall["disagreement_count"]
        <= overall["game_count"]
    )
    assert (
        0.0
        <= overall["disagreement_rate"]
        <= 1.0
    )
    assert (
        overall[
            "mean_absolute_probability_change"
        ]
        >= 0.0
    )