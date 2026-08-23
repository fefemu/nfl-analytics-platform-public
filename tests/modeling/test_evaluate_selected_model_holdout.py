"""Tests for the frozen selected-model holdout evaluation."""

import pandas as pd
import pytest

from src.modeling.evaluate_selected_model_holdout import (
    evaluate_frozen_model_on_holdout,
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


def create_final_evaluation_frame() -> pd.DataFrame:
    """Create deterministic development and holdout data."""

    rows = []

    split_names = (
        [TRAIN_SPLIT] * 20
        + [VALIDATION_SPLIT] * 10
        + [HOLDOUT_SPLIT] * 10
    )

    for index, split_name in enumerate(split_names):
        target = index % 2
        direction = 1.0 if target else -1.0

        rows.append(
            {
                "game_id": f"game_{index}",
                "season": (
                    2022
                    if split_name == TRAIN_SPLIT
                    else (
                        2024
                        if split_name
                        == VALIDATION_SPLIT
                        else 2025
                    )
                ),
                "game_date": pd.Timestamp(
                    "2022-01-01"
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
                "post_bye_difference": 0,
            }
        )

    return pd.DataFrame(rows)


def test_train_frozen_model_uses_development_splits() -> None:
    """Train the frozen model before holdout evaluation."""

    evaluation_data = create_final_evaluation_frame()

    model = train_frozen_selected_model(
        evaluation_data
    )

    probabilities = model.predict_proba(
        evaluation_data.loc[
            evaluation_data["split_name"]
            == HOLDOUT_SPLIT,
            SELECTED_MODEL.feature_columns,
        ]
    )[:, 1]

    assert len(probabilities) == 10
    assert all(
        0.0 < probability < 1.0
        for probability in probabilities
    )


def test_evaluate_frozen_model_uses_holdout_only() -> None:
    """Return metrics for the ten holdout games only."""

    evaluation_data = create_final_evaluation_frame()

    model = train_frozen_selected_model(
        evaluation_data
    )

    evaluations = evaluate_frozen_model_on_holdout(
        model=model,
        evaluation_data=evaluation_data,
    )

    assert set(evaluations) == {
        SELECTED_MODEL.model_name,
        "elo",
    }
    assert (
        evaluations[
            SELECTED_MODEL.model_name
        ].game_count
        == 10
    )
    assert evaluations["elo"].game_count == 10


def test_holdout_evaluation_rejects_missing_holdout() -> None:
    """Reject evaluation data without holdout games."""

    evaluation_data = create_final_evaluation_frame()

    development_only = evaluation_data.loc[
        evaluation_data["split_name"]
        != HOLDOUT_SPLIT
    ].copy()

    model = train_frozen_selected_model(
        evaluation_data
    )

    with pytest.raises(
        RuntimeError,
        match="No holdout games",
    ):
        evaluate_frozen_model_on_holdout(
            model=model,
            evaluation_data=development_only,
        )