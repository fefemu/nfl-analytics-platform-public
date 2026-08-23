"""
Tests for selected candidate-model evaluation.
"""

import pandas as pd

from src.modeling.evaluate_model_candidates import (
    evaluate_model_candidates,
    train_candidate_models,
)
from src.modeling.train_logistic_baseline import (
    MODEL_FEATURE_COLUMNS,
    TARGET_COLUMN,
    SCHEDULE_CONTEXT_FEATURE_COLUMNS,
)


def create_development_frame() -> pd.DataFrame:
    """Create deterministic train and validation data."""

    rows = []

    for index in range(60):
        target = index % 2

        row = {
            "game_id": f"game_{index}",
            "season": (
                2020
                if index < 40
                else 2023
            ),
            "game_date": pd.Timestamp(
                "2020-01-01"
            )
            + pd.Timedelta(days=index),
            "split_name": (
                "train"
                if index < 40
                else "validation"
            ),
            TARGET_COLUMN: target,
            "elo_home_win_probability": (
                0.65 if target else 0.35
            ),
        }

        direction = (
            1.0 if target else -1.0
        )

        for feature_index, feature_name in enumerate(
            MODEL_FEATURE_COLUMNS
        ):
            row[feature_name] = (
                direction
                * (
                    1.0
                    + 0.01 * feature_index
                )
            )
        for (
            feature_index,
            feature_name,
        ) in enumerate(
            SCHEDULE_CONTEXT_FEATURE_COLUMNS
        ):
            row[feature_name] = (
                direction
                * (
                    0.10
                    + 0.01 * feature_index
                )
            )

        rows.append(row)

    return pd.DataFrame(rows)


def test_train_candidate_models_returns_selected_models() -> None:
    """Train every selected development candidate."""

    data = create_development_frame()

    candidates = train_candidate_models(data)

    assert set(candidates) == {
        "logistic_full_core",
        "logistic_elo_plus_qb",
        "hist_gradient_boosting_full_core",
        "xgboost_full_core",
        "logistic_elo_qb_post_bye",
    }


def test_evaluate_model_candidates_includes_elo() -> None:
    """Evaluate candidates and the Elo reference."""

    data = create_development_frame()

    results = evaluate_model_candidates(data)

    assert set(results["model_name"]) == {
        "elo",
        "logistic_full_core",
        "logistic_elo_plus_qb",
        "hist_gradient_boosting_full_core",
        "xgboost_full_core",
        "logistic_elo_qb_post_bye",
    }

    assert set(results["game_count"]) == {
        20,
    }


def test_evaluate_model_candidates_orders_by_brier() -> None:
    """Order external validation results by Brier score."""

    data = create_development_frame()

    results = evaluate_model_candidates(data)

    brier_scores = results[
        "brier_score"
    ].tolist()

    assert brier_scores == sorted(
        brier_scores
    )