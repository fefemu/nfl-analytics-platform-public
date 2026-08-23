"""
Tests for logistic-regression feature ablation.
"""

import pandas as pd
import pytest

from src.modeling.run_logistic_ablation import (
    FEATURE_GROUPS,
    evaluation_to_row,
    run_feature_ablation,
    validate_feature_groups,
    ELO_QB_SCHEDULE_FEATURES,
    ELO_QB_EXTENDED_REST_FEATURES,
    ELO_QB_POST_BYE_FEATURES,
    ELO_QB_REST_DIFFERENCE_FEATURES,
    ELO_QB_SHORT_WEEK_FEATURES,
)
from src.modeling.train_logistic_baseline import (
    MODEL_FEATURE_COLUMNS,
    TARGET_COLUMN,
    ModelEvaluation,
    SCHEDULE_CONTEXT_FEATURE_COLUMNS,
)


def create_development_frame() -> pd.DataFrame:
    """Create deterministic train and validation data."""

    rows = []

    for index in range(20):
        split_name = (
            "train"
            if index < 16
            else "validation"
        )

        target = index % 2

        row = {
            "game_id": f"game_{index}",
            "season": 2020,
            "game_date": pd.Timestamp(
                "2020-01-01"
            )
            + pd.Timedelta(days=index),
            "split_name": split_name,
            TARGET_COLUMN: target,
            "elo_home_win_probability": (
                0.65 if target else 0.35
            ),
        }

        for feature_index, feature_name in enumerate(
            MODEL_FEATURE_COLUMNS
        ):
            direction = 1.0 if target else -1.0

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


def test_validate_feature_groups_accepts_configuration() -> None:
    """Accept known, unique and non-empty feature groups."""

    validate_feature_groups()


def test_feature_groups_are_nested_as_expected() -> None:
    """Keep Elo and QB groups logically nested."""

    assert set(FEATURE_GROUPS["elo_only"]).issubset(
        FEATURE_GROUPS["elo_plus_qb"]
    )

    assert set(FEATURE_GROUPS["elo_plus_qb"]).issubset(
        FEATURE_GROUPS["full_core"]
    )

    assert set(
        FEATURE_GROUPS["elo_plus_stable_epa"]
    ).issubset(
        FEATURE_GROUPS["full_core"]
    )


def test_evaluation_to_row_calculates_elo_improvement() -> None:
    """Calculate positive improvement for lower model losses."""

    evaluation = ModelEvaluation(
        game_count=10,
        accuracy=0.70,
        brier_score=0.20,
        log_loss=0.60,
    )

    elo_evaluation = ModelEvaluation(
        game_count=10,
        accuracy=0.60,
        brier_score=0.22,
        log_loss=0.64,
    )

    row = evaluation_to_row(
        model_name="test_model",
        feature_columns=("elo_rating_difference",),
        evaluation=evaluation,
        elo_evaluation=elo_evaluation,
        regularization_c=1.0,
    )

    assert row["brier_improvement_vs_elo"] == pytest.approx(
        0.02
    )

    assert row["log_loss_improvement_vs_elo"] == pytest.approx(
        0.04
    )


def test_run_feature_ablation_returns_every_model() -> None:
    """Train and evaluate every configured feature group."""

    data = create_development_frame()

    results = run_feature_ablation(
        development_data=data,
        regularization_c=0.1,
    )

    assert set(results["model_name"]) == set(
        FEATURE_GROUPS
    )

    assert len(results) == len(FEATURE_GROUPS)
    assert set(results["game_count"]) == {4}
    assert set(results["regularization_c"]) == {0.1}


def test_run_feature_ablation_orders_by_brier_score() -> None:
    """Order candidate models by the primary metric."""

    data = create_development_frame()

    results = run_feature_ablation(
        development_data=data,
    )

    brier_scores = results["brier_score"].tolist()

    assert brier_scores == sorted(brier_scores)


def test_schedule_group_extends_elo_and_qb() -> None:
    """Add schedule context to the compact model."""

    assert set(
        FEATURE_GROUPS["elo_plus_qb"]
    ).issubset(
        ELO_QB_SCHEDULE_FEATURES
    )

    assert len(
        ELO_QB_SCHEDULE_FEATURES
    ) == 6


@pytest.mark.parametrize(
    (
        "model_name",
        "feature_columns",
    ),
    [
        (
            "elo_qb_rest_difference",
            ELO_QB_REST_DIFFERENCE_FEATURES,
        ),
        (
            "elo_qb_short_week",
            ELO_QB_SHORT_WEEK_FEATURES,
        ),
        (
            "elo_qb_extended_rest",
            ELO_QB_EXTENDED_REST_FEATURES,
        ),
        (
            "elo_qb_post_bye",
            ELO_QB_POST_BYE_FEATURES,
        ),
    ],
)
def test_single_schedule_groups_extend_compact_model(
    model_name: str,
    feature_columns: tuple[str, ...],
) -> None:
    """Add one schedule feature to Elo and QB."""

    assert set(
        FEATURE_GROUPS["elo_plus_qb"]
    ).issubset(
        feature_columns
    )

    assert len(feature_columns) == 3

    assert FEATURE_GROUPS[
        model_name
    ] == feature_columns