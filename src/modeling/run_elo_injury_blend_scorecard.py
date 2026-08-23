"""
NFL Analytics Platform
Elo-Injury Logistic Blend Scorecard

Purpose:
    Test whether blending stable Elo probabilities with
    the strongest aggregate logistic injury challenger
    improves probability quality and robustness.

Audit policy:
    Select the historical audit weight on 2020-2024 OOF
    predictions, then evaluate it unchanged on 2025.

Production policy:
    Select the 2026 candidate weight using all available
    2020-2025 OOF predictions. The next untouched
    forward test is the 2026 season.

Author:
    Ferenc Kaizer

Version:
    0.1.0
"""

import logging
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from src.modeling.model_governance_candidates import (
    GOVERNANCE_CANDIDATES,
    GovernanceCandidate,
)
from src.modeling.run_model_governance_scorecard import (
    GOVERNANCE_VALIDATION_SEASONS,
    create_governance_fold,
    load_governance_data,
    validate_governance_dataset_columns,
)
from src.modeling.train_logistic_baseline import (
    DATABASE_FILE,
    TARGET_COLUMN,
    VALIDATION_SPLIT,
    evaluate_probabilities,
    train_logistic_model,
    validate_database_file,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


INJURY_MODEL_NAME = (
    "logistic_elo_qb_unit_burdens"
)

AUDIT_SELECTION_SEASONS = (
    2020,
    2021,
    2022,
    2023,
    2024,
)

AUDIT_SEASON = 2025

BLEND_WEIGHT_GRID = tuple(
    round(
        weight,
        2,
    )
    for weight in np.linspace(
        0.0,
        1.0,
        21,
    )
)


def get_injury_candidate() -> GovernanceCandidate:
    """Return the frozen unit-injury challenger."""

    matching_candidates = [
        candidate
        for candidate in GOVERNANCE_CANDIDATES
        if candidate.model_name
        == INJURY_MODEL_NAME
    ]

    if len(matching_candidates) != 1:
        raise RuntimeError(
            "Blend scorecard requires exactly one "
            f"{INJURY_MODEL_NAME} candidate."
        )

    return matching_candidates[0]


def create_elo_injury_oof_predictions(
    governance_data: pd.DataFrame,
) -> pd.DataFrame:
    """Create chronological Elo and injury OOF predictions."""

    injury_candidate = get_injury_candidate()

    prediction_frames: list[
        pd.DataFrame
    ] = []

    for validation_season in (
        GOVERNANCE_VALIDATION_SEASONS
    ):
        fold_data = create_governance_fold(
            governance_data=governance_data,
            validation_season=validation_season,
        )

        validation_data = fold_data.loc[
            fold_data["split_name"]
            == VALIDATION_SPLIT
        ].copy()

        model = train_logistic_model(
            development_data=fold_data,
            feature_columns=(
                injury_candidate.feature_columns
            ),
            regularization_c=(
                injury_candidate.regularization_c
            ),
        )

        injury_probabilities = model.predict_proba(
            validation_data.loc[
                :,
                injury_candidate.feature_columns,
            ]
        )[:, 1]

        prediction_frames.append(
            pd.DataFrame(
                {
                    "game_id": validation_data[
                        "game_id"
                    ].to_numpy(),
                    "season": validation_data[
                        "season"
                    ].to_numpy(
                        dtype=int
                    ),
                    "game_date": validation_data[
                        "game_date"
                    ].to_numpy(),
                    TARGET_COLUMN: validation_data[
                        TARGET_COLUMN
                    ].to_numpy(
                        dtype=int
                    ),
                    "elo_probability": (
                        validation_data[
                            "elo_home_win_probability"
                        ].to_numpy(
                            dtype=float
                        )
                    ),
                    "injury_probability": (
                        injury_probabilities
                    ),
                }
            )
        )

    predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    ).sort_values(
        by=[
            "game_date",
            "game_id",
        ]
    ).reset_index(
        drop=True
    )

    if predictions.empty:
        raise RuntimeError(
            "Blend scorecard produced no OOF predictions."
        )

    if predictions["game_id"].duplicated().any():
        raise RuntimeError(
            "Blend OOF predictions contain duplicate games."
        )

    return predictions


def evaluate_blend_weights(
    predictions: pd.DataFrame,
    selection_seasons: tuple[
        int, ...
    ],
    weight_grid: tuple[
        float, ...
    ] = BLEND_WEIGHT_GRID,
) -> pd.DataFrame:
    """Evaluate injury weights on selected OOF seasons."""

    if not selection_seasons:
        raise ValueError(
            "Blend selection seasons must not be empty."
        )

    if not weight_grid:
        raise ValueError(
            "Blend weight grid must not be empty."
        )

    if any(
        weight < 0.0
        or weight > 1.0
        for weight in weight_grid
    ):
        raise ValueError(
            "Blend weights must be between zero and one."
        )

    selection_data = predictions.loc[
        predictions["season"].isin(
            selection_seasons
        )
    ].copy()

    if selection_data.empty:
        raise RuntimeError(
            "No predictions are available for blend selection."
        )

    result_rows: list[
        dict[str, object]
    ] = []

    actual_values = selection_data[
        TARGET_COLUMN
    ]

    for injury_weight in weight_grid:
        elo_weight = (
            1.0
            - injury_weight
        )

        blended_probabilities = (
            elo_weight
            * selection_data[
                "elo_probability"
            ].to_numpy(
                dtype=float
            )
            + injury_weight
            * selection_data[
                "injury_probability"
            ].to_numpy(
                dtype=float
            )
        )

        evaluation = evaluate_probabilities(
            actual_values=actual_values,
            probabilities=blended_probabilities,
        )

        result_rows.append(
            {
                "injury_weight": (
                    injury_weight
                ),
                "elo_weight": elo_weight,
                "season_count": len(
                    selection_seasons
                ),
                "game_count": (
                    evaluation.game_count
                ),
                "accuracy": (
                    evaluation.accuracy
                ),
                "brier_score": (
                    evaluation.brier_score
                ),
                "log_loss": (
                    evaluation.log_loss
                ),
            }
        )

    return pd.DataFrame(
        result_rows
    ).sort_values(
        by=[
            "brier_score",
            "log_loss",
            "injury_weight",
        ],
        ascending=True,
    ).reset_index(
        drop=True
    )


def select_best_blend_weight(
    weight_results: pd.DataFrame,
) -> float:
    """Select the best injury weight by Brier and log loss."""

    if weight_results.empty:
        raise ValueError(
            "Blend weight results must not be empty."
        )

    return float(
        weight_results.iloc[0][
            "injury_weight"
        ]
    )


def evaluate_probability_models(
    predictions: pd.DataFrame,
    seasons: tuple[int, ...],
    injury_weight: float,
    evaluation_period: str,
) -> pd.DataFrame:
    """Evaluate Elo, injury and blend on one period."""

    evaluation_data = predictions.loc[
        predictions["season"].isin(
            seasons
        )
    ].copy()

    if evaluation_data.empty:
        raise RuntimeError(
            "No predictions are available for "
            f"{evaluation_period}."
        )

    actual_values = evaluation_data[
        TARGET_COLUMN
    ]

    probability_sets = {
        "elo": evaluation_data[
            "elo_probability"
        ].to_numpy(
            dtype=float
        ),
        INJURY_MODEL_NAME: evaluation_data[
            "injury_probability"
        ].to_numpy(
            dtype=float
        ),
        "elo_injury_blend": (
            (
                1.0
                - injury_weight
            )
            * evaluation_data[
                "elo_probability"
            ].to_numpy(
                dtype=float
            )
            + injury_weight
            * evaluation_data[
                "injury_probability"
            ].to_numpy(
                dtype=float
            )
        ),
    }

    result_rows: list[
        dict[str, object]
    ] = []

    for (
        model_name,
        probabilities,
    ) in probability_sets.items():
        evaluation = evaluate_probabilities(
            actual_values=actual_values,
            probabilities=probabilities,
        )

        result_rows.append(
            {
                "evaluation_period": (
                    evaluation_period
                ),
                "model_name": model_name,
                "injury_weight": (
                    injury_weight
                    if model_name
                    == "elo_injury_blend"
                    else (
                        1.0
                        if model_name
                        == INJURY_MODEL_NAME
                        else 0.0
                    )
                ),
                "elo_weight": (
                    1.0
                    - injury_weight
                    if model_name
                    == "elo_injury_blend"
                    else (
                        0.0
                        if model_name
                        == INJURY_MODEL_NAME
                        else 1.0
                    )
                ),
                "game_count": (
                    evaluation.game_count
                ),
                "accuracy": (
                    evaluation.accuracy
                ),
                "brier_score": (
                    evaluation.brier_score
                ),
                "log_loss": (
                    evaluation.log_loss
                ),
            }
        )

    return pd.DataFrame(
        result_rows
    ).sort_values(
        by=[
            "brier_score",
            "log_loss",
        ],
        ascending=True,
    ).reset_index(
        drop=True
    )


def log_blend_results(
    audit_weight_results: pd.DataFrame,
    production_weight_results: pd.DataFrame,
    scorecard_results: pd.DataFrame,
    audit_weight: float,
    production_weight: float,
) -> None:
    """Log selected weights and scorecard results."""

    logger.info(
        "Historical audit blend selected on 2020-2024: "
        "injury weight=%.2f | Elo weight=%.2f",
        audit_weight,
        1.0 - audit_weight,
    )

    logger.info(
        "2026 production-candidate blend selected on "
        "2020-2025: injury weight=%.2f | Elo weight=%.2f",
        production_weight,
        1.0 - production_weight,
    )

    logger.info(
        "Top five 2020-2024 blend weights:"
    )

    for row in audit_weight_results.head(
        5
    ).itertuples(
        index=False
    ):
        logger.info(
            "Injury weight=%.2f | Elo weight=%.2f | "
            "Games=%s | Accuracy=%.2f%% | "
            "Brier=%.6f | Log loss=%.6f",
            row.injury_weight,
            row.elo_weight,
            row.game_count,
            100.0 * row.accuracy,
            row.brier_score,
            row.log_loss,
        )

    logger.info(
        "Top five 2020-2025 production weights:"
    )

    for row in production_weight_results.head(
        5
    ).itertuples(
        index=False
    ):
        logger.info(
            "Injury weight=%.2f | Elo weight=%.2f | "
            "Games=%s | Accuracy=%.2f%% | "
            "Brier=%.6f | Log loss=%.6f",
            row.injury_weight,
            row.elo_weight,
            row.game_count,
            100.0 * row.accuracy,
            row.brier_score,
            row.log_loss,
        )

    logger.info(
        "Elo-injury blend scorecard:"
    )

    for row in scorecard_results.itertuples(
        index=False
    ):
        logger.info(
            "Period=%s | Model=%s | "
            "Injury weight=%.2f | Games=%s | "
            "Accuracy=%.2f%% | Brier=%.6f | "
            "Log loss=%.6f",
            row.evaluation_period,
            row.model_name,
            row.injury_weight,
            row.game_count,
            100.0 * row.accuracy,
            row.brier_score,
            row.log_loss,
        )


def run_elo_injury_blend_scorecard(
    database_file: Path = DATABASE_FILE,
) -> pd.DataFrame:
    """Run historical audit and production blend selection."""

    validate_database_file(
        database_file
    )

    logger.info(
        "Starting Elo-injury blend scorecard..."
    )

    with duckdb.connect(
        str(database_file),
        read_only=True,
    ) as connection:
        validate_governance_dataset_columns(
            connection
        )

        governance_data = load_governance_data(
            connection
        )

    predictions = (
        create_elo_injury_oof_predictions(
            governance_data
        )
    )

    audit_weight_results = (
        evaluate_blend_weights(
            predictions=predictions,
            selection_seasons=(
                AUDIT_SELECTION_SEASONS
            ),
        )
    )

    audit_weight = select_best_blend_weight(
        audit_weight_results
    )

    production_weight_results = (
        evaluate_blend_weights(
            predictions=predictions,
            selection_seasons=(
                GOVERNANCE_VALIDATION_SEASONS
            ),
        )
    )

    production_weight = select_best_blend_weight(
        production_weight_results
    )

    scorecard_results = pd.concat(
        [
            evaluate_probability_models(
                predictions=predictions,
                seasons=(
                    AUDIT_SELECTION_SEASONS
                ),
                injury_weight=audit_weight,
                evaluation_period=(
                    "selection_2020_2024"
                ),
            ),
            evaluate_probability_models(
                predictions=predictions,
                seasons=(
                    AUDIT_SEASON,
                ),
                injury_weight=audit_weight,
                evaluation_period=(
                    "historical_audit_2025"
                ),
            ),
            evaluate_probability_models(
                predictions=predictions,
                seasons=(
                    GOVERNANCE_VALIDATION_SEASONS
                ),
                injury_weight=(
                    production_weight
                ),
                evaluation_period=(
                    "production_selection_2020_2025"
                ),
            ),
        ],
        ignore_index=True,
    )

    log_blend_results(
        audit_weight_results=(
            audit_weight_results
        ),
        production_weight_results=(
            production_weight_results
        ),
        scorecard_results=scorecard_results,
        audit_weight=audit_weight,
        production_weight=production_weight,
    )

    logger.info(
        "Elo-injury blend scorecard "
        "completed successfully."
    )

    return scorecard_results


def main() -> None:
    """Run Elo-injury blend scorecard."""

    try:
        run_elo_injury_blend_scorecard()

    except Exception:
        logger.exception(
            "Elo-injury blend scorecard failed."
        )
        raise


if __name__ == "__main__":
    main()