"""
NFL Analytics Platform
Final Totals Model Holdout Evaluation

Purpose:
    Evaluate the locked totals Ridge model exactly once
    on the untouched 2025 holdout.

Locked specification:
    Features:
        Team offensive and defensive EPA aggregates.
        Venue and continuous weather context.
        Listed-QB rating aggregate.
        Previous 64-game league scoring environment.

    Model:
        StandardScaler plus Ridge(alpha=100).

    Training:
        Train and validation splits combined.

    Final evaluation:
        Holdout split only.

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

from src.modeling.evaluate_totals_model_candidates import (
    DATASET_FULL_NAME,
    RAW_TOTALS_FEATURE_COLUMNS,
    SPLIT_FULL_NAME,
    TOTALS_TARGET_COLUMN,
    create_ridge_pipeline,
    create_totals_aggregate_features,
)
from src.modeling.evaluate_spread_model_candidates import (
    calculate_regression_metrics,
)
from src.modeling.production_totals_model import (
    PRODUCTION_TOTALS_MODEL,
)
from src.modeling.train_logistic_baseline import (
    DATABASE_FILE,
    validate_database_file,
)


logger = logging.getLogger(__name__)

FINAL_RIDGE_ALPHA = (
    PRODUCTION_TOTALS_MODEL.ridge_alpha
)

LOCKED_TOTALS_FEATURES = (
    PRODUCTION_TOTALS_MODEL.feature_columns
)

DEVELOPMENT_SPLITS = (
    "train",
    "validation",
)

HOLDOUT_SPLIT = "holdout"

REQUIRED_COLUMNS = {
    "game_id",
    "season",
    "split_name",
    "both_short_windows_complete",
    TOTALS_TARGET_COLUMN,
    *LOCKED_TOTALS_FEATURES,
}

RESULT_COLUMNS = (
    "candidate_name",
    "feature_count",
    "ridge_alpha",
    "training_game_count",
    "holdout_game_count",
    "holdout_mae",
    "holdout_rmse",
    "holdout_bias",
    "holdout_r_squared",
    "mae_improvement_vs_baseline",
    "mae_improvement_percent",
)


def load_totals_holdout_data(
    connection: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    """Load development and holdout totals data."""

    feature_select = ",\n            ".join(
        f"dataset.{feature_name}"
        for feature_name
        in RAW_TOTALS_FEATURE_COLUMNS
    )

    data = connection.execute(
        f"""
        SELECT
            dataset.game_id,
            dataset.season,
            splits.split_name,
            dataset.both_short_windows_complete,
            dataset.{TOTALS_TARGET_COLUMN},
            {feature_select}

        FROM {DATASET_FULL_NAME}
            AS dataset

        INNER JOIN {SPLIT_FULL_NAME}
            AS splits
            ON dataset.game_id = splits.game_id

        WHERE splits.split_name IN (
            'train',
            'validation',
            'holdout'
        )

        ORDER BY
            dataset.season,
            dataset.game_date,
            dataset.game_id
        """
    ).fetchdf()

    return create_totals_aggregate_features(
        data
    )


def prepare_totals_holdout_sample(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create complete development and holdout samples."""

    missing_columns = sorted(
        REQUIRED_COLUMNS - set(data.columns)
    )

    if missing_columns:
        raise ValueError(
            "Totals holdout data is missing columns: "
            + ", ".join(missing_columns)
        )

    if data["game_id"].duplicated().any():
        raise ValueError(
            "Totals holdout data contains duplicate "
            "game identifiers."
        )

    allowed_splits = {
        *DEVELOPMENT_SPLITS,
        HOLDOUT_SPLIT,
    }

    unexpected_splits = sorted(
        set(data["split_name"].dropna())
        - allowed_splits
    )

    if unexpected_splits:
        raise ValueError(
            "Unexpected totals split names: "
            + ", ".join(unexpected_splits)
        )

    complete_mask = (
        data[
            "both_short_windows_complete"
        ].fillna(False).astype(bool)
        & data[TOTALS_TARGET_COLUMN].notna()
        & data[
            list(LOCKED_TOTALS_FEATURES)
        ].notna().all(axis=1)
    )

    complete_data = data.loc[
        complete_mask
    ].copy()

    development_data = complete_data.loc[
        complete_data["split_name"].isin(
            DEVELOPMENT_SPLITS
        )
    ].copy()

    holdout_data = complete_data.loc[
        complete_data["split_name"]
        == HOLDOUT_SPLIT
    ].copy()

    if development_data.empty:
        raise RuntimeError(
            "No complete totals development games "
            "are available."
        )

    if holdout_data.empty:
        raise RuntimeError(
            "No complete totals holdout games "
            "are available."
        )

    if (
        development_data["season"].max()
        >= holdout_data["season"].min()
    ):
        raise ValueError(
            "Totals development seasons must precede "
            "the holdout season."
        )

    return development_data, holdout_data


def rename_holdout_metrics(
    metrics: dict[str, float],
) -> dict[str, float]:
    """Rename shared regression metrics for holdout."""

    return {
        "holdout_mae": metrics[
            "validation_mae"
        ],
        "holdout_rmse": metrics[
            "validation_rmse"
        ],
        "holdout_bias": metrics[
            "validation_bias"
        ],
        "holdout_r_squared": metrics[
            "validation_r_squared"
        ],
    }


def evaluate_locked_totals_holdout(
    data: pd.DataFrame,
    ridge_alpha: float = FINAL_RIDGE_ALPHA,
) -> pd.DataFrame:
    """Evaluate the locked model and constant baseline."""

    if ridge_alpha != FINAL_RIDGE_ALPHA:
        raise ValueError(
            "The final totals Ridge alpha is locked "
            f"at {FINAL_RIDGE_ALPHA}."
        )

    development_data, holdout_data = (
        prepare_totals_holdout_sample(data)
    )

    development_target = development_data[
        TOTALS_TARGET_COLUMN
    ]

    holdout_target = holdout_data[
        TOTALS_TARGET_COLUMN
    ]

    constant_prediction = np.full(
        shape=len(holdout_data),
        fill_value=float(
            development_target.mean()
        ),
    )

    baseline_metrics = rename_holdout_metrics(
        calculate_regression_metrics(
            actual_margin=holdout_target,
            predicted_margin=constant_prediction,
        )
    )

    model = create_ridge_pipeline(
        ridge_alpha=ridge_alpha
    )

    model.fit(
        development_data.loc[
            :,
            list(LOCKED_TOTALS_FEATURES),
        ],
        development_target,
    )

    model_prediction = model.predict(
        holdout_data.loc[
            :,
            list(LOCKED_TOTALS_FEATURES),
        ]
    )

    model_metrics = rename_holdout_metrics(
        calculate_regression_metrics(
            actual_margin=holdout_target,
            predicted_margin=model_prediction,
        )
    )

    baseline_mae = baseline_metrics[
        "holdout_mae"
    ]

    model_improvement = (
        baseline_mae
        - model_metrics["holdout_mae"]
    )

    result_rows = [
        {
            "candidate_name": (
                "constant_development_mean"
            ),
            "feature_count": 0,
            "ridge_alpha": None,
            "training_game_count": len(
                development_data
            ),
            "holdout_game_count": len(
                holdout_data
            ),
            **baseline_metrics,
            "mae_improvement_vs_baseline": 0.0,
            "mae_improvement_percent": 0.0,
        },
        {
            "candidate_name": (
                "ridge_epa_weather_qb_league_64_locked"
            ),
            "feature_count": len(
                LOCKED_TOTALS_FEATURES
            ),
            "ridge_alpha": ridge_alpha,
            "training_game_count": len(
                development_data
            ),
            "holdout_game_count": len(
                holdout_data
            ),
            **model_metrics,
            "mae_improvement_vs_baseline": (
                model_improvement
            ),
            "mae_improvement_percent": (
                100.0
                * model_improvement
                / baseline_mae
            ),
        },
    ]

    return pd.DataFrame(
        result_rows,
        columns=RESULT_COLUMNS,
    ).sort_values(
        by=[
            "holdout_mae",
            "candidate_name",
        ],
        kind="stable",
    ).reset_index(drop=True)


def run_locked_totals_holdout_evaluation(
    database_file: Path = DATABASE_FILE,
) -> pd.DataFrame:
    """Run the final totals holdout evaluation."""

    validate_database_file(
        database_file
    )

    with duckdb.connect(
        str(database_file),
        read_only=True,
    ) as connection:
        data = load_totals_holdout_data(
            connection
        )

    results = evaluate_locked_totals_holdout(
        data=data
    )

    logger.info(
        "Locked totals model evaluated once on "
        "%s holdout games after training on %s games.",
        int(results.iloc[0]["holdout_game_count"]),
        int(results.iloc[0]["training_game_count"]),
    )

    return results


def main() -> None:
    """Run and print final holdout results."""

    results = (
        run_locked_totals_holdout_evaluation()
    )

    print("\nFINAL TOTALS HOLDOUT RESULTS\n")

    print(
        results.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()