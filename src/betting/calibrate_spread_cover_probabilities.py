"""
NFL Analytics Platform
Spread Cover Probability Calibration

Purpose:
    Create chronological out-of-sample residuals for the
    locked primary and fallback spread models.

    The residual distributions provide the uncertainty
    layer required to convert current point-margin
    predictions into market-line cover probabilities.

    The 2025 holdout is never loaded or evaluated.

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

from src.modeling.evaluate_spread_model_candidates import (
    SPREAD_TARGET_COLUMN,
    create_ridge_pipeline,
)
from src.modeling.production_spread_component import (
    FALLBACK_PREDICTION_MODE,
    PRIMARY_PREDICTION_MODE,
)
from src.modeling.production_spread_model import (
    PRODUCTION_SPREAD_MODEL,
)
from src.modeling.train_logistic_baseline import (
    DATABASE_FILE,
    validate_database_file,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

CALIBRATION_VALIDATION_SEASONS = (
    2021,
    2022,
    2023,
    2024,
)

ALLOWED_DEVELOPMENT_SPLITS = {
    "train",
    "validation",
}

RESIDUAL_COLUMNS = (
    "prediction_mode",
    "model_name",
    "ridge_alpha",
    "validation_season",
    "game_id",
    "training_game_count",
    "actual_home_margin",
    "predicted_home_margin",
    "residual_home_margin",
    "absolute_error",
)

SUMMARY_COLUMNS = (
    "prediction_mode",
    "model_name",
    "ridge_alpha",
    "fold_count",
    "validation_game_count",
    "mean_residual",
    "residual_standard_deviation",
    "mean_absolute_error",
    "root_mean_squared_error",
    "residual_p05",
    "residual_p25",
    "residual_median",
    "residual_p75",
    "residual_p95",
)


def load_external_spread_development_data(
    connection: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    """Load external nfelo spread features before holdout."""

    return connection.execute(
        """
        SELECT
            dataset.game_id,
            dataset.season,
            splits.split_name,
            dataset.target_point_differential,
            external.starting_nfelo_home
                - external.starting_nfelo_away
                AS external_nfelo_rating_difference,
            external.home_538_qb_adj
                - external.away_538_qb_adj
                AS external_nfelo_qb_adjustment_difference
        FROM analytics.game_modeling_dataset AS dataset
        INNER JOIN analytics.modeling_game_splits AS splits
            ON dataset.game_id = splits.game_id
        INNER JOIN processed.external_nfelo_game_ratings AS external
            ON dataset.game_id = external.normalized_game_id
        WHERE splits.split_name IN ('train', 'validation')
        ORDER BY dataset.game_date, dataset.game_id
        """
    ).fetchdf()


def validate_development_data(
    development_data: pd.DataFrame,
) -> None:
    """Validate leakage-safe spread development inputs."""

    required_columns = {
        "game_id",
        "season",
        "split_name",
        SPREAD_TARGET_COLUMN,
        *PRODUCTION_SPREAD_MODEL.feature_columns,
        *(
            PRODUCTION_SPREAD_MODEL
            .fallback_feature_columns
        ),
    }

    missing_columns = sorted(
        required_columns - set(
            development_data.columns
        )
    )

    if missing_columns:
        raise ValueError(
            "Spread calibration data is missing "
            "columns: "
            + ", ".join(missing_columns)
        )

    if development_data.empty:
        raise RuntimeError(
            "Spread calibration data is empty."
        )

    if development_data[
        "game_id"
    ].duplicated().any():
        raise ValueError(
            "Spread calibration data contains "
            "duplicate game identifiers."
        )

    unexpected_splits = sorted(
        set(
            development_data[
                "split_name"
            ].dropna()
        )
        - ALLOWED_DEVELOPMENT_SPLITS
    )

    if unexpected_splits:
        raise ValueError(
            "Spread calibration must not contain "
            "holdout or unknown splits: "
            + ", ".join(unexpected_splits)
        )

    maximum_season = int(
        development_data["season"].max()
    )

    if maximum_season >= (
        PRODUCTION_SPREAD_MODEL
        .forward_test_season
        - 1
    ):
        raise ValueError(
            "Spread calibration data must end before "
            "the 2025 holdout season."
        )


def create_mode_residuals(
    development_data: pd.DataFrame,
    prediction_mode: str,
    model_name: str,
    feature_columns: tuple[str, ...],
    ridge_alpha: float,
    validation_seasons: tuple[
        int, ...
    ] = CALIBRATION_VALIDATION_SEASONS,
) -> pd.DataFrame:
    """Create chronological residuals for one model mode."""

    if not validation_seasons:
        raise ValueError(
            "At least one calibration season is required."
        )

    if tuple(
        sorted(validation_seasons)
    ) != validation_seasons:
        raise ValueError(
            "Calibration seasons must be chronological."
        )

    if len(validation_seasons) != len(
        set(validation_seasons)
    ):
        raise ValueError(
            "Calibration seasons must be unique."
        )

    complete_mask = (
        development_data[
            SPREAD_TARGET_COLUMN
        ].notna()
        & development_data[
            list(feature_columns)
        ].notna().all(axis=1)
    )

    sample = development_data.loc[
        complete_mask
    ].copy()

    if sample.empty:
        raise RuntimeError(
            f"No complete calibration games exist for "
            f"{prediction_mode}."
        )

    available_seasons = set(
        sample["season"].astype(int)
    )

    missing_seasons = sorted(
        set(validation_seasons)
        - available_seasons
    )

    if missing_seasons:
        raise ValueError(
            f"Calibration seasons are missing for "
            f"{prediction_mode}: "
            + ", ".join(
                str(season)
                for season in missing_seasons
            )
        )

    residual_rows: list[
        dict[str, object]
    ] = []

    for validation_season in validation_seasons:
        training_data = sample.loc[
            sample["season"]
            < validation_season
        ].copy()

        validation_data = sample.loc[
            sample["season"]
            == validation_season
        ].copy()

        if training_data.empty:
            raise RuntimeError(
                "No spread calibration training games "
                f"precede {validation_season} for "
                f"{prediction_mode}."
            )

        if validation_data.empty:
            raise RuntimeError(
                "No spread calibration validation games "
                f"exist for {validation_season} and "
                f"{prediction_mode}."
            )

        model = create_ridge_pipeline(
            ridge_alpha=ridge_alpha
        )

        model.fit(
            training_data.loc[
                :,
                list(feature_columns),
            ],
            training_data[
                SPREAD_TARGET_COLUMN
            ],
        )

        predictions = model.predict(
            validation_data.loc[
                :,
                list(feature_columns),
            ]
        )

        for game_id, actual, predicted in zip(
            validation_data["game_id"],
            validation_data[
                SPREAD_TARGET_COLUMN
            ],
            predictions,
            strict=True,
        ):
            residual = float(
                actual - predicted
            )

            residual_rows.append(
                {
                    "prediction_mode": (
                        prediction_mode
                    ),
                    "model_name": model_name,
                    "ridge_alpha": ridge_alpha,
                    "validation_season": (
                        validation_season
                    ),
                    "game_id": game_id,
                    "training_game_count": len(
                        training_data
                    ),
                    "actual_home_margin": float(
                        actual
                    ),
                    "predicted_home_margin": float(
                        predicted
                    ),
                    "residual_home_margin": (
                        residual
                    ),
                    "absolute_error": abs(
                        residual
                    ),
                }
            )

    return pd.DataFrame(
        residual_rows,
        columns=RESIDUAL_COLUMNS,
    )


def create_spread_calibration_residuals(
    development_data: pd.DataFrame,
    validation_seasons: tuple[
        int, ...
    ] = CALIBRATION_VALIDATION_SEASONS,
) -> pd.DataFrame:
    """Create OOF residuals for the locked external model."""

    validate_development_data(
        development_data
    )

    primary_residuals = create_mode_residuals(
        development_data=development_data,
        prediction_mode=PRIMARY_PREDICTION_MODE,
        model_name=(
            PRODUCTION_SPREAD_MODEL.model_name
        ),
        feature_columns=(
            PRODUCTION_SPREAD_MODEL.feature_columns
        ),
        ridge_alpha=(
            PRODUCTION_SPREAD_MODEL.ridge_alpha
        ),
        validation_seasons=validation_seasons,
    )

    residuals = primary_residuals

    if residuals[
        [
            "prediction_mode",
            "game_id",
        ]
    ].duplicated().any():
        raise RuntimeError(
            "Spread calibration contains duplicate "
            "mode-game residuals."
        )

    if not np.isfinite(
        residuals[
            [
                "actual_home_margin",
                "predicted_home_margin",
                "residual_home_margin",
                "absolute_error",
            ]
        ].to_numpy(dtype=float)
    ).all():
        raise RuntimeError(
            "Spread calibration residuals must "
            "be finite."
        )

    return residuals.sort_values(
        by=[
            "prediction_mode",
            "validation_season",
            "game_id",
        ],
        kind="stable",
    ).reset_index(drop=True)


def summarize_spread_calibration(
    residuals: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize uncertainty by production routing mode."""

    missing_columns = sorted(
        set(RESIDUAL_COLUMNS)
        - set(residuals.columns)
    )

    if missing_columns:
        raise ValueError(
            "Spread residual data is missing columns: "
            + ", ".join(missing_columns)
        )

    if residuals.empty:
        raise RuntimeError(
            "Spread residual data is empty."
        )

    summary_rows: list[
        dict[str, object]
    ] = []

    grouped = residuals.groupby(
        [
            "prediction_mode",
            "model_name",
            "ridge_alpha",
        ],
        sort=True,
        dropna=False,
    )

    for (
        prediction_mode,
        model_name,
        ridge_alpha,
    ), group in grouped:
        residual_values = group[
            "residual_home_margin"
        ].to_numpy(dtype=float)

        summary_rows.append(
            {
                "prediction_mode": (
                    prediction_mode
                ),
                "model_name": model_name,
                "ridge_alpha": float(
                    ridge_alpha
                ),
                "fold_count": int(
                    group[
                        "validation_season"
                    ].nunique()
                ),
                "validation_game_count": len(
                    group
                ),
                "mean_residual": float(
                    np.mean(residual_values)
                ),
                "residual_standard_deviation": (
                    float(
                        np.std(
                            residual_values,
                            ddof=1,
                        )
                    )
                ),
                "mean_absolute_error": float(
                    np.mean(
                        np.abs(residual_values)
                    )
                ),
                "root_mean_squared_error": float(
                    np.sqrt(
                        np.mean(
                            np.square(
                                residual_values
                            )
                        )
                    )
                ),
                "residual_p05": float(
                    np.quantile(
                        residual_values,
                        0.05,
                    )
                ),
                "residual_p25": float(
                    np.quantile(
                        residual_values,
                        0.25,
                    )
                ),
                "residual_median": float(
                    np.median(
                        residual_values
                    )
                ),
                "residual_p75": float(
                    np.quantile(
                        residual_values,
                        0.75,
                    )
                ),
                "residual_p95": float(
                    np.quantile(
                        residual_values,
                        0.95,
                    )
                ),
            }
        )

    return pd.DataFrame(
        summary_rows,
        columns=SUMMARY_COLUMNS,
    ).sort_values(
        by="prediction_mode",
        kind="stable",
    ).reset_index(drop=True)


def run_spread_cover_calibration(
    database_file: Path = DATABASE_FILE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load development data and run calibration."""

    validate_database_file(
        database_file
    )

    with duckdb.connect(
        str(database_file),
        read_only=True,
    ) as connection:
        development_data = load_external_spread_development_data(
            connection
        )

    residuals = (
        create_spread_calibration_residuals(
            development_data
        )
    )

    summary = summarize_spread_calibration(
        residuals
    )

    logger.info(
        "Spread cover calibration completed: "
        "%s chronological residuals across "
        "%s production modes without opening holdout.",
        len(residuals),
        residuals[
            "prediction_mode"
        ].nunique(),
    )

    return residuals, summary


def main() -> None:
    """Run and print spread calibration diagnostics."""

    _, summary = run_spread_cover_calibration()

    print("\nSPREAD CALIBRATION SUMMARY\n")

    print(
        summary.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
