"""
NFL Analytics Platform
Totals Model Candidate Evaluation

Purpose:
    Compare leakage-safe totals regression candidates on
    one identical train and validation sample.

Target:
    Home score plus away score.

The 2025 holdout is never loaded or evaluated.

Author:
    Ferenc Kaizer

Version:
    0.1.0
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from src.modeling.evaluate_spread_model_candidates import (
    calculate_regression_metrics,
    create_ridge_pipeline,
)
from src.modeling.train_logistic_baseline import (
    DATABASE_FILE,
    validate_database_file,
)


logger = logging.getLogger(__name__)

DATASET_FULL_NAME = (
    "analytics.game_modeling_dataset"
)

SPLIT_FULL_NAME = (
    "analytics.modeling_game_splits"
)

TRAIN_SPLIT = "train"
VALIDATION_SPLIT = "validation"

TOTALS_TARGET_COLUMN = "target_total_points"

RAW_TOTALS_FEATURE_COLUMNS = (
    "home_offensive_plays_last_4",
    "away_offensive_plays_last_4",
    "home_points_scored_last_4",
    "away_points_scored_last_4",
    "home_points_allowed_last_4",
    "away_points_allowed_last_4",
    "home_offensive_epa_per_play_last_4",
    "away_offensive_epa_per_play_last_4",
    "home_defensive_epa_allowed_per_play_last_4",
    "away_defensive_epa_allowed_per_play_last_4",
    "home_listed_qb_rating",
    "away_listed_qb_rating",
    "home_success_rate_last_4",
    "away_success_rate_last_4",
    "home_defensive_success_rate_allowed_last_4",
    "away_defensive_success_rate_allowed_last_4",
    "home_explosive_play_rate_last_4",
    "away_explosive_play_rate_last_4",
    "home_explosive_play_rate_allowed_last_4",
    "away_explosive_play_rate_allowed_last_4",
    "is_indoor",
    "has_game_weather",
    "cold_degrees_below_50",
    "heat_degrees_above_80",
    "wind_mph_above_10",
    "is_freezing",
    "is_high_wind",
    "is_extreme_heat",
    "league_average_total_last_32",
    "league_average_total_last_64",
    "league_average_total_last_128",
)

PACE_TOTALS_FEATURES = (
    "offensive_plays_sum_last_4",
)

SCORING_TOTALS_FEATURES = (
    "points_scored_sum_last_4",
    "points_allowed_sum_last_4",
)

EPA_TOTALS_FEATURES = (
    "offensive_epa_sum_last_4",
    "defensive_epa_allowed_sum_last_4",
)

QB_TOTALS_FEATURES = (
    "listed_qb_rating_sum",
)

SUCCESS_TOTALS_FEATURES = (
    "offensive_success_rate_sum_last_4",
    "defensive_success_rate_allowed_sum_last_4",
)

EXPLOSIVE_TOTALS_FEATURES = (
    "offensive_explosive_play_rate_sum_last_4",
    "defensive_explosive_play_rate_allowed_sum_last_4",
)

VENUE_TOTALS_FEATURES = (
    "is_indoor",
)

CONTINUOUS_WEATHER_TOTALS_FEATURES = (
    "is_indoor",
    "has_game_weather",
    "cold_degrees_below_50",
    "heat_degrees_above_80",
    "wind_mph_above_10",
)

EXTREME_WEATHER_TOTALS_FEATURES = (
    "is_indoor",
    "has_game_weather",
    "is_freezing",
    "is_high_wind",
    "is_extreme_heat",
)

LEAGUE_SCORING_32_TOTALS_FEATURES = (
    "league_average_total_last_32",
)

LEAGUE_SCORING_64_TOTALS_FEATURES = (
    "league_average_total_last_64",
)

LEAGUE_SCORING_128_TOTALS_FEATURES = (
    "league_average_total_last_128",
)

TOTALS_SELECTED_BASE_FEATURES = (
    *EPA_TOTALS_FEATURES,
    *CONTINUOUS_WEATHER_TOTALS_FEATURES,
    *QB_TOTALS_FEATURES,
)

TOTALS_CORE_FEATURES = (
    *EPA_TOTALS_FEATURES,
    *CONTINUOUS_WEATHER_TOTALS_FEATURES,
    *QB_TOTALS_FEATURES,
    *SUCCESS_TOTALS_FEATURES,
    *EXPLOSIVE_TOTALS_FEATURES,
)

TOTALS_CANDIDATE_SAMPLE_FEATURES = (
    *TOTALS_CORE_FEATURES,
    *LEAGUE_SCORING_32_TOTALS_FEATURES,
    *LEAGUE_SCORING_64_TOTALS_FEATURES,
    *LEAGUE_SCORING_128_TOTALS_FEATURES,
)

REQUIRED_SOURCE_COLUMNS = {
    "game_id",
    "season",
    "split_name",
    "both_short_windows_complete",
    TOTALS_TARGET_COLUMN,
    *RAW_TOTALS_FEATURE_COLUMNS,
}


@dataclass(frozen=True)
class TotalsModelCandidate:
    """Describe one totals model candidate."""

    candidate_name: str
    feature_columns: tuple[str, ...]
    ridge_alpha: float


TOTALS_MODEL_CANDIDATES = (
    TotalsModelCandidate(
        candidate_name="ridge_epa",
        feature_columns=EPA_TOTALS_FEATURES,
        ridge_alpha=1.0,
    ),
    TotalsModelCandidate(
        candidate_name="ridge_epa_indoor",
        feature_columns=(
            *EPA_TOTALS_FEATURES,
            *VENUE_TOTALS_FEATURES,
        ),
        ridge_alpha=1.0,
    ),
    TotalsModelCandidate(
        candidate_name=(
            "ridge_epa_weather_continuous"
        ),
        feature_columns=(
            *EPA_TOTALS_FEATURES,
            *CONTINUOUS_WEATHER_TOTALS_FEATURES,
        ),
        ridge_alpha=1.0,
    ),
    TotalsModelCandidate(
        candidate_name=(
            "ridge_epa_weather_extremes"
        ),
        feature_columns=(
            *EPA_TOTALS_FEATURES,
            *EXTREME_WEATHER_TOTALS_FEATURES,
        ),
        ridge_alpha=1.0,
    ),
    TotalsModelCandidate(
        candidate_name=(
            "ridge_epa_weather_continuous_qb"
        ),
        feature_columns=(
            *EPA_TOTALS_FEATURES,
            *CONTINUOUS_WEATHER_TOTALS_FEATURES,
            *QB_TOTALS_FEATURES,
        ),
        ridge_alpha=1.0,
    ),
    TotalsModelCandidate(
        candidate_name=(
            "ridge_epa_weather_continuous_qb_"
            "success_explosive"
        ),
        feature_columns=TOTALS_CORE_FEATURES,
        ridge_alpha=1.0,
    ),
    TotalsModelCandidate(
        candidate_name=(
            "ridge_epa_weather_qb_league_32"
        ),
        feature_columns=(
            *TOTALS_SELECTED_BASE_FEATURES,
            *LEAGUE_SCORING_32_TOTALS_FEATURES,
        ),
        ridge_alpha=1.0,
    ),
    TotalsModelCandidate(
        candidate_name=(
            "ridge_epa_weather_qb_league_64"
        ),
        feature_columns=(
            *TOTALS_SELECTED_BASE_FEATURES,
            *LEAGUE_SCORING_64_TOTALS_FEATURES,
        ),
        ridge_alpha=1.0,
    ),
    TotalsModelCandidate(
        candidate_name=(
            "ridge_epa_weather_qb_league_128"
        ),
        feature_columns=(
            *TOTALS_SELECTED_BASE_FEATURES,
            *LEAGUE_SCORING_128_TOTALS_FEATURES,
        ),
        ridge_alpha=1.0,
    ),
)


RESULT_COLUMNS = (
    "candidate_name",
    "feature_count",
    "ridge_alpha",
    "train_game_count",
    "validation_game_count",
    "validation_mae",
    "validation_rmse",
    "validation_bias",
    "validation_r_squared",
)


def validate_source_columns(
    data: pd.DataFrame,
) -> None:
    """Validate the totals source schema."""

    missing_columns = sorted(
        REQUIRED_SOURCE_COLUMNS
        - set(data.columns)
    )

    if missing_columns:
        raise ValueError(
            "Totals development data is missing "
            "columns: "
            + ", ".join(missing_columns)
        )


def create_totals_aggregate_features(
    source_data: pd.DataFrame,
) -> pd.DataFrame:
    """Create team-order-invariant totals features."""

    validate_source_columns(source_data)

    data = source_data.copy()
    data[
        "offensive_plays_sum_last_4"
    ] = (
        data["home_offensive_plays_last_4"]
        + data["away_offensive_plays_last_4"]
    )

    data[
        "points_scored_sum_last_4"
    ] = (
        data["home_points_scored_last_4"]
        + data["away_points_scored_last_4"]
    )

    data[
        "points_allowed_sum_last_4"
    ] = (
        data["home_points_allowed_last_4"]
        + data["away_points_allowed_last_4"]
    )

    data[
        "offensive_epa_sum_last_4"
    ] = (
        data[
            "home_offensive_epa_per_play_last_4"
        ]
        + data[
            "away_offensive_epa_per_play_last_4"
        ]
    )

    data[
        "defensive_epa_allowed_sum_last_4"
    ] = (
        data[
            "home_defensive_epa_allowed_per_play_last_4"
        ]
        + data[
            "away_defensive_epa_allowed_per_play_last_4"
        ]
    )

    data["listed_qb_rating_sum"] = (
        data["home_listed_qb_rating"]
        + data["away_listed_qb_rating"]
    )

    data[
        "offensive_success_rate_sum_last_4"
    ] = (
        data["home_success_rate_last_4"]
        + data["away_success_rate_last_4"]
    )

    data[
        "defensive_success_rate_allowed_sum_last_4"
    ] = (
        data[
            "home_defensive_success_rate_allowed_last_4"
        ]
        + data[
            "away_defensive_success_rate_allowed_last_4"
        ]
    )

    data[
        "offensive_explosive_play_rate_sum_last_4"
    ] = (
        data[
            "home_explosive_play_rate_last_4"
        ]
        + data[
            "away_explosive_play_rate_last_4"
        ]
    )

    data[
        "defensive_explosive_play_rate_allowed_sum_last_4"
    ] = (
        data[
            "home_explosive_play_rate_allowed_last_4"
        ]
        + data[
            "away_explosive_play_rate_allowed_last_4"
        ]
    )

    return data


def load_totals_development_data(
    connection: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    """Load train and validation totals inputs."""

    feature_select = ",\n            ".join(
        f"dataset.{column_name}"
        for column_name
        in RAW_TOTALS_FEATURE_COLUMNS
    )

    source_data = connection.execute(
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
            '{TRAIN_SPLIT}',
            '{VALIDATION_SPLIT}'
        )

        ORDER BY
            dataset.season,
            dataset.game_date,
            dataset.game_id
        """
    ).fetchdf()

    if source_data.empty:
        raise RuntimeError(
            "No totals development data is available."
        )

    return create_totals_aggregate_features(
        source_data
    )


def prepare_common_totals_sample(
    development_data: pd.DataFrame,
) -> pd.DataFrame:
    """Create one identical complete candidate sample."""

    validate_source_columns(
        development_data
    )

    data = create_totals_aggregate_features(
        development_data
    )

    allowed_split_mask = data[
        "split_name"
    ].isin(
        [
            TRAIN_SPLIT,
            VALIDATION_SPLIT,
        ]
    )

    complete_mask = (
        allowed_split_mask
        & data[
            "both_short_windows_complete"
        ].fillna(False).astype(bool)
        & data[TOTALS_TARGET_COLUMN].notna()
        & data[
            list(TOTALS_CANDIDATE_SAMPLE_FEATURES)
        ].notna().all(axis=1)
    )

    sample = data.loc[
        complete_mask
    ].copy()

    if sample.empty:
        raise RuntimeError(
            "No complete totals candidate games "
            "are available."
        )

    if sample["game_id"].duplicated().any():
        raise ValueError(
            "Totals development data contains "
            "duplicate game identifiers."
        )

    if set(sample["split_name"]) != {
        TRAIN_SPLIT,
        VALIDATION_SPLIT,
    }:
        raise RuntimeError(
            "Totals candidate sample must contain "
            "train and validation games."
        )

    return sample


def evaluate_totals_model_candidates(
    development_data: pd.DataFrame,
    candidates: tuple[
        TotalsModelCandidate, ...
    ] = TOTALS_MODEL_CANDIDATES,
) -> pd.DataFrame:
    """Evaluate totals candidates on identical games."""

    if not candidates:
        raise ValueError(
            "At least one totals candidate is required."
        )

    candidate_names = [
        candidate.candidate_name
        for candidate in candidates
    ]

    if len(candidate_names) != len(
        set(candidate_names)
    ):
        raise ValueError(
            "Totals candidate names must be unique."
        )

    sample = prepare_common_totals_sample(
        development_data
    )

    train_data = sample.loc[
        sample["split_name"] == TRAIN_SPLIT
    ]

    validation_data = sample.loc[
        sample["split_name"]
        == VALIDATION_SPLIT
    ]

    train_target = train_data[
        TOTALS_TARGET_COLUMN
    ]

    validation_target = validation_data[
        TOTALS_TARGET_COLUMN
    ]

    result_rows: list[
        dict[str, object]
    ] = []

    constant_prediction = np.full(
        shape=len(validation_data),
        fill_value=float(train_target.mean()),
    )

    result_rows.append(
        {
            "candidate_name": (
                "constant_train_mean"
            ),
            "feature_count": 0,
            "ridge_alpha": None,
            "train_game_count": len(
                train_data
            ),
            "validation_game_count": len(
                validation_data
            ),
            **calculate_regression_metrics(
                actual_margin=validation_target,
                predicted_margin=constant_prediction,
            ),
        }
    )

    for candidate in candidates:
        model = create_ridge_pipeline(
            ridge_alpha=candidate.ridge_alpha
        )

        feature_columns = list(
            candidate.feature_columns
        )

        model.fit(
            train_data.loc[
                :,
                feature_columns,
            ],
            train_target,
        )

        predicted_total = model.predict(
            validation_data.loc[
                :,
                feature_columns,
            ]
        )

        result_rows.append(
            {
                "candidate_name": (
                    candidate.candidate_name
                ),
                "feature_count": len(
                    candidate.feature_columns
                ),
                "ridge_alpha": (
                    candidate.ridge_alpha
                ),
                "train_game_count": len(
                    train_data
                ),
                "validation_game_count": len(
                    validation_data
                ),
                **calculate_regression_metrics(
                    actual_margin=validation_target,
                    predicted_margin=predicted_total,
                ),
            }
        )

    return pd.DataFrame(
        result_rows,
        columns=RESULT_COLUMNS,
    ).sort_values(
        by=[
            "validation_mae",
            "validation_rmse",
            "candidate_name",
        ],
        kind="stable",
    ).reset_index(drop=True)


def run_totals_candidate_evaluation(
    database_file: Path = DATABASE_FILE,
) -> pd.DataFrame:
    """Run totals candidate evaluation from DuckDB."""

    validate_database_file(database_file)

    with duckdb.connect(
        str(database_file),
        read_only=True,
    ) as connection:
        development_data = (
            load_totals_development_data(
                connection
            )
        )

    results = evaluate_totals_model_candidates(
        development_data
    )

    logger.info(
        "Totals candidate evaluation completed on "
        "%s train and %s validation games.",
        int(results.iloc[0]["train_game_count"]),
        int(
            results.iloc[0][
                "validation_game_count"
            ]
        ),
    )

    return results


def main() -> None:
    """Run and print totals candidate evaluation."""

    results = (
        run_totals_candidate_evaluation()
    )

    print(
        results.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()