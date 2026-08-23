"""Backtest external unit and preseason win-total signals."""

import logging
from pathlib import Path

import duckdb
import pandas as pd

from src.modeling.backtest_elo_rating_sources import (
    BACKTEST_VALIDATION_SEASONS,
)
from src.modeling.train_logistic_baseline import (
    DATABASE_FILE,
    TARGET_COLUMN,
    create_logistic_pipeline,
    evaluate_probabilities,
    validate_database_file,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

BASE_FEATURES = (
    "external_nfelo_rating_difference",
    "external_nfelo_qb_adjustment_difference",
)
UNIT_FEATURES = (
    "unit_offense_rating_difference",
    "unit_defense_rating_difference",
)
WIN_TOTAL_FEATURES = ("win_total_elo_difference",)

CANDIDATE_FEATURES = {
    "external_nfelo_qb": BASE_FEATURES,
    "external_nfelo_qb_units": BASE_FEATURES + UNIT_FEATURES,
    "external_nfelo_qb_win_total": BASE_FEATURES + WIN_TOTAL_FEATURES,
    "external_nfelo_qb_units_win_total": (
        BASE_FEATURES + UNIT_FEATURES + WIN_TOTAL_FEATURES
    ),
}

SUMMARY_COLUMNS = (
    "candidate_name",
    "feature_count",
    "fold_count",
    "validation_game_count",
    "accuracy",
    "brier_score",
    "log_loss",
    "probability_standard_deviation",
)

SEASON_COLUMNS = (
    "candidate_name",
    "validation_season",
    "training_game_count",
    "validation_game_count",
    "accuracy",
    "brier_score",
    "log_loss",
    "probability_standard_deviation",
)


def load_team_strength_backtest_data(
    connection: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    """Load fully covered regular-season development games only."""

    data = connection.execute(
        """
        SELECT
            dataset.game_id,
            dataset.season,
            dataset.game_date,
            CAST(dataset.target_home_win AS INTEGER)
                AS target_home_win,
            external.starting_nfelo_home
                - external.starting_nfelo_away
                AS external_nfelo_rating_difference,
            external.home_538_qb_adj
                - external.away_538_qb_adj
                AS external_nfelo_qb_adjustment_difference,
            home_units.off_value_pre
                - away_units.off_value_pre
                AS unit_offense_rating_difference,
            home_units.def_value_pre
                - away_units.def_value_pre
                AS unit_defense_rating_difference,
            home_win_total.wt_rating_elo
                - away_win_total.wt_rating_elo
                AS win_total_elo_difference
        FROM analytics.game_modeling_dataset AS dataset
        INNER JOIN processed.schedule AS schedule
            ON dataset.game_id = schedule.game_id
        INNER JOIN processed.external_nfelo_game_ratings AS external
            ON dataset.game_id = external.normalized_game_id
        INNER JOIN processed.external_nfelounits_units AS home_units
            ON dataset.season = home_units.season
           AND dataset.week = home_units.week
           AND dataset.home_team = home_units.team
        INNER JOIN processed.external_nfelounits_units AS away_units
            ON dataset.season = away_units.season
           AND dataset.week = away_units.week
           AND dataset.away_team = away_units.team
        INNER JOIN processed.external_win_total_ratings AS home_win_total
            ON dataset.season = home_win_total.season
           AND dataset.home_team = home_win_total.team
        INNER JOIN processed.external_win_total_ratings AS away_win_total
            ON dataset.season = away_win_total.season
           AND dataset.away_team = away_win_total.team
        WHERE schedule.game_type = 'REG'
          AND dataset.season < 2025
          AND dataset.target_home_win IS NOT NULL
        ORDER BY dataset.game_date, dataset.game_id
        """
    ).fetchdf()

    if data.empty:
        raise RuntimeError("No team-strength backtest games were loaded.")
    if data["game_id"].duplicated().any():
        raise RuntimeError("Team-strength backtest data contains duplicates.")
    if data.isna().any().any():
        raise RuntimeError("Team-strength backtest data contains null values.")
    if int(data["season"].max()) >= 2025:
        raise RuntimeError("Protected holdout leaked into team-strength data.")

    return data


def backtest_team_strength_candidates(
    source_data: pd.DataFrame,
    validation_seasons: tuple[int, ...] = BACKTEST_VALIDATION_SEASONS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run expanding-window folds for every candidate."""

    rows: list[dict[str, object]] = []
    for validation_season in validation_seasons:
        train = source_data.loc[
            source_data["season"] < validation_season
        ]
        validation = source_data.loc[
            source_data["season"] == validation_season
        ]
        if train.empty or validation.empty:
            raise RuntimeError(
                f"Missing fold data for validation season {validation_season}."
            )

        for candidate_name, features in CANDIDATE_FEATURES.items():
            model = create_logistic_pipeline(
                feature_columns=features,
                regularization_c=1.0,
            )
            model.fit(train.loc[:, features], train[TARGET_COLUMN])
            probabilities = model.predict_proba(
                validation.loc[:, features]
            )[:, 1]
            metrics = evaluate_probabilities(
                validation[TARGET_COLUMN], probabilities
            )
            rows.append(
                {
                    "candidate_name": candidate_name,
                    "validation_season": validation_season,
                    "training_game_count": len(train),
                    "validation_game_count": len(validation),
                    "accuracy": metrics.accuracy,
                    "brier_score": metrics.brier_score,
                    "log_loss": metrics.log_loss,
                    "probability_standard_deviation": float(
                        pd.Series(probabilities).std(ddof=0)
                    ),
                }
            )

    season_results = pd.DataFrame(rows, columns=SEASON_COLUMNS)
    summary_rows = []
    for candidate_name, features in CANDIDATE_FEATURES.items():
        candidate = season_results.loc[
            season_results["candidate_name"] == candidate_name
        ]
        weights = candidate["validation_game_count"]
        summary_rows.append(
            {
                "candidate_name": candidate_name,
                "feature_count": len(features),
                "fold_count": len(candidate),
                "validation_game_count": int(weights.sum()),
                **{
                    column: float(
                        (candidate[column] * weights).sum() / weights.sum()
                    )
                    for column in (
                        "accuracy",
                        "brier_score",
                        "log_loss",
                        "probability_standard_deviation",
                    )
                },
            }
        )

    summary = pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)
    summary = summary.sort_values(
        ["brier_score", "log_loss"], kind="stable"
    ).reset_index(drop=True)
    return summary, season_results


def run_team_strength_backtest(
    database_file: Path = DATABASE_FILE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and backtest without accessing the 2025 holdout."""

    validate_database_file(database_file)
    with duckdb.connect(str(database_file), read_only=True) as connection:
        source_data = load_team_strength_backtest_data(connection)
    results = backtest_team_strength_candidates(source_data)
    logger.info(
        "External team-strength backtest completed on %s regular-season "
        "development games without opening holdout.",
        len(source_data),
    )
    return results


def main() -> None:
    """Run and print the candidate comparison."""

    summary, season_results = run_team_strength_backtest()
    print("\nEXTERNAL TEAM-STRENGTH BACKTEST SUMMARY\n")
    print(summary.to_string(index=False))
    print("\nSEASON-LEVEL RESULTS\n")
    print(season_results.to_string(index=False))


if __name__ == "__main__":
    main()
