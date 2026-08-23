"""Backtest external pregame unit ratings for NFL totals."""

import logging
from pathlib import Path

import duckdb
import pandas as pd

from src.modeling.backtest_totals_model_candidates import (
    BACKTEST_VALIDATION_SEASONS,
)
from src.modeling.evaluate_spread_model_candidates import (
    calculate_regression_metrics,
    create_ridge_pipeline,
)
from src.modeling.evaluate_totals_model_candidates import (
    LEAGUE_SCORING_64_TOTALS_FEATURES,
    TOTALS_SELECTED_BASE_FEATURES,
    TOTALS_TARGET_COLUMN,
    load_totals_development_data,
    prepare_common_totals_sample,
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

BASE_FEATURES = (
    *TOTALS_SELECTED_BASE_FEATURES,
    *LEAGUE_SCORING_64_TOTALS_FEATURES,
)
UNIT_TOTAL_FEATURES = (
    "unit_offense_rating_sum",
    "unit_defense_rating_sum",
)
UNIT_DETAIL_FEATURES = (
    "unit_pass_offense_rating_sum",
    "unit_rush_offense_rating_sum",
    "unit_pass_defense_rating_sum",
    "unit_rush_defense_rating_sum",
    "unit_special_teams_rating_sum",
)
CANDIDATE_FEATURES = {
    "current_totals_base": BASE_FEATURES,
    "current_totals_plus_unit_totals": BASE_FEATURES + UNIT_TOTAL_FEATURES,
    "current_totals_plus_unit_detail": BASE_FEATURES + UNIT_DETAIL_FEATURES,
}
RIDGE_ALPHA = 100.0

SUMMARY_COLUMNS = (
    "candidate_name",
    "feature_count",
    "fold_count",
    "validation_game_count",
    "validation_mae",
    "validation_rmse",
    "validation_bias",
    "validation_r_squared",
)


def load_unit_totals_backtest_data(
    connection: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    """Load the established totals sample plus complete pregame units."""

    totals = prepare_common_totals_sample(
        load_totals_development_data(connection)
    )
    units = connection.execute(
        """
        SELECT
            dataset.game_id,
            home.off_value_pre + away.off_value_pre
                AS unit_offense_rating_sum,
            home.def_value_pre + away.def_value_pre
                AS unit_defense_rating_sum,
            home.pass_off_value_pre + away.pass_off_value_pre
                AS unit_pass_offense_rating_sum,
            home.rush_off_value_pre + away.rush_off_value_pre
                AS unit_rush_offense_rating_sum,
            home.pass_def_value_pre + away.pass_def_value_pre
                AS unit_pass_defense_rating_sum,
            home.rush_def_value_pre + away.rush_def_value_pre
                AS unit_rush_defense_rating_sum,
            home.st_off_value_pre + away.st_off_value_pre
                + home.st_def_value_pre + away.st_def_value_pre
                AS unit_special_teams_rating_sum
        FROM analytics.game_modeling_dataset AS dataset
        INNER JOIN processed.schedule AS schedule
            ON dataset.game_id = schedule.game_id
        INNER JOIN processed.external_nfelounits_units AS home
            ON dataset.season = home.season
           AND dataset.week = home.week
           AND dataset.home_team = home.team
        INNER JOIN processed.external_nfelounits_units AS away
            ON dataset.season = away.season
           AND dataset.week = away.week
           AND dataset.away_team = away.team
        WHERE schedule.game_type = 'REG'
          AND dataset.season < 2025
        """
    ).fetchdf()
    data = totals.merge(units, on="game_id", how="inner", validate="one_to_one")
    required = {feature for values in CANDIDATE_FEATURES.values() for feature in values}
    if data.empty or data[list(required)].isna().any().any():
        raise RuntimeError("Complete external-unit totals data is unavailable.")
    return data


def backtest_unit_totals_candidates(
    source_data: pd.DataFrame,
    validation_seasons: tuple[int, ...] = BACKTEST_VALIDATION_SEASONS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate candidates on chronological expanding-window folds."""

    prediction_rows: list[dict[str, object]] = []
    for validation_season in validation_seasons:
        train = source_data.loc[source_data["season"] < validation_season]
        validation = source_data.loc[source_data["season"] == validation_season]
        if train.empty or validation.empty:
            raise RuntimeError(f"Missing totals fold for {validation_season}.")
        for candidate_name, features in CANDIDATE_FEATURES.items():
            model = create_ridge_pipeline(ridge_alpha=RIDGE_ALPHA)
            model.fit(train.loc[:, features], train[TOTALS_TARGET_COLUMN])
            predicted = model.predict(validation.loc[:, features])
            for game_id, actual, estimate in zip(
                validation["game_id"],
                validation[TOTALS_TARGET_COLUMN],
                predicted,
                strict=True,
            ):
                prediction_rows.append(
                    {
                        "candidate_name": candidate_name,
                        "validation_season": validation_season,
                        "game_id": game_id,
                        "actual_total": float(actual),
                        "predicted_total": float(estimate),
                    }
                )

    predictions = pd.DataFrame(prediction_rows)
    summaries = []
    for candidate_name, features in CANDIDATE_FEATURES.items():
        candidate = predictions.loc[predictions["candidate_name"] == candidate_name]
        metrics = calculate_regression_metrics(
            actual_margin=candidate["actual_total"],
            predicted_margin=candidate["predicted_total"],
        )
        summaries.append(
            {
                "candidate_name": candidate_name,
                "feature_count": len(features),
                "fold_count": candidate["validation_season"].nunique(),
                "validation_game_count": len(candidate),
                "validation_mae": metrics["validation_mae"],
                "validation_rmse": metrics["validation_rmse"],
                "validation_bias": metrics["validation_bias"],
                "validation_r_squared": metrics["validation_r_squared"],
            }
        )
    summary = pd.DataFrame(summaries, columns=SUMMARY_COLUMNS).sort_values(
        ["validation_mae", "validation_rmse"], kind="stable"
    ).reset_index(drop=True)
    return summary, predictions


def run_unit_totals_backtest(
    database_file: Path = DATABASE_FILE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run totals candidate testing without opening holdout."""

    validate_database_file(database_file)
    with duckdb.connect(str(database_file), read_only=True) as connection:
        source_data = load_unit_totals_backtest_data(connection)
    results = backtest_unit_totals_candidates(source_data)
    logger.info(
        "External-unit totals backtest completed on %s games without "
        "opening holdout.", len(source_data)
    )
    return results


def main() -> None:
    """Run and print totals results."""

    summary, _ = run_unit_totals_backtest()
    print("\nEXTERNAL UNIT TOTALS BACKTEST SUMMARY\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
