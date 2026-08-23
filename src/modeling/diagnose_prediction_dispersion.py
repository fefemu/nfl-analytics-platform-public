"""
NFL Analytics Platform
Prediction Dispersion Diagnostics

Purpose:
    Measure whether production Spread, Total and season-win
    outputs are compressed toward the league average.

    This module is read-only. It does not change a production
    model, open a new holdout decision or persist database data.

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

from src.modeling.external_spread_holdout_component import (
    evaluate_locked_spread_holdout,
    load_spread_holdout_data,
)
from src.modeling.external_totals_holdout_component import (
    evaluate_locked_totals_routing_holdout,
    load_totals_holdout_data,
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


DISPERSION_COLUMNS = (
    "model_layer",
    "game_count",
    "actual_mean",
    "prediction_mean",
    "actual_standard_deviation",
    "prediction_standard_deviation",
    "prediction_to_actual_dispersion_ratio",
    "calibration_intercept",
    "calibration_slope",
    "actual_p10",
    "prediction_p10",
    "actual_p90",
    "prediction_p90",
)

EXTREME_COLUMNS = (
    "model_layer",
    "threshold",
    "actual_game_count",
    "predicted_game_count",
    "actual_game_rate",
    "predicted_game_rate",
)

SEASON_WIN_COLUMNS = (
    "comparison_group",
    "season_count",
    "team_season_count",
    "mean_wins",
    "standard_deviation",
    "minimum_wins",
    "maximum_wins",
    "p10_wins",
    "p90_wins",
)

MARKET_BENCHMARK_COLUMNS = (
    "model_layer",
    "game_count",
    "model_standard_deviation",
    "closing_market_standard_deviation",
    "model_to_market_dispersion_ratio",
    "market_on_model_intercept",
    "market_on_model_slope",
    "mean_absolute_model_market_difference",
)


def calculate_dispersion_summary(
    *,
    model_layer: str,
    actual_values: pd.Series | np.ndarray,
    predicted_values: pd.Series | np.ndarray,
) -> pd.DataFrame:
    """Calculate spread and calibration diagnostics."""

    actual = np.asarray(actual_values, dtype=float)
    predicted = np.asarray(predicted_values, dtype=float)

    if actual.shape != predicted.shape:
        raise ValueError(
            "Actual and predicted arrays must have "
            "the same shape."
        )

    valid_mask = np.isfinite(actual) & np.isfinite(predicted)
    actual = actual[valid_mask]
    predicted = predicted[valid_mask]

    if len(actual) < 2:
        raise ValueError(
            "At least two valid prediction pairs are required."
        )

    prediction_variance = float(np.var(predicted))

    if np.isclose(prediction_variance, 0.0):
        calibration_slope = np.nan
        calibration_intercept = np.nan
    else:
        calibration_slope = float(
            np.cov(
                predicted,
                actual,
                ddof=0,
            )[0, 1]
            / prediction_variance
        )
        calibration_intercept = float(
            np.mean(actual)
            - calibration_slope * np.mean(predicted)
        )

    actual_standard_deviation = float(
        np.std(actual, ddof=0)
    )
    prediction_standard_deviation = float(
        np.std(predicted, ddof=0)
    )

    dispersion_ratio = (
        prediction_standard_deviation
        / actual_standard_deviation
        if actual_standard_deviation > 0.0
        else np.nan
    )

    return pd.DataFrame(
        [
            {
                "model_layer": model_layer,
                "game_count": len(actual),
                "actual_mean": float(np.mean(actual)),
                "prediction_mean": float(np.mean(predicted)),
                "actual_standard_deviation": (
                    actual_standard_deviation
                ),
                "prediction_standard_deviation": (
                    prediction_standard_deviation
                ),
                "prediction_to_actual_dispersion_ratio": (
                    dispersion_ratio
                ),
                "calibration_intercept": calibration_intercept,
                "calibration_slope": calibration_slope,
                "actual_p10": float(np.quantile(actual, 0.10)),
                "prediction_p10": float(
                    np.quantile(predicted, 0.10)
                ),
                "actual_p90": float(np.quantile(actual, 0.90)),
                "prediction_p90": float(
                    np.quantile(predicted, 0.90)
                ),
            }
        ],
        columns=DISPERSION_COLUMNS,
    )


def calculate_extreme_rate_summary(
    *,
    model_layer: str,
    actual_values: pd.Series | np.ndarray,
    predicted_values: pd.Series | np.ndarray,
    thresholds: tuple[float, ...],
) -> pd.DataFrame:
    """Compare observed and predicted extreme-value rates."""

    actual = np.asarray(actual_values, dtype=float)
    predicted = np.asarray(predicted_values, dtype=float)
    valid_mask = np.isfinite(actual) & np.isfinite(predicted)
    actual = actual[valid_mask]
    predicted = predicted[valid_mask]

    if len(actual) == 0:
        raise ValueError("No valid prediction pairs are available.")

    rows = []

    for threshold in thresholds:
        actual_count = int(
            np.sum(np.abs(actual) >= threshold)
        )
        predicted_count = int(
            np.sum(np.abs(predicted) >= threshold)
        )
        rows.append(
            {
                "model_layer": model_layer,
                "threshold": float(threshold),
                "actual_game_count": actual_count,
                "predicted_game_count": predicted_count,
                "actual_game_rate": actual_count / len(actual),
                "predicted_game_rate": (
                    predicted_count / len(predicted)
                ),
            }
        )

    return pd.DataFrame(rows, columns=EXTREME_COLUMNS)


def load_historical_regular_season_wins(
    connection: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    """Load one row per historical team and regular season."""

    return connection.execute(
        """
        WITH team_results AS (
            SELECT
                season,
                home_team AS team,
                CASE
                    WHEN target_home_score > target_away_score
                        THEN 1
                    ELSE 0
                END AS win
            FROM analytics.game_modeling_dataset
            WHERE game_type = 'REG'

            UNION ALL

            SELECT
                season,
                away_team AS team,
                CASE
                    WHEN target_away_score > target_home_score
                        THEN 1
                    ELSE 0
                END AS win
            FROM analytics.game_modeling_dataset
            WHERE game_type = 'REG'
        )

        SELECT
            season,
            team,
            SUM(win)::DOUBLE AS wins
        FROM team_results
        GROUP BY season, team
        ORDER BY season, team
        """
    ).fetchdf()


def load_current_simulated_wins(
    connection: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    """Load current season expected wins."""

    return connection.execute(
        """
        SELECT
            season,
            team,
            expected_wins AS wins
        FROM analytics.current_season_simulation_summary
        ORDER BY team
        """
    ).fetchdf()


def load_current_market_expected_wins(
    connection: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    """Aggregate schedule-level no-vig market probabilities."""

    return connection.execute(
        """
        SELECT
            season,
            CASE
                WHEN outcome_type = 'home' THEN home_team
                WHEN outcome_type = 'away' THEN away_team
            END AS team,
            SUM(consensus_no_vig_probability) AS wins
        FROM analytics.current_moneyline_value
        WHERE outcome_type IN ('home', 'away')
        GROUP BY season, team
        ORDER BY team
        """
    ).fetchdf()


def load_closing_market_lines(
    connection: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    """Load closing spread and Total expectations by game."""

    return connection.execute(
        """
        SELECT
            normalized_game_id AS game_id,
            -CAST(home_line_close AS DOUBLE)
                AS closing_expected_home_margin,
            CAST(total_line_close AS DOUBLE)
                AS closing_expected_total
        FROM processed.external_nfelo_game_ratings
        WHERE home_line_close IS NOT NULL
           OR total_line_close IS NOT NULL
        """
    ).fetchdf()


def calculate_market_benchmark_summary(
    *,
    model_layer: str,
    model_values: pd.Series | np.ndarray,
    market_values: pd.Series | np.ndarray,
) -> pd.DataFrame:
    """Compare model expectations with closing market lines."""

    model = np.asarray(model_values, dtype=float)
    market = np.asarray(market_values, dtype=float)

    if model.shape != market.shape:
        raise ValueError(
            "Model and market arrays must have the same shape."
        )

    valid_mask = np.isfinite(model) & np.isfinite(market)
    model = model[valid_mask]
    market = market[valid_mask]

    if len(model) < 2:
        raise ValueError(
            "At least two model-market pairs are required."
        )

    model_standard_deviation = float(np.std(model, ddof=0))
    market_standard_deviation = float(np.std(market, ddof=0))
    model_variance = float(np.var(model))

    slope = float(
        np.cov(model, market, ddof=0)[0, 1]
        / model_variance
    )
    intercept = float(
        np.mean(market) - slope * np.mean(model)
    )

    return pd.DataFrame(
        [
            {
                "model_layer": model_layer,
                "game_count": len(model),
                "model_standard_deviation": (
                    model_standard_deviation
                ),
                "closing_market_standard_deviation": (
                    market_standard_deviation
                ),
                "model_to_market_dispersion_ratio": (
                    model_standard_deviation
                    / market_standard_deviation
                ),
                "market_on_model_intercept": intercept,
                "market_on_model_slope": slope,
                "mean_absolute_model_market_difference": float(
                    np.mean(np.abs(model - market))
                ),
            }
        ],
        columns=MARKET_BENCHMARK_COLUMNS,
    )


def summarize_season_wins(
    *,
    comparison_group: str,
    team_seasons: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize the cross-team season-win distribution."""

    required_columns = {"season", "team", "wins"}
    missing_columns = sorted(
        required_columns - set(team_seasons.columns)
    )

    if missing_columns:
        raise ValueError(
            "Season-win data is missing columns: "
            + ", ".join(missing_columns)
        )

    wins = pd.to_numeric(
        team_seasons["wins"],
        errors="coerce",
    ).dropna().to_numpy(dtype=float)

    if len(wins) == 0:
        raise ValueError("Season-win data is empty.")

    return pd.DataFrame(
        [
            {
                "comparison_group": comparison_group,
                "season_count": int(
                    team_seasons["season"].nunique()
                ),
                "team_season_count": len(wins),
                "mean_wins": float(np.mean(wins)),
                "standard_deviation": float(
                    np.std(wins, ddof=0)
                ),
                "minimum_wins": float(np.min(wins)),
                "maximum_wins": float(np.max(wins)),
                "p10_wins": float(np.quantile(wins, 0.10)),
                "p90_wins": float(np.quantile(wins, 0.90)),
            }
        ],
        columns=SEASON_WIN_COLUMNS,
    )


def run_prediction_dispersion_diagnostics(
    database_file: Path = DATABASE_FILE,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Run all read-only production dispersion diagnostics."""

    validate_database_file(database_file)

    with duckdb.connect(
        str(database_file),
        read_only=True,
    ) as connection:
        spread_source = load_spread_holdout_data(connection)
        totals_source = load_totals_holdout_data(connection)
        historical_wins = load_historical_regular_season_wins(
            connection
        )
        simulated_wins = load_current_simulated_wins(connection)
        market_expected_wins = load_current_market_expected_wins(
            connection
        )
        closing_market_lines = load_closing_market_lines(connection)

    _, spread_predictions = evaluate_locked_spread_holdout(
        spread_source
    )
    _, totals_predictions = (
        evaluate_locked_totals_routing_holdout(totals_source)
    )

    spread_dispersion = calculate_dispersion_summary(
        model_layer="SPREAD",
        actual_values=spread_predictions["actual_home_margin"],
        predicted_values=spread_predictions[
            "external_predicted_home_margin"
        ],
    )
    totals_dispersion = calculate_dispersion_summary(
        model_layer="TOTALS",
        actual_values=totals_predictions["actual_total"],
        predicted_values=totals_predictions[
            "current_predicted_total"
        ],
    )
    dispersion_summary = pd.concat(
        [spread_dispersion, totals_dispersion],
        ignore_index=True,
    )

    extreme_summary = pd.concat(
        [
            calculate_extreme_rate_summary(
                model_layer="SPREAD_ABSOLUTE_MARGIN",
                actual_values=spread_predictions[
                    "actual_home_margin"
                ],
                predicted_values=spread_predictions[
                    "external_predicted_home_margin"
                ],
                thresholds=(3.0, 7.0, 10.0, 14.0),
            ),
            calculate_extreme_rate_summary(
                model_layer="TOTALS_DISTANCE_FROM_45",
                actual_values=(
                    totals_predictions["actual_total"] - 45.0
                ),
                predicted_values=(
                    totals_predictions[
                        "current_predicted_total"
                    ]
                    - 45.0
                ),
                thresholds=(3.0, 7.0, 10.0, 14.0),
            ),
        ],
        ignore_index=True,
    )

    spread_market_comparison = spread_predictions.merge(
        closing_market_lines.loc[
            :,
            ["game_id", "closing_expected_home_margin"],
        ],
        on="game_id",
        how="inner",
        validate="one_to_one",
    )
    totals_market_comparison = totals_predictions.merge(
        closing_market_lines.loc[
            :,
            ["game_id", "closing_expected_total"],
        ],
        on="game_id",
        how="inner",
        validate="one_to_one",
    )

    market_benchmark_summary = pd.concat(
        [
            calculate_market_benchmark_summary(
                model_layer="SPREAD",
                model_values=spread_market_comparison[
                    "external_predicted_home_margin"
                ],
                market_values=spread_market_comparison[
                    "closing_expected_home_margin"
                ],
            ),
            calculate_market_benchmark_summary(
                model_layer="TOTALS",
                model_values=totals_market_comparison[
                    "current_predicted_total"
                ],
                market_values=totals_market_comparison[
                    "closing_expected_total"
                ],
            ),
        ],
        ignore_index=True,
    )

    season_win_summary = pd.concat(
        [
            summarize_season_wins(
                comparison_group=(
                    "HISTORICAL_ACTUAL_REGULAR_SEASONS"
                ),
                team_seasons=historical_wins,
            ),
            summarize_season_wins(
                comparison_group=(
                    "CURRENT_MONTE_CARLO_EXPECTED_WINS"
                ),
                team_seasons=simulated_wins,
            ),
            summarize_season_wins(
                comparison_group=(
                    "CURRENT_MARKET_EXPECTED_WINS"
                ),
                team_seasons=market_expected_wins,
            ),
        ],
        ignore_index=True,
    )

    logger.info(
        "Prediction dispersion diagnostics completed: "
        "%s Spread games, %s Totals games.",
        len(spread_predictions),
        len(totals_predictions),
    )

    return (
        dispersion_summary,
        extreme_summary,
        market_benchmark_summary,
        season_win_summary,
    )


def main() -> None:
    """Run and print prediction dispersion diagnostics."""

    (
        dispersion_summary,
        extreme_summary,
        market_benchmark_summary,
        season_win_summary,
    ) = run_prediction_dispersion_diagnostics()

    print("\nPREDICTION DISPERSION SUMMARY\n")
    print(dispersion_summary.to_string(index=False))

    print("\nEXTREME OUTCOME SUMMARY\n")
    print(extreme_summary.to_string(index=False))

    print("\nCLOSING MARKET DISPERSION BENCHMARK\n")
    print(market_benchmark_summary.to_string(index=False))

    print("\nSEASON WIN DISPERSION SUMMARY\n")
    print(season_win_summary.to_string(index=False))


if __name__ == "__main__":
    main()
