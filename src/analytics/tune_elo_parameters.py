"""
NFL Analytics Platform
Elo Parameter Tuning

Purpose:
    Compare candidate Elo parameter combinations
    using time-based development, validation,
    and final holdout periods.

Author:
    Ferenc Kaizer

Version:
    0.1.0
"""

import logging
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import duckdb

from src.analytics.build_elo_ratings import (
    load_historical_games,
    validate_database_file,
    validate_source_columns,
    validate_source_table,
)

from src.analytics.evaluate_elo_model import (
    evaluate_probabilities,
)
from src.models.elo_history import (
    EloHistoryRecord,
    HistoricalGame,
    process_elo_history,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATABASE_FILE = PROJECT_ROOT / "data" / "nfl_analytics.duckdb"

SOURCE_SCHEMA = "processed"
SOURCE_TABLE = "schedule"
SOURCE_FULL_NAME = f"{SOURCE_SCHEMA}.{SOURCE_TABLE}"

BURN_IN_SEASON = 1999
DEVELOPMENT_START_SEASON = 2000
DEVELOPMENT_END_SEASON = 2022
VALIDATION_START_SEASON = 2023
VALIDATION_END_SEASON = 2024
HOLDOUT_SEASON = 2025

K_FACTOR_CANDIDATES = (
    15.0,
    20.0,
    25.0,
    30.0,
    35.0,
    40.0,
    45.0,
    50.0,
)

HOME_ADVANTAGE_CANDIDATES = (
    35.0,
    40.0,
    45.0,
    50.0,
    55.0,
)

SEASON_RETENTION_CANDIDATES = (
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
)


@dataclass(frozen=True)
class EloParameterSet:
    """Store one candidate Elo parameter combination."""

    k_factor: float
    home_advantage: float
    season_retention: float


@dataclass(frozen=True)
class EloPeriodMetrics:
    """Store evaluation metrics for one time period."""

    game_count: int
    brier_score: float
    log_loss: float
    accuracy: float


@dataclass(frozen=True)
class EloTuningResult:
    """Store one parameter set and its period results."""

    parameters: EloParameterSet
    development: EloPeriodMetrics
    validation: EloPeriodMetrics
    holdout: EloPeriodMetrics


def create_parameter_grid() -> list[EloParameterSet]:
    """Create every candidate Elo parameter combination."""

    return [
        EloParameterSet(
            k_factor=k_factor,
            home_advantage=home_advantage,
            season_retention=season_retention,
        )
        for (
            k_factor,
            home_advantage,
            season_retention,
        ) in product(
            K_FACTOR_CANDIDATES,
            HOME_ADVANTAGE_CANDIDATES,
            SEASON_RETENTION_CANDIDATES,
        )
    ]


def select_tuning_games(
    games: list[HistoricalGame],
) -> list[HistoricalGame]:
    """Validate season coverage and select the tuning period."""

    available_seasons = {
        game.season
        for game in games
    }

    required_seasons = set(
        range(
            BURN_IN_SEASON,
            HOLDOUT_SEASON + 1,
        )
    )
    missing_seasons = required_seasons - available_seasons

    if missing_seasons:
        missing_names = ", ".join(
            str(season)
            for season in sorted(missing_seasons)
        )
        raise RuntimeError(
            "Missing seasons required for Elo tuning: "
            f"{missing_names}"
        )

    return [
        game
        for game in games
        if BURN_IN_SEASON
        <= game.season
        <= HOLDOUT_SEASON
    ]


def load_tuning_games(
    database_file: Path = DATABASE_FILE,
) -> list[HistoricalGame]:
    """Load and validate historical games for parameter tuning."""

    validate_database_file(database_file)

    with duckdb.connect(
        str(database_file),
        read_only=True,
    ) as connection:
        validate_source_table(connection)
        validate_source_columns(connection)
        games = load_historical_games(connection)

    tuning_games = select_tuning_games(games)

    logger.info(
        "Elo tuning data loaded: %s games "
        "from seasons %s-%s.",
        len(tuning_games),
        BURN_IN_SEASON,
        HOLDOUT_SEASON,
    )

    return tuning_games


def calculate_period_metrics(
    records: list[EloHistoryRecord],
    start_season: int,
    end_season: int,
) -> EloPeriodMetrics:
    """Calculate Elo metrics for an inclusive season range."""

    period_records = [
        record
        for record in records
        if start_season
        <= record.season
        <= end_season
    ]

    if not period_records:
        raise ValueError(
            "No Elo records are available for seasons "
            f"{start_season}-{end_season}."
        )

    probabilities = [
        record.home_win_probability
        for record in period_records
    ]
    outcomes = [
        record.actual_home_score
        for record in period_records
    ]

    metrics = evaluate_probabilities(
        probabilities=probabilities,
        outcomes=outcomes,
    )

    return EloPeriodMetrics(
        game_count=metrics.game_count,
        brier_score=metrics.brier_score,
        log_loss=metrics.log_loss,
        accuracy=metrics.accuracy,
    )


def evaluate_parameter_set(
    games: list[HistoricalGame],
    parameters: EloParameterSet,
) -> EloTuningResult:
    """Evaluate one Elo parameter set across all time periods."""

    records, _ = process_elo_history(
        games=games,
        k_factor=parameters.k_factor,
        home_advantage=parameters.home_advantage,
        season_retention=parameters.season_retention,
    )

    development_metrics = calculate_period_metrics(
        records=records,
        start_season=DEVELOPMENT_START_SEASON,
        end_season=DEVELOPMENT_END_SEASON,
    )
    validation_metrics = calculate_period_metrics(
        records=records,
        start_season=VALIDATION_START_SEASON,
        end_season=VALIDATION_END_SEASON,
    )
    holdout_metrics = calculate_period_metrics(
        records=records,
        start_season=HOLDOUT_SEASON,
        end_season=HOLDOUT_SEASON,
    )

    return EloTuningResult(
        parameters=parameters,
        development=development_metrics,
        validation=validation_metrics,
        holdout=holdout_metrics,
    )


def evaluate_parameter_grid(
    games: list[HistoricalGame],
) -> list[EloTuningResult]:
    """Evaluate and rank every Elo parameter combination."""

    parameter_grid = create_parameter_grid()
    results = []

    logger.info(
        "Starting Elo parameter comparison: %s combinations.",
        len(parameter_grid),
    )

    for index, parameters in enumerate(
        parameter_grid,
        start=1,
    ):
        logger.info(
            "Evaluating combination %s/%s: "
            "K=%.1f, HFA=%.1f, retention=%.2f",
            index,
            len(parameter_grid),
            parameters.k_factor,
            parameters.home_advantage,
            parameters.season_retention,
        )

        result = evaluate_parameter_set(
            games=games,
            parameters=parameters,
        )
        results.append(result)

    return sorted(
        results,
        key=lambda result: (
            result.development.brier_score,
            result.development.log_loss,
            result.parameters.k_factor,
            result.parameters.home_advantage,
            result.parameters.season_retention,
        ),
    )


def select_best_result(
    results: list[EloTuningResult],
) -> EloTuningResult:
    """Select the best result using development data only."""

    if not results:
        raise ValueError(
            "At least one Elo tuning result is required."
        )

    return min(
        results,
        key=lambda result: (
            result.development.brier_score,
            result.development.log_loss,
        ),
    )


def log_top_development_results(
    results: list[EloTuningResult],
    limit: int = 10,
) -> None:
    """Log the strongest parameter sets on development data."""

    logger.info(
        "Top Elo parameter sets on development data:"
    )

    for rank, result in enumerate(
        results[:limit],
        start=1,
    ):
        parameters = result.parameters
        metrics = result.development

        logger.info(
            "Rank %s | "
            "K=%.1f | "
            "HFA=%.1f | "
            "Retention=%.2f | "
            "Games=%s | "
            "Brier=%.6f | "
            "Log loss=%.6f | "
            "Accuracy=%.2f%%",
            rank,
            parameters.k_factor,
            parameters.home_advantage,
            parameters.season_retention,
            metrics.game_count,
            metrics.brier_score,
            metrics.log_loss,
            metrics.accuracy * 100.0,
        )


def log_selected_result(
    result: EloTuningResult,
) -> None:
    """Log the selected parameters and untouched period results."""

    parameters = result.parameters

    logger.info(
        "Selected Elo parameters: "
        "K=%.1f, HFA=%.1f, retention=%.2f",
        parameters.k_factor,
        parameters.home_advantage,
        parameters.season_retention,
    )

    for period_name, metrics in (
        ("Development", result.development),
        ("Validation", result.validation),
        ("Holdout", result.holdout),
    ):
        logger.info(
            "%s | "
            "Games=%s | "
            "Brier=%.6f | "
            "Log loss=%.6f | "
            "Accuracy=%.2f%%",
            period_name,
            metrics.game_count,
            metrics.brier_score,
            metrics.log_loss,
            metrics.accuracy * 100.0,
        )


def tune_elo_parameters(
    database_file: Path = DATABASE_FILE,
) -> EloTuningResult:
    """Run the complete Elo parameter comparison."""

    logger.info("Starting Elo parameter tuning...")

    games = load_tuning_games(
        database_file=database_file,
    )
    results = evaluate_parameter_grid(
        games=games,
    )
    best_result = select_best_result(
        results=results,
    )

    log_top_development_results(
        results=results,
    )
    log_selected_result(
        result=best_result,
    )

    return best_result


def main() -> None:
    """Run Elo parameter tuning."""

    try:
        tune_elo_parameters()

        logger.info(
            "Elo parameter tuning completed successfully."
        )
    except Exception:
        logger.exception(
            "Elo parameter tuning failed."
        )
        raise


if __name__ == "__main__":
    main()