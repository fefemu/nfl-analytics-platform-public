"""
NFL Analytics Platform
Elo Model Evaluation

Purpose:
    Evaluate the predictive performance
    of historical Elo win probabilities.

Author:
    Ferenc Kaizer

Version:
    0.1.0
"""

import logging
from pathlib import Path
from math import log
from dataclasses import dataclass

import duckdb


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATABASE_FILE = PROJECT_ROOT / "data" / "nfl_analytics.duckdb"

SOURCE_SCHEMA = "analytics"
SOURCE_TABLE = "elo_game_predictions"
SOURCE_FULL_NAME = f"{SOURCE_SCHEMA}.{SOURCE_TABLE}"

PROBABILITY_EPSILON = 0.000000000000001

@dataclass(frozen=True)
class EloEvaluationMetrics:
    """Store Elo model metrics and baseline comparisons."""

    game_count: int
    tie_count: int
    accuracy: float
    brier_score: float
    log_loss: float
    equal_probability_brier_score: float
    equal_probability_log_loss: float
    home_team_accuracy: float

@dataclass(frozen=True)
class SeasonEvaluation:
    """Store model evaluation metrics for one NFL season."""

    season: int
    metrics: EloEvaluationMetrics

@dataclass(frozen=True)
class CalibrationBin:
    """Store calibration statistics for one probability range."""

    lower_bound: float
    upper_bound: float
    game_count: int
    average_probability: float
    actual_home_result_rate: float
    calibration_gap: float


def calculate_brier_score(
    probabilities: list[float],
    outcomes: list[float],
) -> float:
    """Calculate the mean squared error of predicted probabilities."""

    if not probabilities:
        raise ValueError(
            "At least one probability is required."
        )

    if len(probabilities) != len(outcomes):
        raise ValueError(
            "Probabilities and outcomes must have equal length."
        )

    squared_errors = [
        (probability - outcome) ** 2
        for probability, outcome in zip(
            probabilities,
            outcomes,
            strict=True,
        )
    ]

    return sum(squared_errors) / len(squared_errors)


def calculate_log_loss(
    probabilities: list[float],
    outcomes: list[float],
) -> float:
    """Calculate binary cross-entropy loss."""

    if not probabilities:
        raise ValueError(
            "At least one probability is required."
        )

    if len(probabilities) != len(outcomes):
        raise ValueError(
            "Probabilities and outcomes must have equal length."
        )

    losses = []

    for probability, outcome in zip(
        probabilities,
        outcomes,
        strict=True,
    ):
        if not 0.0 <= probability <= 1.0:
            raise ValueError(
                "Probabilities must be between 0 and 1."
            )

        clipped_probability = min(
            max(
                probability,
                PROBABILITY_EPSILON,
            ),
            1.0 - PROBABILITY_EPSILON,
        )

        loss = -(
            outcome * log(clipped_probability)
            + (1.0 - outcome)
            * log(1.0 - clipped_probability)
        )

        losses.append(loss)

    return sum(losses) / len(losses)


def calculate_accuracy(
    probabilities: list[float],
    outcomes: list[float],
) -> float:
    """Calculate winner prediction accuracy, excluding tied games."""

    if not probabilities:
        raise ValueError(
            "At least one probability is required."
        )

    if len(probabilities) != len(outcomes):
        raise ValueError(
            "Probabilities and outcomes must have equal length."
        )

    correct_predictions = 0
    evaluated_games = 0

    for probability, outcome in zip(
        probabilities,
        outcomes,
        strict=True,
    ):
        if outcome == 0.5:
            continue

        predicted_home_win = probability >= 0.5
        actual_home_win = outcome == 1.0

        if predicted_home_win == actual_home_win:
            correct_predictions += 1

        evaluated_games += 1

    if evaluated_games == 0:
        raise ValueError(
            "At least one non-tied game is required."
        )

    return correct_predictions / evaluated_games


def evaluate_probabilities(
    probabilities: list[float],
    outcomes: list[float],
) -> EloEvaluationMetrics:
    """Evaluate Elo probabilities against simple baselines."""

    equal_probabilities = [
        0.5
        for _ in probabilities
    ]
    home_team_probabilities = [
        1.0
        for _ in probabilities
    ]

    return EloEvaluationMetrics(
        game_count=len(probabilities),
        tie_count=sum(
            outcome == 0.5
            for outcome in outcomes
        ),
        accuracy=calculate_accuracy(
            probabilities=probabilities,
            outcomes=outcomes,
        ),
        brier_score=calculate_brier_score(
            probabilities=probabilities,
            outcomes=outcomes,
        ),
        log_loss=calculate_log_loss(
            probabilities=probabilities,
            outcomes=outcomes,
        ),
        equal_probability_brier_score=(
            calculate_brier_score(
                probabilities=equal_probabilities,
                outcomes=outcomes,
            )
        ),
        equal_probability_log_loss=(
            calculate_log_loss(
                probabilities=equal_probabilities,
                outcomes=outcomes,
            )
        ),
        home_team_accuracy=calculate_accuracy(
            probabilities=home_team_probabilities,
            outcomes=outcomes,
        ),
    )


def evaluate_probabilities_by_season(
    seasons: list[int],
    probabilities: list[float],
    outcomes: list[float],
) -> list[SeasonEvaluation]:
    """Evaluate Elo predictions separately for every season."""

    if not seasons:
        raise ValueError(
            "At least one season value is required."
        )

    if not (
        len(seasons)
        == len(probabilities)
        == len(outcomes)
    ):
        raise ValueError(
            "Seasons, probabilities, and outcomes "
            "must have equal length."
        )

    grouped_data: dict[
        int,
        tuple[list[float], list[float]],
    ] = {}

    for season, probability, outcome in zip(
        seasons,
        probabilities,
        outcomes,
        strict=True,
    ):
        if season not in grouped_data:
            grouped_data[season] = (
                [],
                [],
            )

        season_probabilities, season_outcomes = (
            grouped_data[season]
        )

        season_probabilities.append(probability)
        season_outcomes.append(outcome)

    return [
        SeasonEvaluation(
            season=season,
            metrics=evaluate_probabilities(
                probabilities=season_probabilities,
                outcomes=season_outcomes,
            ),
        )
        for season, (
            season_probabilities,
            season_outcomes,
        ) in sorted(grouped_data.items())
    ]


def validate_database_file(
    database_file: Path = DATABASE_FILE,
) -> None:
    """Validate that the DuckDB database file exists."""

    if not database_file.is_file():
        raise FileNotFoundError(
            f"DuckDB database file does not exist: {database_file}"
        )


def validate_source_table(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Validate that the Elo prediction source table exists."""

    table_exists = connection.execute(
        """
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_schema = ?
          AND table_name = ?
        """,
        [SOURCE_SCHEMA, SOURCE_TABLE],
    ).fetchone()[0]

    if table_exists == 0:
        raise RuntimeError(
            f"Source table does not exist: {SOURCE_FULL_NAME}"
        )


def load_evaluation_data(
    connection: duckdb.DuckDBPyConnection,
) -> tuple[
    list[int],
    list[float],
    list[float],
    int,
]:
    """Load Elo predictions after the initial burn-in season."""

    first_season = connection.execute(
        f"""
        SELECT MIN(season)
        FROM {SOURCE_FULL_NAME}
        """
    ).fetchone()[0]

    if first_season is None:
        raise RuntimeError(
            "No Elo predictions are available for evaluation."
        )

    rows = connection.execute(
        f"""
        SELECT
            season,
            home_win_probability,
            actual_home_score
        FROM {SOURCE_FULL_NAME}
        WHERE season > ?
        ORDER BY
            gameday,
            game_id
        """,
        [first_season],
    ).fetchall()

    if not rows:
        raise RuntimeError(
            "No Elo predictions remain after the burn-in season."
        )

    seasons = [
        row[0]
        for row in rows
    ]
    probabilities = [
        row[1]
        for row in rows
    ]
    outcomes = [
        row[2]
        for row in rows
    ]

    evaluation_start_season = first_season + 1

    logger.info(
        "Elo evaluation data loaded: %s games, "
        "starting with season %s.",
        len(rows),
        evaluation_start_season,
    )

    return (
        seasons,
        probabilities,
        outcomes,
        evaluation_start_season,
    )


def calculate_calibration_bins(
    probabilities: list[float],
    outcomes: list[float],
    bin_count: int = 10,
) -> list[CalibrationBin]:
    """Group predictions into probability calibration bins."""

    if not probabilities:
        raise ValueError(
            "At least one probability is required."
        )

    if len(probabilities) != len(outcomes):
        raise ValueError(
            "Probabilities and outcomes must have equal length."
        )

    if bin_count < 1:
        raise ValueError(
            "Calibration bin count must be at least one."
        )

    grouped_predictions: list[list[tuple[float, float]]] = [
        []
        for _ in range(bin_count)
    ]

    for probability, outcome in zip(
        probabilities,
        outcomes,
        strict=True,
    ):
        if not 0.0 <= probability <= 1.0:
            raise ValueError(
                "Probabilities must be between 0 and 1."
            )

        bin_index = min(
            int(probability * bin_count),
            bin_count - 1,
        )

        grouped_predictions[bin_index].append(
            (
                probability,
                outcome,
            )
        )

    calibration_bins = []

    for bin_index, observations in enumerate(
        grouped_predictions
    ):
        if not observations:
            continue

        average_probability = sum(
            probability
            for probability, _ in observations
        ) / len(observations)

        actual_home_result_rate = sum(
            outcome
            for _, outcome in observations
        ) / len(observations)

        lower_bound = bin_index / bin_count
        upper_bound = (bin_index + 1) / bin_count

        calibration_bins.append(
            CalibrationBin(
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                game_count=len(observations),
                average_probability=average_probability,
                actual_home_result_rate=(
                    actual_home_result_rate
                ),
                calibration_gap=(
                    actual_home_result_rate
                    - average_probability
                ),
            )
        )

    return calibration_bins


def evaluate_elo_model(
    database_file: Path = DATABASE_FILE,
) -> tuple[
    EloEvaluationMetrics,
    int,
    list[CalibrationBin],
    list[SeasonEvaluation],
]:
    """Evaluate historical Elo predictions from DuckDB."""

    validate_database_file(database_file)

    logger.info("Starting Elo model evaluation...")

    with duckdb.connect(
        str(database_file),
        read_only=True,
    ) as connection:
        validate_source_table(connection)

        (
            seasons,
            probabilities,
            outcomes,
            evaluation_start_season,
        ) = load_evaluation_data(connection)

    metrics = evaluate_probabilities(
        probabilities=probabilities,
        outcomes=outcomes,
    )
    calibration_bins = calculate_calibration_bins(
        probabilities=probabilities,
        outcomes=outcomes,
    )
    season_results = evaluate_probabilities_by_season(
        seasons=seasons,
        probabilities=probabilities,
        outcomes=outcomes,
    )

    return (
        metrics,
        evaluation_start_season,
        calibration_bins,
        season_results,
    )


def log_evaluation_results(
    metrics: EloEvaluationMetrics,
    evaluation_start_season: int,
) -> None:
    """Log Elo model metrics and baseline comparisons."""

    logger.info(
        "Evaluation period starts with season: %s",
        evaluation_start_season,
    )
    logger.info(
        "Games evaluated: %s",
        metrics.game_count,
    )
    logger.info(
        "Tied games: %s",
        metrics.tie_count,
    )
    logger.info(
        "Elo accuracy: %.2f%%",
        metrics.accuracy * 100.0,
    )
    logger.info(
        "Always-home accuracy: %.2f%%",
        metrics.home_team_accuracy * 100.0,
    )
    logger.info(
        "Elo Brier score: %.6f",
        metrics.brier_score,
    )
    logger.info(
        "50-50 baseline Brier score: %.6f",
        metrics.equal_probability_brier_score,
    )
    logger.info(
        "Elo log loss: %.6f",
        metrics.log_loss,
    )
    logger.info(
        "50-50 baseline log loss: %.6f",
        metrics.equal_probability_log_loss,
    )
    logger.info(
        "Brier improvement over baseline: %.6f",
        (
            metrics.equal_probability_brier_score
            - metrics.brier_score
        ),
    )
    logger.info(
        "Log-loss improvement over baseline: %.6f",
        (
            metrics.equal_probability_log_loss
            - metrics.log_loss
        ),
    )

def log_calibration_results(
    calibration_bins: list[CalibrationBin],
) -> None:
    """Log probability calibration statistics."""

    logger.info("Elo probability calibration:")

    for calibration_bin in calibration_bins:
        logger.info(
            "Range %.0f%%-%.0f%% | "
            "Games: %s | "
            "Predicted: %.2f%% | "
            "Actual: %.2f%% | "
            "Gap: %+.2f percentage points",
            calibration_bin.lower_bound * 100.0,
            calibration_bin.upper_bound * 100.0,
            calibration_bin.game_count,
            calibration_bin.average_probability * 100.0,
            (
                calibration_bin.actual_home_result_rate
                * 100.0
            ),
            calibration_bin.calibration_gap * 100.0,
        )



def log_season_results(
    season_results: list[SeasonEvaluation],
) -> None:
    """Log Elo evaluation metrics for every NFL season."""

    logger.info("Elo performance by season:")

    for season_result in season_results:
        metrics = season_result.metrics

        logger.info(
            "Season %s | "
            "Games: %s | "
            "Accuracy: %.2f%% | "
            "Home baseline: %.2f%% | "
            "Brier: %.6f | "
            "Log loss: %.6f",
            season_result.season,
            metrics.game_count,
            metrics.accuracy * 100.0,
            metrics.home_team_accuracy * 100.0,
            metrics.brier_score,
            metrics.log_loss,
        )

def main() -> None:
    """Run the Elo model evaluation."""

    try:
        (
            metrics,
            evaluation_start_season,
            calibration_bins,
            season_results,
        ) = evaluate_elo_model()

        log_evaluation_results(
            metrics=metrics,
            evaluation_start_season=(
                evaluation_start_season
            ),
        )
        log_calibration_results(
            calibration_bins=calibration_bins,
        )
        log_season_results(
            season_results=season_results,
        )

        logger.info(
            "Elo model evaluation completed successfully."
        )
    except Exception:
        logger.exception(
            "Elo model evaluation failed."
        )
        raise


if __name__ == "__main__":
    main()