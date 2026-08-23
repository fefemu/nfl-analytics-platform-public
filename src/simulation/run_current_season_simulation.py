"""
NFL Analytics Platform
Current Season Simulation Runner

Purpose:
    Load the latest upcoming regular-season schedule and run
    the dynamic Elo Monte Carlo simulation.

Author:
    Ferenc Kaizer

Version:
    0.1.0
"""

import argparse
import logging
from pathlib import Path
from time import perf_counter

import duckdb
import numpy as np
import pandas as pd

from src.modeling.build_current_game_predictions import (
    TARGET_FULL_NAME as PREDICTION_FULL_NAME,
)
from src.modeling.train_logistic_baseline import (
    DATABASE_FILE,
    validate_database_file,
)
from src.simulation.run_elo_monte_carlo import (
    DEFAULT_RANDOM_SEED,
    DEFAULT_SIMULATION_COUNT,
    MonteCarloSimulationResult,
    run_elo_monte_carlo,
)
from src.simulation.simulate_elo_season import (
    extract_initial_ratings,
)
from src.processing.build_processed_schedule import (
    TARGET_FULL_NAME as SCHEDULE_FULL_NAME,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


REQUIRED_SIMULATION_COLUMNS = {
    "game_id",
    "season",
    "game_type",
    "week",
    "gameday",
    "gametime",
    "home_team",
    "away_team",
    "is_neutral",
    "home_rating_pregame",
    "away_rating_pregame",
}

PRESEASON_RATING_DISPERSION_WARNING_THRESHOLD = 60.0


def evaluate_starting_rating_dispersion(
    schedule: pd.DataFrame,
    warning_threshold: float = (
        PRESEASON_RATING_DISPERSION_WARNING_THRESHOLD
    ),
) -> dict[str, float]:
    """Measure and warn about unusually compressed starting ratings."""

    if warning_threshold <= 0.0:
        raise ValueError("Rating dispersion threshold must be positive.")

    rating_map = extract_initial_ratings(schedule)
    ratings = np.asarray(list(rating_map.values()), dtype=float)
    metrics = {
        "team_count": float(len(ratings)),
        "mean_rating": float(ratings.mean()),
        "rating_standard_deviation": float(ratings.std(ddof=0)),
        "minimum_rating": float(ratings.min()),
        "maximum_rating": float(ratings.max()),
    }
    if metrics["rating_standard_deviation"] < warning_threshold:
        logger.warning(
            "Preseason starting ratings are unusually compressed: "
            "teams=%s | standard deviation=%.2f | range=%.1f-%.1f | "
            "warning threshold=%.1f.",
            int(metrics["team_count"]),
            metrics["rating_standard_deviation"],
            metrics["minimum_rating"],
            metrics["maximum_rating"],
            warning_threshold,
        )
    return metrics


def validate_prediction_source(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Validate the current prediction source table."""

    available_columns = {
        row[0]
        for row in connection.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'analytics'
              AND table_name
                  = 'current_game_predictions'
            """
        ).fetchall()
    }

    if not available_columns:
        raise RuntimeError(
            "Simulation source table does not exist: "
            f"{PREDICTION_FULL_NAME}"
        )

    missing_columns = sorted(
        REQUIRED_SIMULATION_COLUMNS
        - available_columns
    )

    if missing_columns:
        raise RuntimeError(
            "Simulation source is missing columns: "
            + ", ".join(missing_columns)
        )

    external_table_exists = connection.execute(
        """
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_schema = 'processed'
          AND table_name = 'external_nfelo_game_ratings'
        """
    ).fetchone()[0]

    if external_table_exists == 0:
        raise RuntimeError(
            "Simulation external Elo source table does not exist: "
            "processed.external_nfelo_game_ratings"
        )


def load_current_team_records(
    connection: duckdb.DuckDBPyConnection,
    season: int,
) -> pd.DataFrame:
    """Load completed regular-season team records."""

    records = connection.execute(
        f"""
        WITH team_games AS (
            SELECT
                home_team AS team,

                CASE
                    WHEN home_score > away_score
                        THEN 1
                    ELSE 0
                END AS win,

                CASE
                    WHEN home_score < away_score
                        THEN 1
                    ELSE 0
                END AS loss,

                CASE
                    WHEN home_score = away_score
                        THEN 1
                    ELSE 0
                END AS tie

            FROM {SCHEDULE_FULL_NAME}

            WHERE season = ?
              AND game_type = 'REG'
              AND is_completed = TRUE

            UNION ALL

            SELECT
                away_team AS team,

                CASE
                    WHEN away_score > home_score
                        THEN 1
                    ELSE 0
                END AS win,

                CASE
                    WHEN away_score < home_score
                        THEN 1
                    ELSE 0
                END AS loss,

                CASE
                    WHEN away_score = home_score
                        THEN 1
                    ELSE 0
                END AS tie

            FROM {SCHEDULE_FULL_NAME}

            WHERE season = ?
              AND game_type = 'REG'
              AND is_completed = TRUE
        )

        SELECT
            team,
            CAST(SUM(win) AS INTEGER) AS wins,
            CAST(SUM(loss) AS INTEGER) AS losses,
            CAST(SUM(tie) AS INTEGER) AS ties
        FROM team_games
        GROUP BY team
        ORDER BY team
        """,
        [
            season,
            season,
        ],
    ).fetchdf()

    completed_game_count = int(
        (
            records["wins"].sum()
            + records["losses"].sum()
            + records["ties"].sum()
        )
        / 2
    )

    logger.info(
        "Current team records loaded: season=%s | "
        "completed games=%s | teams with results=%s.",
        season,
        completed_game_count,
        len(records),
    )

    return records


def load_latest_regular_season_schedule(
    connection: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    """Load the latest upcoming regular-season schedule."""

    latest_season = connection.execute(
        f"""
        SELECT MAX(season)
        FROM {PREDICTION_FULL_NAME}
        WHERE game_type = 'REG'
        """
    ).fetchone()[0]

    if latest_season is None:
        raise RuntimeError(
            "No upcoming regular-season predictions "
            "are available."
        )

    schedule = connection.execute(
        f"""
        WITH external_team_ratings AS (
            SELECT
                source_season,
                source_week,
                home_team AS team,
                starting_nfelo_home AS rating
            FROM processed.external_nfelo_game_ratings

            UNION ALL

            SELECT
                source_season,
                source_week,
                away_team AS team,
                starting_nfelo_away AS rating
            FROM processed.external_nfelo_game_ratings
        ),

        latest_external_team_ratings AS (
            SELECT team, rating
            FROM external_team_ratings
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY team
                ORDER BY source_season DESC, source_week DESC
            ) = 1
        )

        SELECT
            predictions.game_id,
            predictions.season,
            predictions.game_type,
            predictions.week,
            predictions.gameday,
            predictions.gametime,
            predictions.home_team,
            predictions.away_team,
            predictions.is_neutral,
            predictions.home_win_probability,
            home_external.rating AS home_rating_pregame,
            away_external.rating AS away_rating_pregame
        FROM {PREDICTION_FULL_NAME} AS predictions
        INNER JOIN latest_external_team_ratings AS home_external
            ON predictions.home_team = home_external.team
        INNER JOIN latest_external_team_ratings AS away_external
            ON predictions.away_team = away_external.team
        WHERE predictions.season = ?
          AND predictions.game_type = 'REG'
        ORDER BY
            week,
            gameday,
            gametime,
            game_id
        """,
        [latest_season],
    ).fetchdf()

    validate_simulation_schedule(schedule)
    dispersion = evaluate_starting_rating_dispersion(schedule)

    logger.info(
        "Simulation schedule loaded: season=%s | "
        "games=%s | teams=%s.",
        latest_season,
        len(schedule),
        len(
            set(schedule["home_team"])
            | set(schedule["away_team"])
        ),
    )

    logger.info(
        "Simulation starting-rating dispersion: standard deviation=%.2f | "
        "range=%.1f-%.1f.",
        dispersion["rating_standard_deviation"],
        dispersion["minimum_rating"],
        dispersion["maximum_rating"],
    )

    return schedule


def validate_simulation_schedule(
    schedule: pd.DataFrame,
) -> None:
    """Validate a regular-season simulation schedule."""

    missing_columns = sorted(
        REQUIRED_SIMULATION_COLUMNS
        - set(schedule.columns)
    )

    if missing_columns:
        raise ValueError(
            "Simulation schedule is missing columns: "
            + ", ".join(missing_columns)
        )

    if schedule.empty:
        raise ValueError(
            "Simulation schedule must not be empty."
        )

    if schedule["game_id"].duplicated().any():
        raise ValueError(
            "Simulation schedule contains duplicate "
            "game identifiers."
        )

    if schedule["season"].nunique() != 1:
        raise ValueError(
            "Simulation schedule must contain exactly "
            "one season."
        )

    if (
        schedule["home_team"]
        == schedule["away_team"]
    ).any():
        raise ValueError(
            "Simulation schedule contains a self-matchup."
        )

    extract_initial_ratings(schedule)


def log_simulation_summary(
    result: MonteCarloSimulationResult,
) -> None:
    """Log the complete expected standings table."""

    logger.info(
        "%s season simulation results:",
        result.simulation_mode,
    )

    for rank, row in enumerate(
        result.team_summary.itertuples(
            index=False
        ),
        start=1,
    ):
        logger.info(
            "%s. %s | Expected record=%.2f-%.2f-%s | "
            "Median wins=%.1f | P10-P90=%.1f-%.1f | "
            "Most likely wins=%s | Expected final "
            "Elo=%.1f",
            rank,
            row.team,
            row.expected_wins,
            row.expected_losses,
            row.expected_ties,
            row.median_wins,
            row.p10_wins,
            row.p90_wins,
            row.most_likely_wins,
            row.expected_final_elo,
        )


def run_current_season_simulation(
    database_file: Path = DATABASE_FILE,
    simulation_count: int = (
        DEFAULT_SIMULATION_COUNT
    ),
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> MonteCarloSimulationResult:
    """Run the current regular-season simulation."""

    validate_database_file(database_file)

    logger.info(
        "Starting current season simulation: "
        "simulations=%s | seed=%s.",
        simulation_count,
        random_seed,
    )

    with duckdb.connect(
        str(database_file),
        read_only=True,
    ) as connection:
        validate_prediction_source(connection)

        schedule = (
            load_latest_regular_season_schedule(
                connection
            )
        )

        simulation_season = int(
            schedule["season"].iloc[0]
        )

        current_records = (
            load_current_team_records(
                connection=connection,
                season=simulation_season,
            )
        )

    started_at = perf_counter()

    result = run_elo_monte_carlo(
        schedule=schedule,
        simulation_count=simulation_count,
        random_seed=random_seed,
        current_records=current_records,
    )

    duration = perf_counter() - started_at

    log_simulation_summary(result)

    logger.info(
        "Current season simulation completed: "
        "%s simulations in %.2f seconds.",
        simulation_count,
        duration,
    )

    return result


def parse_arguments() -> argparse.Namespace:
    """Parse command-line simulation arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Run the dynamic Elo current-season "
            "simulation."
        )
    )

    parser.add_argument(
        "--simulations",
        type=int,
        default=DEFAULT_SIMULATION_COUNT,
        help="Number of Monte Carlo season simulations.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Random seed for reproducible results.",
    )

    return parser.parse_args()


def main() -> None:
    """Run the current season simulation entry point."""

    arguments = parse_arguments()

    try:
        run_current_season_simulation(
            simulation_count=(
                arguments.simulations
            ),
            random_seed=arguments.seed,
        )
    except Exception:
        logger.exception(
            "Current season simulation failed."
        )
        raise


if __name__ == "__main__":
    main()
