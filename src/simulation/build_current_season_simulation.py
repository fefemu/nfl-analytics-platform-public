"""
NFL Analytics Platform
Current Season Simulation Builder

Purpose:
    Run the production Frozen Elo Monte Carlo simulation and persist
    dashboard-ready season summary and win distributions.

Author:
    Ferenc Kaizer

Version:
    0.1.0
"""

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd

from src.modeling.production_model import (
    PRODUCTION_MODEL,
)
from src.modeling.train_logistic_baseline import (
    DATABASE_FILE,
    validate_database_file,
)
from src.simulation.run_current_elo_simulation_benchmark import (
    run_current_elo_simulation_benchmark,
)
from src.simulation.build_current_elo_simulation_benchmark import (
    create_benchmark_tables,
    prepare_benchmark_tables,
    validate_benchmark_tables,
)
from src.simulation.run_elo_monte_carlo import (
    DEFAULT_RANDOM_SEED,
    DEFAULT_SIMULATION_COUNT,
    MonteCarloSimulationResult,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


TARGET_SCHEMA = "analytics"

SUMMARY_TABLE = (
    "current_season_simulation_summary"
)
SUMMARY_FULL_NAME = (
    f"{TARGET_SCHEMA}.{SUMMARY_TABLE}"
)

DISTRIBUTION_TABLE = (
    "current_season_win_distribution"
)
DISTRIBUTION_FULL_NAME = (
    f"{TARGET_SCHEMA}.{DISTRIBUTION_TABLE}"
)


def prepare_simulation_tables(
    result: MonteCarloSimulationResult,
    generated_at: datetime | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add reproducibility metadata to simulation outputs."""

    if generated_at is None:
        generated_at = datetime.now(
            timezone.utc
        ).replace(tzinfo=None)

    summary = result.team_summary.copy()

    summary.insert(
        0,
        "season",
        result.season,
    )

    summary["simulation_count"] = (
        result.simulation_count
    )
    summary["random_seed"] = result.random_seed
    summary["simulation_mode"] = result.simulation_mode
    summary["probability_source"] = result.probability_source
    summary["model_name"] = (
        PRODUCTION_MODEL.model_name
    )
    summary["model_version"] = (
        PRODUCTION_MODEL.model_version
    )
    summary["simulation_generated_at"] = (
        generated_at
    )

    distribution = (
        result.win_distribution.copy()
    )

    distribution.insert(
        0,
        "season",
        result.season,
    )

    distribution["total_simulations"] = (
        result.simulation_count
    )
    distribution["random_seed"] = (
        result.random_seed
    )
    distribution["simulation_mode"] = result.simulation_mode
    distribution["probability_source"] = result.probability_source
    distribution["model_name"] = (
        PRODUCTION_MODEL.model_name
    )
    distribution["model_version"] = (
        PRODUCTION_MODEL.model_version
    )
    distribution["simulation_generated_at"] = (
        generated_at
    )

    return summary, distribution


def create_simulation_tables(
    connection: duckdb.DuckDBPyConnection,
    summary: pd.DataFrame,
    distribution: pd.DataFrame,
) -> None:
    """Create current simulation analytics tables."""

    connection.execute(
        f"CREATE SCHEMA IF NOT EXISTS {TARGET_SCHEMA}"
    )

    connection.register(
        "simulation_summary_frame",
        summary,
    )

    connection.register(
        "simulation_distribution_frame",
        distribution,
    )

    try:
        connection.execute(
            f"""
            CREATE OR REPLACE TABLE
                {SUMMARY_FULL_NAME}
            AS
            SELECT *
            FROM simulation_summary_frame
            """
        )

        connection.execute(
            f"""
            CREATE OR REPLACE TABLE
                {DISTRIBUTION_FULL_NAME}
            AS
            SELECT *
            FROM simulation_distribution_frame
            """
        )
    finally:
        connection.unregister(
            "simulation_summary_frame"
        )
        connection.unregister(
            "simulation_distribution_frame"
        )


def validate_simulation_tables(
    connection: duckdb.DuckDBPyConnection,
    expected_team_count: int,
    expected_distribution_count: int,
) -> None:
    """Validate persisted simulation analytics."""

    summary_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {SUMMARY_FULL_NAME}
        """
    ).fetchone()[0]

    if summary_count != expected_team_count:
        raise RuntimeError(
            "Simulation summary row count does not "
            f"match: expected {expected_team_count}, "
            f"found {summary_count}."
        )

    duplicate_team_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM (
            SELECT
                season,
                team
            FROM {SUMMARY_FULL_NAME}
            GROUP BY
                season,
                team
            HAVING COUNT(*) > 1
        )
        """
    ).fetchone()[0]

    if duplicate_team_count > 0:
        raise RuntimeError(
            "Duplicate teams found in simulation "
            "summary."
        )

    invalid_summary_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {SUMMARY_FULL_NAME}
        WHERE games < 0
           OR expected_wins < 0.0
           OR expected_losses < 0.0
           OR expected_ties < 0
           OR ABS(
                expected_wins
                + expected_losses
                + expected_ties
                - games
           ) > 0.000001
           OR simulation_count <= 0
           OR simulation_mode IS NULL
           OR probability_source IS NULL
           OR model_name IS NULL
           OR model_version IS NULL
           OR simulation_generated_at IS NULL
        """
    ).fetchone()[0]

    if invalid_summary_count > 0:
        raise RuntimeError(
            "Invalid simulation summary rows found."
        )

    distribution_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {DISTRIBUTION_FULL_NAME}
        """
    ).fetchone()[0]

    if (
        distribution_count
        != expected_distribution_count
    ):
        raise RuntimeError(
            "Simulation distribution row count does "
            f"not match: expected "
            f"{expected_distribution_count}, "
            f"found {distribution_count}."
        )

    invalid_distribution_count = (
        connection.execute(
            f"""
            SELECT COUNT(*)
            FROM {DISTRIBUTION_FULL_NAME}
            WHERE wins < 0
               OR simulation_count < 0
               OR probability NOT BETWEEN 0.0 AND 1.0
               OR total_simulations <= 0
               OR simulation_mode IS NULL
               OR probability_source IS NULL
               OR model_name IS NULL
               OR model_version IS NULL
               OR simulation_generated_at IS NULL
            """
        ).fetchone()[0]
    )

    if invalid_distribution_count > 0:
        raise RuntimeError(
            "Invalid simulation distribution rows "
            "found."
        )

    invalid_probability_sum_count = (
        connection.execute(
            f"""
            SELECT COUNT(*)
            FROM (
                SELECT
                    season,
                    team,
                    SUM(probability)
                        AS probability_sum
                FROM {DISTRIBUTION_FULL_NAME}
                GROUP BY
                    season,
                    team
                HAVING ABS(
                    probability_sum - 1.0
                ) > 0.000001
            )
            """
        ).fetchone()[0]
    )

    if invalid_probability_sum_count > 0:
        raise RuntimeError(
            "Simulation win probabilities do not "
            "sum to one."
        )

    logger.info(
        "Simulation tables validated: %s teams and "
        "%s distribution rows.",
        summary_count,
        distribution_count,
    )


def build_current_season_simulation(
    database_file: Path = DATABASE_FILE,
    simulation_count: int = (
        DEFAULT_SIMULATION_COUNT
    ),
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> MonteCarloSimulationResult:
    """Run and persist the current season simulation."""

    validate_database_file(database_file)

    benchmark_result = (
        run_current_elo_simulation_benchmark(
            database_file=database_file,
            simulation_count=simulation_count,
            random_seed=random_seed,
        )
    )

    # Historical 2021-2024 season-level backtesting and the
    # current market benchmark both slightly favored Frozen Elo.
    # The paired Dynamic result remains persisted in the benchmark
    # tables for transparency, but is no longer the production view.
    result = benchmark_result.frozen_result

    generated_at = datetime.now(
        timezone.utc
    )

    summary, distribution = (
        prepare_simulation_tables(
            result=result,
            generated_at=generated_at,
        )
    )

    (
        benchmark_summary,
        benchmark_comparison,
    ) = prepare_benchmark_tables(
        result=benchmark_result,
        generated_at=generated_at,
    )

    with duckdb.connect(
        str(database_file)
    ) as connection:
        connection.execute("BEGIN TRANSACTION")

        try:
            create_simulation_tables(
                connection=connection,
                summary=summary,
                distribution=distribution,
            )

            create_benchmark_tables(
                connection=connection,
                summary=benchmark_summary,
                comparison=benchmark_comparison,
            )

            validate_simulation_tables(
                connection=connection,
                expected_team_count=len(summary),
                expected_distribution_count=len(
                    distribution
                ),
            )

            validate_benchmark_tables(
                connection=connection,
                expected_team_count=len(
                    benchmark_comparison
                ),
            )

            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise

    logger.info(
        "Current season simulation and Elo benchmark "
        "persisted successfully."
    )

    return result


def parse_arguments() -> argparse.Namespace:
    """Parse simulation builder arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Run and persist the production Frozen Elo "
            "season simulation."
        )
    )

    parser.add_argument(
        "--simulations",
        type=int,
        default=DEFAULT_SIMULATION_COUNT,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
    )

    return parser.parse_args()


def main() -> None:
    """Run the simulation builder entry point."""

    arguments = parse_arguments()

    try:
        build_current_season_simulation(
            simulation_count=(
                arguments.simulations
            ),
            random_seed=arguments.seed,
        )
    except Exception:
        logger.exception(
            "Current season simulation build failed."
        )
        raise


if __name__ == "__main__":
    main()
