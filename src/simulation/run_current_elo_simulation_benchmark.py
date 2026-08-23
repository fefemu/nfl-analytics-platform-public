"""
NFL Analytics Platform
Current Elo Simulation Benchmark Runner

Purpose:
    Load the current regular-season simulation inputs and
    run paired dynamic and frozen Elo simulations.

Author:
    Ferenc Kaizer

Version:
    0.1.0
"""

import logging
from pathlib import Path
from time import perf_counter

import duckdb

from src.modeling.train_logistic_baseline import (
    DATABASE_FILE,
    validate_database_file,
)
from src.simulation.benchmark_elo_simulation_modes import (
    EloSimulationBenchmarkResult,
    run_elo_simulation_benchmark,
)
from src.simulation.run_current_season_simulation import (
    load_current_team_records,
    load_latest_regular_season_schedule,
    log_simulation_summary,
    validate_prediction_source,
)
from src.simulation.run_elo_monte_carlo import (
    DEFAULT_RANDOM_SEED,
    DEFAULT_SIMULATION_COUNT,
)


logger = logging.getLogger(__name__)


def run_current_elo_simulation_benchmark(
    database_file: Path = DATABASE_FILE,
    simulation_count: int = (
        DEFAULT_SIMULATION_COUNT
    ),
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> EloSimulationBenchmarkResult:
    """Run the paired current-season Elo benchmark."""

    validate_database_file(database_file)

    logger.info(
        "Starting current Elo simulation benchmark: "
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

    result = run_elo_simulation_benchmark(
        schedule=schedule,
        simulation_count=simulation_count,
        random_seed=random_seed,
        current_records=current_records,
    )

    duration = (
        perf_counter()
        - started_at
    )

    log_simulation_summary(
        result.frozen_result
    )

    logger.info(
        "Current Elo simulation benchmark completed: "
        "%s paired simulations per mode in %.2f seconds.",
        simulation_count,
        duration,
    )

    return result
