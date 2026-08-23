"""
NFL Analytics Platform
Current Elo Simulation Benchmark Persistence

Purpose:
    Prepare, persist and validate frozen-versus-dynamic
    Elo season-simulation benchmark outputs.

Author:
    Ferenc Kaizer

Version:
    0.1.0
"""

from datetime import datetime, timezone

import duckdb
import pandas as pd

from src.simulation.benchmark_elo_simulation_modes import (
    BENCHMARK_SUMMARY_COLUMNS,
    TEAM_COMPARISON_COLUMNS,
    EloSimulationBenchmarkResult,
)


TARGET_SCHEMA = "analytics"

SUMMARY_TABLE = (
    "current_season_elo_benchmark_summary"
)

COMPARISON_TABLE = (
    "current_season_elo_benchmark_team_comparison"
)

SUMMARY_FULL_NAME = (
    f"{TARGET_SCHEMA}.{SUMMARY_TABLE}"
)

COMPARISON_FULL_NAME = (
    f"{TARGET_SCHEMA}.{COMPARISON_TABLE}"
)

PERSISTED_SUMMARY_COLUMNS = (
    *BENCHMARK_SUMMARY_COLUMNS,
    "benchmark_generated_at",
)

PERSISTED_COMPARISON_COLUMNS = (
    "season",
    "simulation_count",
    "random_seed",
    "comparison_method",
    *TEAM_COMPARISON_COLUMNS,
    "benchmark_generated_at",
)


def prepare_benchmark_tables(
    result: EloSimulationBenchmarkResult,
    generated_at: datetime | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add stable benchmark run metadata."""

    benchmark_generated_at = (
        generated_at
        if generated_at is not None
        else datetime.now(timezone.utc)
    )

    summary = (
        result.benchmark_summary.copy()
    )

    summary[
        "benchmark_generated_at"
    ] = benchmark_generated_at

    summary = summary.loc[
        :,
        PERSISTED_SUMMARY_COLUMNS,
    ]

    comparison = (
        result.team_comparison.copy()
    )

    comparison.insert(
        0,
        "comparison_method",
        "common_random_numbers",
    )
    comparison.insert(
        0,
        "random_seed",
        result.dynamic_result.random_seed,
    )
    comparison.insert(
        0,
        "simulation_count",
        result.dynamic_result.simulation_count,
    )
    comparison.insert(
        0,
        "season",
        result.dynamic_result.season,
    )

    comparison[
        "benchmark_generated_at"
    ] = benchmark_generated_at

    comparison = comparison.loc[
        :,
        PERSISTED_COMPARISON_COLUMNS,
    ]

    return summary, comparison


def create_benchmark_tables(
    connection: duckdb.DuckDBPyConnection,
    summary: pd.DataFrame,
    comparison: pd.DataFrame,
) -> None:
    """Create benchmark summary and team tables."""

    missing_summary_columns = sorted(
        set(PERSISTED_SUMMARY_COLUMNS)
        - set(summary.columns)
    )

    if missing_summary_columns:
        raise ValueError(
            "Benchmark summary is missing columns: "
            + ", ".join(
                missing_summary_columns
            )
        )

    missing_comparison_columns = sorted(
        set(PERSISTED_COMPARISON_COLUMNS)
        - set(comparison.columns)
    )

    if missing_comparison_columns:
        raise ValueError(
            "Benchmark comparison is missing columns: "
            + ", ".join(
                missing_comparison_columns
            )
        )

    connection.execute(
        f"CREATE SCHEMA IF NOT EXISTS {TARGET_SCHEMA}"
    )

    connection.register(
        "benchmark_summary_frame",
        summary.loc[
            :,
            PERSISTED_SUMMARY_COLUMNS,
        ],
    )

    connection.register(
        "benchmark_comparison_frame",
        comparison.loc[
            :,
            PERSISTED_COMPARISON_COLUMNS,
        ],
    )

    try:
        connection.execute(
            f"""
            CREATE OR REPLACE TABLE {SUMMARY_FULL_NAME}
            AS
            SELECT *
            FROM benchmark_summary_frame
            """
        )

        connection.execute(
            f"""
            CREATE OR REPLACE TABLE {COMPARISON_FULL_NAME}
            AS
            SELECT *
            FROM benchmark_comparison_frame
            """
        )
    finally:
        connection.unregister(
            "benchmark_summary_frame"
        )
        connection.unregister(
            "benchmark_comparison_frame"
        )


def validate_benchmark_tables(
    connection: duckdb.DuckDBPyConnection,
    expected_team_count: int,
) -> None:
    """Validate persisted benchmark outputs."""

    summary_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {SUMMARY_FULL_NAME}
        """
    ).fetchone()[0]

    if summary_count != 1:
        raise RuntimeError(
            "Benchmark summary must contain exactly "
            f"one row, found {summary_count}."
        )

    comparison_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {COMPARISON_FULL_NAME}
        """
    ).fetchone()[0]

    if comparison_count != expected_team_count:
        raise RuntimeError(
            "Benchmark comparison team count does not "
            f"match: expected {expected_team_count}, "
            f"found {comparison_count}."
        )

    invalid_summary_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {SUMMARY_FULL_NAME}
        WHERE season IS NULL
           OR simulation_count <= 0
           OR random_seed IS NULL
           OR comparison_method
                <> 'common_random_numbers'
           OR dynamic_mode <> 'DYNAMIC_ELO'
           OR frozen_mode <> 'FROZEN_ELO'
           OR team_count <> {expected_team_count}
           OR mean_absolute_expected_wins_delta < 0.0
           OR maximum_absolute_expected_wins_delta < 0.0
           OR mean_win_distribution_total_variation
                NOT BETWEEN 0.0 AND 1.0
           OR maximum_win_distribution_total_variation
                NOT BETWEEN 0.0 AND 1.0
           OR benchmark_generated_at IS NULL
        """
    ).fetchone()[0]

    if invalid_summary_count > 0:
        raise RuntimeError(
            "Invalid Elo benchmark summary found."
        )

    duplicate_team_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM (
            SELECT team
            FROM {COMPARISON_FULL_NAME}
            GROUP BY team
            HAVING COUNT(*) > 1
        )
        """
    ).fetchone()[0]

    if duplicate_team_count > 0:
        raise RuntimeError(
            "Duplicate Elo benchmark teams found."
        )

    invalid_comparison_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {COMPARISON_FULL_NAME}
        WHERE team IS NULL
           OR games <= 0
           OR simulation_count <= 0
           OR comparison_method
                <> 'common_random_numbers'
           OR ABS(
                expected_wins_delta
                -
                (
                    dynamic_expected_wins
                    - frozen_expected_wins
                )
              ) > 0.000000001
           OR ABS(
                absolute_expected_wins_delta
                - ABS(expected_wins_delta)
              ) > 0.000000001
           OR ABS(
                dynamic_win_range
                -
                (
                    dynamic_p90_wins
                    - dynamic_p10_wins
                )
              ) > 0.000000001
           OR ABS(
                frozen_win_range
                -
                (
                    frozen_p90_wins
                    - frozen_p10_wins
                )
              ) > 0.000000001
           OR ABS(
                win_range_delta
                -
                (
                    dynamic_win_range
                    - frozen_win_range
                )
              ) > 0.000000001
           OR ABS(
                expected_final_elo_delta
                -
                (
                    dynamic_expected_final_elo
                    - frozen_expected_final_elo
                )
              ) > 0.000000001
           OR dynamic_probability_14_plus_wins
                NOT BETWEEN 0.0 AND 1.0
           OR frozen_probability_14_plus_wins
                NOT BETWEEN 0.0 AND 1.0
           OR dynamic_probability_3_or_fewer_wins
                NOT BETWEEN 0.0 AND 1.0
           OR frozen_probability_3_or_fewer_wins
                NOT BETWEEN 0.0 AND 1.0
           OR dynamic_probability_2_or_fewer_wins
                NOT BETWEEN 0.0 AND 1.0
           OR frozen_probability_2_or_fewer_wins
                NOT BETWEEN 0.0 AND 1.0
           OR win_distribution_total_variation
                NOT BETWEEN 0.0 AND 1.0
           OR benchmark_generated_at IS NULL
        """
    ).fetchone()[0]

    if invalid_comparison_count > 0:
        raise RuntimeError(
            "Invalid Elo benchmark team comparison "
            "found."
        )

    invalid_total_delta = connection.execute(
        f"""
        SELECT ABS(SUM(expected_wins_delta))
        FROM {COMPARISON_FULL_NAME}
        """
    ).fetchone()[0]

    if invalid_total_delta > 0.000000001:
        raise RuntimeError(
            "Paired benchmark does not preserve total "
            "expected wins."
        )

    metadata_mismatch_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {COMPARISON_FULL_NAME}
            AS comparison

        CROSS JOIN {SUMMARY_FULL_NAME}
            AS summary

        WHERE comparison.season
                IS DISTINCT FROM summary.season
           OR comparison.simulation_count
                IS DISTINCT FROM
                    summary.simulation_count
           OR comparison.random_seed
                IS DISTINCT FROM summary.random_seed
           OR comparison.comparison_method
                IS DISTINCT FROM
                    summary.comparison_method
           OR comparison.benchmark_generated_at
                IS DISTINCT FROM
                    summary.benchmark_generated_at
        """
    ).fetchone()[0]

    if metadata_mismatch_count > 0:
        raise RuntimeError(
            "Elo benchmark metadata mismatch found."
        )