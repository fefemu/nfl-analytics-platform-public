"""Tests for persisted Elo simulation benchmarks."""

from datetime import datetime, timezone

import duckdb
import pandas as pd
import pytest

from src.simulation.benchmark_elo_simulation_modes import (
    run_elo_simulation_benchmark,
)
from src.simulation.build_current_elo_simulation_benchmark import (
    COMPARISON_FULL_NAME,
    PERSISTED_COMPARISON_COLUMNS,
    PERSISTED_SUMMARY_COLUMNS,
    SUMMARY_FULL_NAME,
    create_benchmark_tables,
    prepare_benchmark_tables,
    validate_benchmark_tables,
)


@pytest.fixture
def connection() -> duckdb.DuckDBPyConnection:
    """Create an in-memory benchmark database."""

    database = duckdb.connect(":memory:")

    yield database

    database.close()


def create_schedule() -> pd.DataFrame:
    """Create repeated two-team matchups."""

    rows: list[dict[str, object]] = []

    for week in range(1, 5):
        home_team = (
            "NE"
            if week % 2 == 1
            else "NYJ"
        )
        away_team = (
            "NYJ"
            if home_team == "NE"
            else "NE"
        )

        rows.append(
            {
                "game_id": f"game_{week}",
                "season": 2026,
                "week": week,
                "gameday": pd.Timestamp(
                    "2026-09-01"
                )
                + pd.Timedelta(
                    days=7 * week
                ),
                "gametime": "13:00",
                "home_team": home_team,
                "away_team": away_team,
                "is_neutral": False,
                "home_rating_pregame": (
                    1550.0
                    if home_team == "NE"
                    else 1450.0
                ),
                "away_rating_pregame": (
                    1450.0
                    if away_team == "NYJ"
                    else 1550.0
                ),
            }
        )

    return pd.DataFrame(rows)


def create_tables() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """Create deterministic persisted frames."""

    result = run_elo_simulation_benchmark(
        schedule=create_schedule(),
        simulation_count=500,
        random_seed=42,
    )

    return prepare_benchmark_tables(
        result=result,
        generated_at=datetime(
            2026,
            8,
            6,
            15,
            0,
            tzinfo=timezone.utc,
        ),
    )


def test_prepare_benchmark_tables(
) -> None:
    """Add stable shared benchmark metadata."""

    summary, comparison = create_tables()

    assert tuple(
        summary.columns
    ) == PERSISTED_SUMMARY_COLUMNS

    assert tuple(
        comparison.columns
    ) == PERSISTED_COMPARISON_COLUMNS

    assert len(summary) == 1
    assert len(comparison) == 2

    assert set(
        comparison["comparison_method"]
    ) == {
        "common_random_numbers",
    }


def test_create_and_validate_benchmark_tables(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Persist valid benchmark outputs."""

    summary, comparison = create_tables()

    create_benchmark_tables(
        connection=connection,
        summary=summary,
        comparison=comparison,
    )

    validate_benchmark_tables(
        connection=connection,
        expected_team_count=2,
    )

    assert connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {SUMMARY_FULL_NAME}
        """
    ).fetchone()[0] == 1

    assert connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {COMPARISON_FULL_NAME}
        """
    ).fetchone()[0] == 2


def test_validator_rejects_invalid_delta(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Reject incorrect expected-win differences."""

    summary, comparison = create_tables()

    create_benchmark_tables(
        connection=connection,
        summary=summary,
        comparison=comparison,
    )

    connection.execute(
        f"""
        UPDATE {COMPARISON_FULL_NAME}
        SET expected_wins_delta = 99.0
        WHERE team = 'NE'
        """
    )

    with pytest.raises(
        RuntimeError,
        match="team comparison",
    ):
        validate_benchmark_tables(
            connection=connection,
            expected_team_count=2,
        )


def test_validator_rejects_metadata_mismatch(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Reject unpaired benchmark metadata."""

    summary, comparison = create_tables()

    create_benchmark_tables(
        connection=connection,
        summary=summary,
        comparison=comparison,
    )

    connection.execute(
        f"""
        UPDATE {COMPARISON_FULL_NAME}
        SET random_seed = 99
        WHERE team = 'NE'
        """
    )

    with pytest.raises(
        RuntimeError,
        match="metadata mismatch",
    ):
        validate_benchmark_tables(
            connection=connection,
            expected_team_count=2,
        )