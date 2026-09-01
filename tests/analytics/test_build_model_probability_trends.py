"""Tests for published win-probability history and current trend preparation."""

from pathlib import Path

import duckdb
import pytest

from src.analytics.build_model_probability_trends import (
    ARCHIVE_TABLE,
    CURRENT_TABLE,
    ProbabilityTrendConfig,
    archive_previous_published_predictions,
    build_current_game_probability_trends,
)


def create_predictions(database: Path, home_probability: float = 0.60) -> None:
    with duckdb.connect(str(database)) as connection:
        connection.execute("CREATE SCHEMA IF NOT EXISTS analytics")
        connection.execute(
            """
            CREATE OR REPLACE TABLE analytics.current_game_predictions AS
            SELECT '2026_01_BUF_KC'::VARCHAR AS game_id, 2026::INTEGER AS season,
                   1::INTEGER AS week, 'KC'::VARCHAR AS home_team,
                   'BUF'::VARCHAR AS away_team, ?::DOUBLE AS home_win_probability,
                   (1.0 - ?)::DOUBLE AS away_win_probability,
                   TIMESTAMP '2026-09-01 12:00:00' AS prediction_generated_at
            """,
            [home_probability, home_probability],
        )


def test_archive_is_idempotent_and_preserves_both_outcomes(tmp_path: Path) -> None:
    database = tmp_path / "source.duckdb"
    create_predictions(database)

    assert archive_previous_published_predictions(database) == 2
    assert archive_previous_published_predictions(database) == 2

    with duckdb.connect(str(database)) as connection:
        rows = connection.execute(
            f"SELECT team, outcome_side, publication_status FROM {ARCHIVE_TABLE} ORDER BY team"
        ).fetchall()
    assert rows == [("BUF", "AWAY", "PUBLISHED"), ("KC", "HOME", "PUBLISHED")]


def test_build_trends_classifies_increase_decrease_and_complements(
    tmp_path: Path,
) -> None:
    database = tmp_path / "source.duckdb"
    create_predictions(database, 0.60)
    archive_previous_published_predictions(database)
    create_predictions(database, 0.64)

    assert build_current_game_probability_trends(database) == 1

    with duckdb.connect(str(database)) as connection:
        row = connection.execute(
            f"""
            SELECT home_probability_change_pp, away_probability_change_pp,
                   home_probability_trend, away_probability_trend
            FROM {CURRENT_TABLE}
            """
        ).fetchone()
    assert row[0] == pytest.approx(4.0)
    assert row[1] == pytest.approx(-4.0)
    assert row[2:] == ("INCREASE", "DECREASE")


def test_neutral_threshold_is_configurable_and_strict(tmp_path: Path) -> None:
    database = tmp_path / "source.duckdb"
    create_predictions(database, 0.60)
    archive_previous_published_predictions(database)
    create_predictions(database, 0.604)

    build_current_game_probability_trends(
        database, ProbabilityTrendConfig(neutral_threshold_pp=0.5)
    )

    with duckdb.connect(str(database)) as connection:
        trends = connection.execute(
            f"SELECT home_probability_trend, away_probability_trend FROM {CURRENT_TABLE}"
        ).fetchone()
    assert trends == ("UNCHANGED", "UNCHANGED")


def test_missing_prior_is_new_prediction_not_zero_change(tmp_path: Path) -> None:
    database = tmp_path / "source.duckdb"
    create_predictions(database)

    build_current_game_probability_trends(database)

    with duckdb.connect(str(database)) as connection:
        row = connection.execute(
            f"""
            SELECT home_probability_trend, away_probability_trend,
                   home_probability_change_pp, away_probability_change_pp
            FROM {CURRENT_TABLE}
            """
        ).fetchone()
    assert row == ("NEW", "NEW", None, None)


def test_unpublished_archive_rows_never_become_reference(tmp_path: Path) -> None:
    database = tmp_path / "source.duckdb"
    create_predictions(database, 0.60)
    archive_previous_published_predictions(database)
    with duckdb.connect(str(database)) as connection:
        connection.execute(
            f"UPDATE {ARCHIVE_TABLE} SET publication_status = 'FAILED'"
        )

    build_current_game_probability_trends(database)

    with duckdb.connect(str(database)) as connection:
        trends = connection.execute(
            f"SELECT home_probability_trend, away_probability_trend FROM {CURRENT_TABLE}"
        ).fetchone()
    assert trends == ("NEW", "NEW")
