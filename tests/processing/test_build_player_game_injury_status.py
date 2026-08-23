"""Tests for the player-game injury status builder."""

from pathlib import Path

import duckdb
import pytest

from src.processing.build_player_game_injury_status import (
    build_player_game_injury_status,
    count_source_player_week_keys,
    create_player_game_injury_status,
    validate_database_file,
    validate_schedule_coverage,
    validate_source_tables,
    validate_target_table,
)


def create_source_tables(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Create representative injury and schedule sources."""

    connection.execute(
        """
        CREATE SCHEMA raw;
        CREATE SCHEMA processed;

        CREATE TABLE processed.schedule AS
        SELECT *
        FROM (
            VALUES
                (
                    '2024_15_MIA_HOU',
                    2024,
                    'REG',
                    15,
                    DATE '2024-12-15',
                    'MIA',
                    'HOU'
                ),
                (
                    '2024_15_NYJ_JAX',
                    2024,
                    'REG',
                    15,
                    DATE '2024-12-15',
                    'NYJ',
                    'JAX'
                )
        ) AS schedule(
            game_id,
            season,
            game_type,
            week,
            gameday,
            away_team,
            home_team
        );

        CREATE TABLE raw.injury_reports AS
        SELECT *
        FROM (
            VALUES
                (
                    2024,
                    'REG',
                    'REG',
                    'HOU',
                    15,
                    '00-0039359',
                    'TE',
                    'Cade Stover',
                    'Cade',
                    'Stover',
                    'Illness',
                    NULL,
                    'Questionable',
                    NULL,
                    NULL,
                    '   ',
                    TIMESTAMPTZ
                        '2024-12-15 03:34:33+00'
                ),
                (
                    2024,
                    'REG',
                    'REG',
                    'HOU',
                    15,
                    '00-0039359',
                    'TE',
                    'Cade Stover',
                    'Cade',
                    'Stover',
                    'Illness',
                    NULL,
                    'Out',
                    NULL,
                    NULL,
                    '   ',
                    TIMESTAMPTZ
                        '2024-12-15 14:17:06+00'
                ),
                (
                    2024,
                    'REG',
                    'REG',
                    'NYJ',
                    15,
                    '00-0034270',
                    'TE',
                    'Tyler Conklin',
                    'Tyler',
                    'Conklin',
                    'Personal matter',
                    NULL,
                    'Note',
                    NULL,
                    NULL,
                    'Note',
                    TIMESTAMPTZ
                        '2024-12-14 20:55:19+00'
                )
        ) AS injuries(
            season,
            season_type,
            game_type,
            team,
            week,
            gsis_id,
            position,
            full_name,
            first_name,
            last_name,
            report_primary_injury,
            report_secondary_injury,
            report_status,
            practice_primary_injury,
            practice_secondary_injury,
            practice_status,
            date_modified
        );
        """
    )


def insert_unmatched_injury(
    connection: duckdb.DuckDBPyConnection,
    season: int,
    team: str,
    week: int,
    gsis_id: str,
) -> None:
    """Insert one injury key without a schedule match."""

    connection.execute(
        """
        INSERT INTO raw.injury_reports
        VALUES (
            ?,
            'REG',
            'REG',
            ?,
            ?,
            ?,
            'QB',
            'Unmatched Player',
            'Unmatched',
            'Player',
            'Shoulder',
            NULL,
            'Questionable',
            NULL,
            NULL,
            'Limited Participation in Practice',
            TIMESTAMPTZ
                '2022-12-31 12:00:00+00'
        )
        """,
        [
            season,
            team,
            week,
            gsis_id,
        ],
    )


def test_validate_database_file_rejects_missing_file(
    tmp_path: Path,
) -> None:
    """Reject a missing DuckDB database."""

    missing_file = tmp_path / "missing.duckdb"

    with pytest.raises(
        FileNotFoundError,
        match="Database file does not exist",
    ):
        validate_database_file(
            missing_file
        )


def test_validate_database_file_accepts_database(
    tmp_path: Path,
) -> None:
    """Accept an existing DuckDB database file."""

    database_file = tmp_path / "test.duckdb"

    with duckdb.connect(
        str(database_file)
    ):
        pass

    validate_database_file(
        database_file
    )


def test_validate_source_tables_rejects_missing_table() -> None:
    """Reject missing injury and schedule sources."""

    with duckdb.connect(
        ":memory:"
    ) as connection:
        with pytest.raises(
            RuntimeError,
            match="Source table does not exist",
        ):
            validate_source_tables(
                connection
            )


def test_count_source_keys_ignores_snapshot_history() -> None:
    """Count one source key for multiple status snapshots."""

    with duckdb.connect(
        ":memory:"
    ) as connection:
        create_source_tables(
            connection
        )

        source_key_count = (
            count_source_player_week_keys(
                connection
            )
        )

    assert source_key_count == 2


def test_validate_schedule_coverage_accepts_known_unplayed_game(
) -> None:
    """Allow injury keys from the cancelled BUF-CIN game."""

    with duckdb.connect(
        ":memory:"
    ) as connection:
        create_source_tables(
            connection
        )

        insert_unmatched_injury(
            connection=connection,
            season=2022,
            team="BUF",
            week=17,
            gsis_id="00-0099991",
        )

        source_key_count = (
            count_source_player_week_keys(
                connection
            )
        )

        result = validate_schedule_coverage(
            connection=connection,
            source_key_count=source_key_count,
        )

    assert result == (
        2,
        1,
    )


def test_validate_schedule_coverage_rejects_unexpected_key(
) -> None:
    """Reject an unexplained schedule join failure."""

    with duckdb.connect(
        ":memory:"
    ) as connection:
        create_source_tables(
            connection
        )

        insert_unmatched_injury(
            connection=connection,
            season=2024,
            team="NE",
            week=2,
            gsis_id="00-0099992",
        )

        source_key_count = (
            count_source_player_week_keys(
                connection
            )
        )

        with pytest.raises(
            RuntimeError,
            match="unexpected unmatched team-week",
        ):
            validate_schedule_coverage(
                connection=connection,
                source_key_count=source_key_count,
            )


def test_create_player_game_injury_status_selects_latest_snapshot(
) -> None:
    """Select the latest status for one player-game."""

    with duckdb.connect(
        ":memory:"
    ) as connection:
        create_source_tables(
            connection
        )

        create_player_game_injury_status(
            connection
        )

        stover_row = connection.execute(
            """
            SELECT
                game_id,
                opponent,
                is_home,
                report_status,
                source_snapshot_count,
                source_date_modified =
                    TIMESTAMPTZ
                    '2024-12-15 14:17:06+00'
                    AS has_expected_timestamp
            FROM processed.player_game_injury_status
            WHERE gsis_id = '00-0039359'
            """
        ).fetchone()

    assert stover_row == (
        "2024_15_MIA_HOU",
        "MIA",
        True,
        "Out",
        2,
        True,
    )


def test_create_player_game_injury_status_cleans_statuses(
) -> None:
    """Convert Note and whitespace statuses to null."""

    with duckdb.connect(
        ":memory:"
    ) as connection:
        create_source_tables(
            connection
        )

        create_player_game_injury_status(
            connection
        )

        conklin_row = connection.execute(
            """
            SELECT
                report_status,
                practice_status,
                is_out,
                is_doubtful,
                is_questionable,
                did_not_practice,
                limited_practice,
                full_practice
            FROM processed.player_game_injury_status
            WHERE gsis_id = '00-0034270'
            """
        ).fetchone()

    assert conklin_row == (
        None,
        None,
        False,
        False,
        False,
        False,
        False,
        False,
    )


def test_validate_target_table_accepts_valid_result() -> None:
    """Validate a unique and completely matched target."""

    with duckdb.connect(
        ":memory:"
    ) as connection:
        create_source_tables(
            connection
        )

        source_key_count = (
            count_source_player_week_keys(
                connection
            )
        )

        create_player_game_injury_status(
            connection
        )

        result = validate_target_table(
            connection=connection,
            source_key_count=source_key_count,
        )

    assert result == (
        2,
        1,
    )


def test_validate_target_table_rejects_join_loss() -> None:
    """Reject a source key without a scheduled-game match."""

    with duckdb.connect(
        ":memory:"
    ) as connection:
        create_source_tables(
            connection
        )

        create_player_game_injury_status(
            connection
        )

        with pytest.raises(
            RuntimeError,
            match="Not every distinct injury",
        ):
            validate_target_table(
                connection=connection,
                source_key_count=3,
            )


def test_build_player_game_injury_status_creates_table(
    tmp_path: Path,
) -> None:
    """Run the complete builder on a temporary database."""

    database_file = tmp_path / "test.duckdb"

    with duckdb.connect(
        str(database_file)
    ) as connection:
        create_source_tables(
            connection
        )

    build_player_game_injury_status(
        database_file
    )

    with duckdb.connect(
        str(database_file),
        read_only=True,
    ) as connection:
        rows = connection.execute(
            """
            SELECT
                gsis_id,
                report_status,
                source_snapshot_count
            FROM processed.player_game_injury_status
            ORDER BY gsis_id
            """
        ).fetchall()

    assert rows == [
        (
            "00-0034270",
            None,
            1,
        ),
        (
            "00-0039359",
            "Out",
            2,
        ),
    ]