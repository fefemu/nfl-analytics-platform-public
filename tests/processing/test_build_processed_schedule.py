from pathlib import Path

import duckdb
import pytest

from src.processing.build_processed_schedule import (
    build_processed_schedule,
    validate_additional_derived_fields,
    validate_database_file,
    validate_game_results,
    validate_source_table,
    validate_target_table,
)


def create_game_results_table(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Create the minimal table required for game result validation."""

    connection.execute("CREATE SCHEMA processed")

    connection.execute(
        """
        CREATE TABLE processed.schedule (
            home_score INTEGER,
            away_score INTEGER,
            is_completed BOOLEAN,
            home_win BOOLEAN,
            away_win BOOLEAN,
            is_tie BOOLEAN,
            point_differential INTEGER,
            total_points INTEGER,
            game_result VARCHAR
        )
        """
    )


def create_target_table(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Create the minimal table required for target validation."""

    connection.execute("CREATE SCHEMA processed")

    connection.execute(
        """
        CREATE TABLE processed.schedule (
            game_id VARCHAR,
            season INTEGER,
            game_type VARCHAR,
            week INTEGER,
            gameday DATE,
            weekday VARCHAR,
            away_team VARCHAR,
            home_team VARCHAR,
            away_rest INTEGER,
            home_rest INTEGER
        )
        """
    )


def create_additional_derived_table(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Create the minimal table for additional derived validation."""

    connection.execute("CREATE SCHEMA processed")

    connection.execute(
        """
        CREATE TABLE processed.schedule (
            home_rest INTEGER,
            away_rest INTEGER,
            home_rest_advantage INTEGER,
            game_type VARCHAR,
            is_regular_season BOOLEAN,
            is_playoff BOOLEAN
        )
        """
    )


def create_raw_schedule_table(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Create a minimal raw schedule table for integration testing."""

    connection.execute("CREATE SCHEMA raw")

    connection.execute(
        """
        CREATE TABLE raw.schedule AS
        SELECT
            '2026_01_BUF_NYJ'::VARCHAR AS game_id,
            2026::INTEGER AS season,
            'REG'::VARCHAR AS game_type,
            1::INTEGER AS week,
            '2026-09-13'::VARCHAR AS gameday,
            '13:00'::VARCHAR AS gametime,
            'Sunday'::VARCHAR AS weekday,
            'BUF'::VARCHAR AS away_team,
            'NYJ'::VARCHAR AS home_team,
            'Home'::VARCHAR AS location,
            17::INTEGER AS away_score,
            21::INTEGER AS home_score,
            0::INTEGER AS overtime,
            5::INTEGER AS away_rest,
            7::INTEGER AS home_rest,
            150::INTEGER AS away_moneyline,
            -170::INTEGER AS home_moneyline,
            -3.5::DOUBLE AS spread_line,
            -110::INTEGER AS away_spread_odds,
            -110::INTEGER AS home_spread_odds,
            38.5::DOUBLE AS total_line,
            -110::INTEGER AS under_odds,
            -110::INTEGER AS over_odds,
            'outdoors'::VARCHAR AS roof,
            'grass'::VARCHAR AS surface,
            70::INTEGER AS temp,
            5::INTEGER AS wind,
            'away-qb-id'::VARCHAR AS away_qb_id,
            'home-qb-id'::VARCHAR AS home_qb_id,
            'Away QB'::VARCHAR AS away_qb_name,
            'Home QB'::VARCHAR AS home_qb_name,
            'Away Coach'::VARCHAR AS away_coach,
            'Home Coach'::VARCHAR AS home_coach,
            'Referee'::VARCHAR AS referee,
            'stadium-id'::VARCHAR AS stadium_id,
            'Test Stadium'::VARCHAR AS stadium
        """
    )


def test_validate_game_results_rejects_inconsistent_home_win() -> None:
    """Fail validation when home_win contradicts the final scores."""

    with duckdb.connect(":memory:") as connection:
        create_game_results_table(connection)

        connection.execute(
            """
            INSERT INTO processed.schedule
            VALUES (
                21,
                17,
                TRUE,
                FALSE,
                FALSE,
                FALSE,
                4,
                38,
                'HOME_WIN'
            )
            """
        )

        with pytest.raises(
            RuntimeError,
            match="inconsistent derived game results",
        ):
            validate_game_results(connection)


def test_validate_game_results_accepts_consistent_home_win() -> None:
    """Pass validation when derived fields match the final scores."""

    with duckdb.connect(":memory:") as connection:
        create_game_results_table(connection)

        connection.execute(
            """
            INSERT INTO processed.schedule
            VALUES (
                21,
                17,
                TRUE,
                TRUE,
                FALSE,
                FALSE,
                4,
                38,
                'HOME_WIN'
            )
            """
        )

        validate_game_results(connection)


def test_validate_game_results_accepts_not_played_game() -> None:
    """Pass validation for a game without final scores."""

    with duckdb.connect(":memory:") as connection:
        create_game_results_table(connection)

        connection.execute(
            """
            INSERT INTO processed.schedule
            VALUES (
                NULL,
                NULL,
                FALSE,
                NULL,
                NULL,
                NULL,
                NULL,
                NULL,
                'NOT_PLAYED'
            )
            """
        )

        validate_game_results(connection)


def test_validate_target_table_rejects_duplicate_game_id() -> None:
    """Fail validation when game_id is not unique."""

    with duckdb.connect(":memory:") as connection:
        create_target_table(connection)

        connection.execute(
            """
            INSERT INTO processed.schedule
            VALUES
                (
                    '2026_01_BUF_NYJ',
                    2026,
                    'REG',
                    1,
                    DATE '2026-09-13',
                    'Sunday',
                    'BUF',
                    'NYJ',
                    7,
                    7
                ),
                (
                    '2026_01_BUF_NYJ',
                    2026,
                    'REG',
                    1,
                    DATE '2026-09-13',
                    'Sunday',
                    'BUF',
                    'NYJ',
                    7,
                    7
                )
            """
        )

        with pytest.raises(
            RuntimeError,
            match="duplicate game_id",
        ):
            validate_target_table(connection)


def test_validate_target_table_rejects_missing_required_field() -> None:
    """Fail validation when a required field is missing."""

    with duckdb.connect(":memory:") as connection:
        create_target_table(connection)

        connection.execute(
            """
            INSERT INTO processed.schedule
            VALUES (
                '2026_01_BUF_NYJ',
                2026,
                'REG',
                1,
                DATE '2026-09-13',
                'Sunday',
                'BUF',
                NULL,
                7,
                7
            )
            """
        )

        with pytest.raises(
            RuntimeError,
            match="missing required fields",
        ):
            validate_target_table(connection)


def test_additional_derived_fields_rejects_invalid_playoff_flag() -> None:
    """Fail validation when a regular-season game is marked as playoff."""

    with duckdb.connect(":memory:") as connection:
        create_additional_derived_table(connection)

        connection.execute(
            """
            INSERT INTO processed.schedule
            VALUES (
                7,
                5,
                2,
                'REG',
                TRUE,
                TRUE
            )
            """
        )

        with pytest.raises(
            RuntimeError,
            match="inconsistent additional derived fields",
        ):
            validate_additional_derived_fields(connection)


def test_additional_derived_fields_accepts_valid_playoff_game() -> None:
    """Pass validation when playoff and rest fields are consistent."""

    with duckdb.connect(":memory:") as connection:
        create_additional_derived_table(connection)

        connection.execute(
            """
            INSERT INTO processed.schedule
            VALUES (
                6,
                7,
                -1,
                'DIV',
                FALSE,
                TRUE
            )
            """
        )

        validate_additional_derived_fields(connection)


def test_validate_source_table_rejects_missing_raw_schedule() -> None:
    """Fail validation when the raw schedule table does not exist."""

    with duckdb.connect(":memory:") as connection:
        with pytest.raises(
            RuntimeError,
            match="Source table does not exist: raw.schedule",
        ):
            validate_source_table(connection)


def test_validate_source_table_accepts_existing_raw_schedule() -> None:
    """Pass validation when the raw schedule table exists."""

    with duckdb.connect(":memory:") as connection:
        connection.execute("CREATE SCHEMA raw")
        connection.execute(
            "CREATE TABLE raw.schedule (game_id VARCHAR)"
        )

        validate_source_table(connection)


def test_validate_database_file_rejects_missing_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail validation when the DuckDB database file is missing."""

    missing_database = tmp_path / "missing.duckdb"

    monkeypatch.setattr(
        "src.processing.build_processed_schedule.DATABASE_FILE",
        missing_database,
    )

    with pytest.raises(
        FileNotFoundError,
        match="DuckDB database does not exist",
    ):
        validate_database_file()


def test_validate_database_file_accepts_existing_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pass validation when the database file exists."""

    database_file = tmp_path / "test.duckdb"
    database_file.touch()

    monkeypatch.setattr(
        "src.processing.build_processed_schedule.DATABASE_FILE",
        database_file,
    )

    validate_database_file()


def test_validate_target_table_accepts_valid_record() -> None:
    """Pass validation when required fields and game_id are valid."""

    with duckdb.connect(":memory:") as connection:
        create_target_table(connection)

        connection.execute(
            """
            INSERT INTO processed.schedule
            VALUES (
                '2026_01_BUF_NYJ',
                2026,
                'REG',
                1,
                DATE '2026-09-13',
                'Sunday',
                'BUF',
                'NYJ',
                7,
                7
            )
            """
        )

        validate_target_table(connection)


def test_build_processed_schedule_creates_expected_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Build processed.schedule from a temporary raw schedule table."""

    database_file = tmp_path / "integration.duckdb"

    with duckdb.connect(str(database_file)) as connection:
        create_raw_schedule_table(connection)

    monkeypatch.setattr(
        "src.processing.build_processed_schedule.DATABASE_FILE",
        database_file,
    )

    build_processed_schedule()

    with duckdb.connect(
        str(database_file),
        read_only=True,
    ) as connection:
        result = connection.execute(
            """
            SELECT
                is_completed,
                home_win,
                away_win,
                is_tie,
                point_differential,
                total_points,
                game_result,
                home_rest_advantage,
                is_regular_season,
                is_playoff
            FROM processed.schedule
            """
        ).fetchone()

        column_count = connection.execute(
            """
            SELECT COUNT(*)
            FROM information_schema.columns
            WHERE table_schema = 'processed'
              AND table_name = 'schedule'
            """
        ).fetchone()[0]

    assert result == (
        True,
        True,
        False,
        False,
        4,
        38,
        "HOME_WIN",
        2,
        True,
        False,
    )
    assert column_count == 46


def test_build_processed_schedule_rolls_back_failed_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve the previous target table when validation fails."""

    database_file = tmp_path / "rollback.duckdb"

    with duckdb.connect(str(database_file)) as connection:
        create_raw_schedule_table(connection)

        connection.execute("CREATE SCHEMA processed")
        connection.execute(
            """
            CREATE TABLE processed.schedule (
                marker VARCHAR
            )
            """
        )
        connection.execute(
            """
            INSERT INTO processed.schedule
            VALUES ('original')
            """
        )

    monkeypatch.setattr(
        "src.processing.build_processed_schedule.DATABASE_FILE",
        database_file,
    )
    monkeypatch.setattr(
        "src.processing.build_processed_schedule."
        "EXPECTED_TARGET_COLUMN_COUNT",
        47,
    )

    with pytest.raises(
        RuntimeError,
        match="Unexpected target column count",
    ):
        build_processed_schedule()

    with duckdb.connect(
        str(database_file),
        read_only=True,
    ) as connection:
        columns = connection.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'processed'
              AND table_name = 'schedule'
            ORDER BY ordinal_position
            """
        ).fetchall()

        marker = connection.execute(
            "SELECT marker FROM processed.schedule"
        ).fetchone()[0]

    assert columns == [("marker",)]
    assert marker == "original"
