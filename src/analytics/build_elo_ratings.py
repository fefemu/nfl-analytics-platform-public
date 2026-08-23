"""
NFL Analytics Platform
Elo Ratings Builder

Purpose:
    Build historical Elo game predictions
    and current team ratings from processed schedule data.

Author:
    Ferenc Kaizer

Version:
    0.1.0
"""

import logging
from pathlib import Path

import duckdb

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

TARGET_SCHEMA = "analytics"
HISTORY_TABLE = "elo_game_predictions"
CURRENT_TABLE = "current_elo_ratings"

SOURCE_FULL_NAME = f"{SOURCE_SCHEMA}.{SOURCE_TABLE}"
HISTORY_FULL_NAME = f"{TARGET_SCHEMA}.{HISTORY_TABLE}"
CURRENT_FULL_NAME = f"{TARGET_SCHEMA}.{CURRENT_TABLE}"

BASELINE_HOME_ADVANTAGE = 50.0

REQUIRED_SOURCE_COLUMNS = {
    "game_id",
    "season",
    "game_type",
    "week",
    "gameday",
    "gametime",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "location",
    "is_completed",
}


def validate_database_file(
    database_file: Path = DATABASE_FILE,
) -> None:
    """Validate that the DuckDB database file exists."""

    if not database_file.is_file():
        raise FileNotFoundError(
            f"DuckDB database file does not exist: {database_file}"
        )

    logger.info(
        "Database file validated: %s",
        database_file,
    )


def validate_source_table(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Validate that the processed schedule source table exists."""

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

    logger.info(
        "Elo source table validated: %s",
        SOURCE_FULL_NAME,
    )


def validate_source_columns(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Validate the columns required by the Elo builder."""

    available_columns = {
        row[0]
        for row in connection.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = ?
              AND table_name = ?
            """,
            [SOURCE_SCHEMA, SOURCE_TABLE],
        ).fetchall()
    }

    missing_columns = (
        REQUIRED_SOURCE_COLUMNS - available_columns
    )

    if missing_columns:
        missing_names = ", ".join(
            sorted(missing_columns)
        )
        raise RuntimeError(
            "Missing Elo source columns: "
            f"{missing_names}"
        )

    logger.info(
        "Required Elo source columns validated successfully."
    )


def load_historical_games(
    connection: duckdb.DuckDBPyConnection,
) -> list[HistoricalGame]:
    """Load completed NFL games required by the Elo model."""

    rows = connection.execute(
        f"""
        SELECT
            game_id,
            season,
            game_type,
            week,
            gameday,
            gametime,
            home_team,
            away_team,
            home_score,
            away_score,
            UPPER(
                TRIM(
                    COALESCE(location, '')
                )
            ) = 'NEUTRAL' AS is_neutral
        FROM {SOURCE_FULL_NAME}
        WHERE is_completed = TRUE
          AND game_type IN (
              'REG',
              'WC',
              'DIV',
              'CON',
              'SB'
          )
        ORDER BY
            gameday,
            gametime NULLS FIRST,
            game_id
        """
    ).fetchall()

    if not rows:
        raise RuntimeError(
            "No completed NFL games are available for Elo processing."
        )

    games = [
        HistoricalGame(
            game_id=row[0],
            season=row[1],
            game_type=row[2],
            week=row[3],
            gameday=row[4],
            gametime=row[5],
            home_team=row[6],
            away_team=row[7],
            home_score=row[8],
            away_score=row[9],
            is_neutral=row[10],
        )
        for row in rows
    ]

    logger.info(
        "Historical Elo games loaded: %s games.",
        len(games),
    )

    return games


def create_history_table(
    connection: duckdb.DuckDBPyConnection,
    history_records: list[EloHistoryRecord],
) -> None:
    """Create the historical Elo game predictions table."""

    if not history_records:
        raise RuntimeError(
            "No Elo history records are available for loading."
        )

    connection.execute(
        f"CREATE SCHEMA IF NOT EXISTS {TARGET_SCHEMA}"
    )

    connection.execute(
        f"""
        CREATE OR REPLACE TABLE {HISTORY_FULL_NAME} (
            game_id VARCHAR,
            season INTEGER,
            game_type VARCHAR,
            week INTEGER,
            gameday DATE,
            home_team VARCHAR,
            away_team VARCHAR,
            home_franchise VARCHAR,
            away_franchise VARCHAR,
            is_neutral BOOLEAN,
            home_advantage DOUBLE,
            home_rating_pre DOUBLE,
            away_rating_pre DOUBLE,
            home_win_probability DOUBLE,
            away_win_probability DOUBLE,
            actual_home_score DOUBLE,
            home_rating_post DOUBLE,
            away_rating_post DOUBLE,
            home_rating_change DOUBLE
        )
        """
    )

    rows = [
        (
            record.game_id,
            record.season,
            record.game_type,
            record.week,
            record.gameday,
            record.home_team,
            record.away_team,
            record.home_franchise,
            record.away_franchise,
            record.is_neutral,
            record.home_advantage,
            record.home_rating_pre,
            record.away_rating_pre,
            record.home_win_probability,
            record.away_win_probability,
            record.actual_home_score,
            record.home_rating_post,
            record.away_rating_post,
            record.home_rating_change,
        )
        for record in history_records
    ]

    connection.executemany(
        f"""
        INSERT INTO {HISTORY_FULL_NAME}
        VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
        """,
        rows,
    )

    logger.info(
        "Historical Elo predictions created: %s rows.",
        len(rows),
    )


def create_current_ratings_table(
    connection: duckdb.DuckDBPyConnection,
    history_records: list[EloHistoryRecord],
    final_ratings: dict[str, float],
) -> None:
    """Create the latest Elo rating table for all franchises."""

    if not history_records or not final_ratings:
        raise RuntimeError(
            "No final Elo ratings are available for loading."
        )

    games_played: dict[str, int] = {}
    last_game: dict[str, EloHistoryRecord] = {}

    for record in history_records:
        games_played[record.home_franchise] = (
            games_played.get(
                record.home_franchise,
                0,
            )
            + 1
        )
        games_played[record.away_franchise] = (
            games_played.get(
                record.away_franchise,
                0,
            )
            + 1
        )

        last_game[record.home_franchise] = record
        last_game[record.away_franchise] = record

    sorted_ratings = sorted(
        final_ratings.items(),
        key=lambda item: (
            -item[1],
            item[0],
        ),
    )

    rows = [
        (
            rank,
            team,
            rating,
            games_played[team],
            last_game[team].game_id,
            last_game[team].gameday,
            last_game[team].season,
        )
        for rank, (team, rating) in enumerate(
            sorted_ratings,
            start=1,
        )
    ]
    connection.execute(
        f"CREATE SCHEMA IF NOT EXISTS {TARGET_SCHEMA}"
    )
    connection.execute(
        f"""
        CREATE OR REPLACE TABLE {CURRENT_FULL_NAME} (
            elo_rank INTEGER,
            team VARCHAR,
            elo_rating DOUBLE,
            games_played INTEGER,
            last_game_id VARCHAR,
            as_of_gameday DATE,
            last_completed_season INTEGER
        )
        """
    )

    connection.executemany(
        f"""
        INSERT INTO {CURRENT_FULL_NAME}
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )

    logger.info(
        "Current Elo ratings created: %s teams.",
        len(rows),
    )


def validate_history_table(
    connection: duckdb.DuckDBPyConnection,
    expected_row_count: int,
) -> None:
    """Validate historical Elo predictions and calculations."""

    actual_row_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {HISTORY_FULL_NAME}
        """
    ).fetchone()[0]

    if actual_row_count != expected_row_count:
        raise RuntimeError(
            "Elo history row count does not match: "
            f"expected {expected_row_count}, "
            f"found {actual_row_count}."
        )

    duplicate_game_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM (
            SELECT game_id
            FROM {HISTORY_FULL_NAME}
            GROUP BY game_id
            HAVING COUNT(*) > 1
        )
        """
    ).fetchone()[0]

    if duplicate_game_count > 0:
        raise RuntimeError(
            "Duplicate game identifiers found in Elo history."
        )

    invalid_probability_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {HISTORY_FULL_NAME}
        WHERE home_win_probability IS NULL
           OR away_win_probability IS NULL
           OR home_win_probability NOT BETWEEN 0.0 AND 1.0
           OR away_win_probability NOT BETWEEN 0.0 AND 1.0
           OR ABS(
                home_win_probability
                + away_win_probability
                - 1.0
           ) > 0.000000001
        """
    ).fetchone()[0]

    if invalid_probability_count > 0:
        raise RuntimeError(
            "Invalid win probabilities found in Elo history."
        )

    invalid_rating_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {HISTORY_FULL_NAME}
        WHERE actual_home_score NOT IN (0.0, 0.5, 1.0)
           OR ABS(
                home_rating_post
                - home_rating_pre
                - home_rating_change
           ) > 0.000000001
           OR ABS(
                away_rating_post
                - away_rating_pre
                + home_rating_change
           ) > 0.000000001
        """
    ).fetchone()[0]

    if invalid_rating_count > 0:
        raise RuntimeError(
            "Invalid rating calculations found in Elo history."
        )

    logger.info(
        "Historical Elo predictions validated: %s rows.",
        actual_row_count,
    )


def validate_current_ratings_table(
    connection: duckdb.DuckDBPyConnection,
    expected_team_count: int,
) -> None:
    """Validate the current franchise Elo ratings table."""

    actual_team_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {CURRENT_FULL_NAME}
        """
    ).fetchone()[0]

    if actual_team_count != expected_team_count:
        raise RuntimeError(
            "Current Elo team count does not match: "
            f"expected {expected_team_count}, "
            f"found {actual_team_count}."
        )

    invalid_team_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM (
            SELECT team
            FROM {CURRENT_FULL_NAME}
            GROUP BY team
            HAVING COUNT(*) > 1
        )
        """
    ).fetchone()[0]

    if invalid_team_count > 0:
        raise RuntimeError(
            "Duplicate teams found in current Elo ratings."
        )

    invalid_rank_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {CURRENT_FULL_NAME}
        WHERE elo_rank < 1
           OR elo_rank > ?
           OR elo_rating IS NULL
           OR games_played < 1
        """,
        [expected_team_count],
    ).fetchone()[0]

    distinct_rank_count = connection.execute(
        f"""
        SELECT COUNT(DISTINCT elo_rank)
        FROM {CURRENT_FULL_NAME}
        """
    ).fetchone()[0]

    if (
        invalid_rank_count > 0
        or distinct_rank_count != expected_team_count
    ):
        raise RuntimeError(
            "Invalid ranks or values found in current Elo ratings."
        )

    logger.info(
        "Current Elo ratings validated: %s teams.",
        actual_team_count,
    )


def build_elo_ratings(
    database_file: Path = DATABASE_FILE,
) -> None:
    """Build historical and current Elo analytics tables."""

    validate_database_file(database_file)

    logger.info("Starting Elo ratings build...")

    with duckdb.connect(str(database_file)) as connection:
        validate_source_table(connection)
        validate_source_columns(connection)

        games = load_historical_games(connection)

        history_records, final_ratings = process_elo_history(
            games=games,
            home_advantage=BASELINE_HOME_ADVANTAGE,
        )

        connection.execute("BEGIN TRANSACTION")

        try:
            create_history_table(
                connection=connection,
                history_records=history_records,
            )
            create_current_ratings_table(
                connection=connection,
                history_records=history_records,
                final_ratings=final_ratings,
            )

            validate_history_table(
                connection=connection,
                expected_row_count=len(history_records),
            )
            validate_current_ratings_table(
                connection=connection,
                expected_team_count=len(final_ratings),
            )

            connection.execute("COMMIT")

            logger.info(
                "Elo ratings transaction committed."
            )
        except Exception:
            connection.execute("ROLLBACK")

            logger.exception(
                "Elo ratings build failed and was rolled back."
            )
            raise

    logger.info(
        "Elo ratings build completed: "
        "%s game predictions and %s team ratings.",
        len(history_records),
        len(final_ratings),
    )


def main() -> None:
    """Run the Elo ratings builder."""

    try:
        build_elo_ratings()
    except Exception:
        logger.exception(
            "Elo ratings builder execution failed."
        )
        raise


if __name__ == "__main__":
    main()