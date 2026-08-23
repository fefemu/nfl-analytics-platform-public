"""
NFL Analytics Platform
External nfelo Game Rating Builder

Purpose:
    Download, normalize and persist nfelo game-level
    pregame ratings for external model benchmarking.

Author:
    Ferenc Kaizer

Version:
    0.1.0
"""

import logging
import time
from io import StringIO
from pathlib import Path

import duckdb
import pandas as pd
import requests

from src.modeling.train_logistic_baseline import (
    DATABASE_FILE,
    validate_database_file,
)
from src.processing.normalize_nfelo_game_ratings import (
    normalize_nfelo_game_ratings,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

EXTERNAL_NFELO_GAMES_URL = (
    "https://raw.githubusercontent.com/"
    "greerreNFL/nfelo/refs/heads/main/"
    "output_data/nfelo_games.csv"
)

TARGET_SCHEMA = "processed"
TARGET_TABLE = "external_nfelo_game_ratings"
TARGET_FULL_NAME = (
    f"{TARGET_SCHEMA}.{TARGET_TABLE}"
)

MODELING_DATASET_FULL_NAME = (
    "analytics.game_modeling_dataset"
)


def download_external_nfelo_games(
    source_url: str = EXTERNAL_NFELO_GAMES_URL,
    maximum_attempts: int = 4,
    initial_retry_delay_seconds: float = 1.0,
) -> pd.DataFrame:
    """Download nfelo data with controlled retries."""

    if maximum_attempts <= 0:
        raise ValueError(
            "Maximum download attempts must "
            "be positive."
        )

    if initial_retry_delay_seconds < 0.0:
        raise ValueError(
            "Initial retry delay must not "
            "be negative."
        )

    logger.info(
        "Downloading external nfelo game ratings..."
    )

    is_remote_source = source_url.lower().startswith(
        (
            "http://",
            "https://",
        )
    )

    if not is_remote_source:
        source_data = pd.read_csv(
            source_url
        )
    else:
        source_data = None

        for attempt in range(
            1,
            maximum_attempts + 1,
        ):
            try:
                response = requests.get(
                    source_url,
                    timeout=60,
                    headers={
                        "User-Agent": (
                            "nfl-analytics-platform/0.1.0"
                        ),
                    },
                )

                response.raise_for_status()

                source_data = pd.read_csv(
                    StringIO(response.text)
                )

                break

            except (
                requests.RequestException,
                pd.errors.ParserError,
            ) as error:
                if attempt == maximum_attempts:
                    raise RuntimeError(
                        "External nfelo download failed "
                        f"after {maximum_attempts} attempts."
                    ) from error

                retry_delay = (
                    initial_retry_delay_seconds
                    * (2 ** (attempt - 1))
                )

                logger.warning(
                    "External nfelo download attempt "
                    "%s/%s failed: %s. "
                    "Retrying in %.1f seconds.",
                    attempt,
                    maximum_attempts,
                    error,
                    retry_delay,
                )

                time.sleep(retry_delay)

    if (
        source_data is None
        or source_data.empty
    ):
        raise RuntimeError(
            "Downloaded nfelo game data is empty."
        )

    logger.info(
        "External nfelo source downloaded: %s rows.",
        len(source_data),
    )

    return source_data


def create_external_nfelo_game_ratings_table(
    connection: duckdb.DuckDBPyConnection,
    normalized_data: pd.DataFrame,
) -> None:
    """Persist normalized external nfelo ratings."""

    if normalized_data.empty:
        raise ValueError(
            "Normalized nfelo data must not be empty."
        )

    persisted = normalized_data.copy()

    persisted["source_url"] = (
        EXTERNAL_NFELO_GAMES_URL
    )

    persisted["source_fetched_at"] = (
        pd.Timestamp.now(tz="UTC")
    )

    connection.execute(
        f"CREATE SCHEMA IF NOT EXISTS {TARGET_SCHEMA}"
    )

    connection.register(
        "_external_nfelo_game_ratings",
        persisted,
    )

    try:
        connection.execute(
            f"""
            CREATE OR REPLACE TABLE
                {TARGET_FULL_NAME}
            AS
            SELECT *
            FROM _external_nfelo_game_ratings
            """
        )
    finally:
        connection.unregister(
            "_external_nfelo_game_ratings"
        )


def validate_external_nfelo_game_ratings_table(
    connection: duckdb.DuckDBPyConnection,
    expected_row_count: int,
) -> None:
    """Validate persisted external rating history."""

    actual_row_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        """
    ).fetchone()[0]

    if actual_row_count != expected_row_count:
        raise RuntimeError(
            "External nfelo row count does not match: "
            f"expected {expected_row_count}, "
            f"found {actual_row_count}."
        )

    duplicate_source_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM (
            SELECT source_game_id
            FROM {TARGET_FULL_NAME}
            GROUP BY source_game_id
            HAVING COUNT(*) > 1
        )
        """
    ).fetchone()[0]

    if duplicate_source_count > 0:
        raise RuntimeError(
            "Duplicate external nfelo source game "
            "identifiers found."
        )

    duplicate_normalized_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM (
            SELECT normalized_game_id
            FROM {TARGET_FULL_NAME}
            GROUP BY normalized_game_id
            HAVING COUNT(*) > 1
        )
        """
    ).fetchone()[0]

    if duplicate_normalized_count > 0:
        raise RuntimeError(
            "Duplicate normalized external nfelo "
            "game identifiers found."
        )

    invalid_identifier_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE source_name <> 'nfelo_games'
           OR source_game_id IS NULL
           OR normalized_game_id IS NULL
           OR source_season IS NULL
           OR source_week IS NULL
           OR away_team IS NULL
           OR home_team IS NULL
           OR away_team = home_team
           OR source_url IS NULL
           OR source_fetched_at IS NULL
        """
    ).fetchone()[0]

    if invalid_identifier_count > 0:
        raise RuntimeError(
            "Invalid external nfelo identifiers "
            "or metadata found."
        )

    invalid_rating_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE starting_nfelo_home IS NULL
           OR starting_nfelo_away IS NULL
           OR nfelo_dif_base IS NULL
           OR NOT isfinite(starting_nfelo_home)
           OR NOT isfinite(starting_nfelo_away)
           OR NOT isfinite(nfelo_dif_base)
        """
    ).fetchone()[0]

    if invalid_rating_count > 0:
        raise RuntimeError(
            "Invalid external nfelo ratings found."
        )

    invalid_probability_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE nfelo_home_probability_open <= 0.0
           OR nfelo_home_probability_open >= 1.0
           OR nfelo_home_probability_close <= 0.0
           OR nfelo_home_probability_close >= 1.0
        """
    ).fetchone()[0]

    if invalid_probability_count > 0:
        raise RuntimeError(
            "Invalid external nfelo probabilities found."
        )

    latest_state = connection.execute(
        f"""
        SELECT
            MAX(source_season) AS latest_season,
            MAX(source_week) FILTER (
                WHERE source_season = (
                    SELECT MAX(source_season)
                    FROM {TARGET_FULL_NAME}
                )
            ) AS latest_week
        FROM {TARGET_FULL_NAME}
        """
    ).fetchone()

    if (
        latest_state[0] is None
        or latest_state[1] is None
    ):
        raise RuntimeError(
            "External nfelo latest state is missing."
        )

    logger.info(
        "External nfelo table validated: %s rows, "
        "latest season %s week %s.",
        actual_row_count,
        latest_state[0],
        latest_state[1],
    )


def validate_modeling_game_coverage(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Validate external coverage of modeling games."""

    table_exists = connection.execute(
        """
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_schema = 'analytics'
          AND table_name = 'game_modeling_dataset'
        """
    ).fetchone()[0]

    if table_exists == 0:
        logger.info(
            "Modeling dataset is not available; "
            "external coverage validation skipped."
        )
        return

    coverage = connection.execute(
        f"""
        SELECT
            COUNT(*) AS modeling_game_count,
            COUNT(external.normalized_game_id)
                AS matched_game_count
        FROM {MODELING_DATASET_FULL_NAME}
            AS modeling
        LEFT JOIN {TARGET_FULL_NAME}
            AS external
            ON modeling.game_id
                = external.normalized_game_id
        """
    ).fetchone()

    modeling_game_count = int(
        coverage[0]
    )

    matched_game_count = int(
        coverage[1]
    )

    if matched_game_count != modeling_game_count:
        raise RuntimeError(
            "External nfelo modeling coverage does "
            f"not match: {matched_game_count} of "
            f"{modeling_game_count} games matched."
        )

    logger.info(
        "External nfelo modeling coverage validated: "
        "%s of %s games matched.",
        matched_game_count,
        modeling_game_count,
    )


def build_external_nfelo_game_ratings(
    database_file: Path = DATABASE_FILE,
    source_url: str = EXTERNAL_NFELO_GAMES_URL,
) -> pd.DataFrame:
    """Download, normalize and persist external ratings."""

    validate_database_file(database_file)

    source_data = download_external_nfelo_games(
        source_url=source_url
    )

    normalized_data = (
        normalize_nfelo_game_ratings(
            source_data
        )
    )

    logger.info(
        "External nfelo data normalized: %s rows.",
        len(normalized_data),
    )

    with duckdb.connect(
        str(database_file)
    ) as connection:
        connection.execute("BEGIN TRANSACTION")

        try:
            create_external_nfelo_game_ratings_table(
                connection=connection,
                normalized_data=normalized_data,
            )

            validate_external_nfelo_game_ratings_table(
                connection=connection,
                expected_row_count=len(
                    normalized_data
                ),
            )

            validate_modeling_game_coverage(
                connection
            )

            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise

    logger.info(
        "External nfelo game rating build completed: "
        "%s rows in %s.",
        len(normalized_data),
        TARGET_FULL_NAME,
    )

    return normalized_data


def main() -> None:
    """Run the external nfelo rating builder."""

    try:
        build_external_nfelo_game_ratings()
    except Exception:
        logger.exception(
            "External nfelo game rating build failed."
        )
        raise


if __name__ == "__main__":
    main()