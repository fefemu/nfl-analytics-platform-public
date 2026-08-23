"""
Download and persist external unit and win-total team ratings.

Only pregame unit fields are persisted. Postgame fields are
intentionally excluded to prevent same-week target leakage.
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
from src.processing.normalize_external_team_strengths import (
    normalize_nfelounits_units,
    normalize_win_total_ratings,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

NFELOUNITS_URL = (
    "https://raw.githubusercontent.com/greerreNFL/"
    "nfelounits/refs/heads/main/Output/units.csv"
)
WIN_TOTAL_RATINGS_URL = (
    "https://raw.githubusercontent.com/greerreNFL/"
    "nfelosrs/main/wt_ratings.csv"
)

UNIT_TABLE = "processed.external_nfelounits_units"
WIN_TOTAL_TABLE = "processed.external_win_total_ratings"


def download_csv_with_retries(
    *,
    source_url: str,
    source_name: str,
    maximum_attempts: int = 4,
    initial_retry_delay_seconds: float = 1.0,
) -> pd.DataFrame:
    """Download one CSV source with exponential retry."""

    if maximum_attempts <= 0:
        raise ValueError("Maximum download attempts must be positive.")

    if initial_retry_delay_seconds < 0.0:
        raise ValueError("Initial retry delay must not be negative.")

    if not source_url.lower().startswith(("http://", "https://")):
        data = pd.read_csv(source_url)
    else:
        data = None

        for attempt in range(1, maximum_attempts + 1):
            try:
                response = requests.get(
                    source_url,
                    timeout=60,
                    headers={"User-Agent": "nfl-analytics-platform/0.1.0"},
                )
                response.raise_for_status()
                data = pd.read_csv(StringIO(response.text))
                break
            except (requests.RequestException, pd.errors.ParserError) as error:
                if attempt == maximum_attempts:
                    raise RuntimeError(
                        f"{source_name} download failed after "
                        f"{maximum_attempts} attempts."
                    ) from error

                delay = initial_retry_delay_seconds * (2 ** (attempt - 1))
                logger.warning(
                    "%s download attempt %s/%s failed: %s. "
                    "Retrying in %.1f seconds.",
                    source_name,
                    attempt,
                    maximum_attempts,
                    error,
                    delay,
                )
                time.sleep(delay)

    if data is None or data.empty:
        raise RuntimeError(f"Downloaded {source_name} data is empty.")

    return data


def persist_external_team_strengths(
    connection: duckdb.DuckDBPyConnection,
    *,
    unit_ratings: pd.DataFrame,
    win_total_ratings: pd.DataFrame,
    fetched_at: pd.Timestamp,
) -> None:
    """Persist both normalized sources in one transaction."""

    units = unit_ratings.assign(
        source_name="nfelounits_units",
        source_url=NFELOUNITS_URL,
        source_fetched_at=fetched_at,
    )
    win_totals = win_total_ratings.assign(
        source_name="wt_ratings",
        source_url=WIN_TOTAL_RATINGS_URL,
        source_fetched_at=fetched_at,
    )

    connection.execute("CREATE SCHEMA IF NOT EXISTS processed")
    connection.register("_external_units", units)
    connection.register("_external_win_totals", win_totals)

    try:
        connection.execute(
            f"CREATE OR REPLACE TABLE {UNIT_TABLE} AS "
            "SELECT * FROM _external_units"
        )
        connection.execute(
            f"CREATE OR REPLACE TABLE {WIN_TOTAL_TABLE} AS "
            "SELECT * FROM _external_win_totals"
        )
    finally:
        connection.unregister("_external_units")
        connection.unregister("_external_win_totals")


def validate_external_team_strengths(
    connection: duckdb.DuckDBPyConnection,
    *,
    expected_unit_rows: int,
    expected_win_total_rows: int,
) -> None:
    """Validate row uniqueness, latest season and model coverage."""

    unit_state = connection.execute(
        f"""
        SELECT
            COUNT(*),
            COUNT(DISTINCT (season, week, team)),
            MAX(season),
            MAX(week) FILTER (WHERE season = (SELECT MAX(season) FROM {UNIT_TABLE}))
        FROM {UNIT_TABLE}
        """
    ).fetchone()
    win_total_state = connection.execute(
        f"""
        SELECT
            COUNT(*),
            COUNT(DISTINCT (season, team)),
            MAX(season)
        FROM {WIN_TOTAL_TABLE}
        """
    ).fetchone()

    if unit_state[0] != expected_unit_rows or unit_state[1] != expected_unit_rows:
        raise RuntimeError("Persisted unit rating row count or keys are invalid.")

    if (
        win_total_state[0] != expected_win_total_rows
        or win_total_state[1] != expected_win_total_rows
    ):
        raise RuntimeError(
            "Persisted win-total rating row count or keys are invalid."
        )

    coverage = connection.execute(
        f"""
        SELECT
            COUNT(*) AS games,
            COUNT(*) FILTER (
                WHERE hu.team IS NOT NULL AND au.team IS NOT NULL
            ) AS unit_games,
            COUNT(*) FILTER (
                WHERE hw.team IS NOT NULL AND aw.team IS NOT NULL
            ) AS win_total_games
        FROM analytics.game_modeling_dataset AS games
        LEFT JOIN {UNIT_TABLE} AS hu
            ON games.season = hu.season
           AND games.week = hu.week
           AND games.home_team = hu.team
        LEFT JOIN {UNIT_TABLE} AS au
            ON games.season = au.season
           AND games.week = au.week
           AND games.away_team = au.team
        LEFT JOIN {WIN_TOTAL_TABLE} AS hw
            ON games.season = hw.season
           AND games.home_team = hw.team
        LEFT JOIN {WIN_TOTAL_TABLE} AS aw
            ON games.season = aw.season
           AND games.away_team = aw.team
        """
    ).fetchone()

    logger.info(
        "External strength tables validated: units=%s rows "
        "(latest %s week %s), win totals=%s rows (latest %s), "
        "model coverage units=%s/%s, win totals=%s/%s.",
        unit_state[0], unit_state[2], unit_state[3],
        win_total_state[0], win_total_state[2],
        coverage[1], coverage[0], coverage[2], coverage[0],
    )


def build_external_team_strengths(
    database_file: Path = DATABASE_FILE,
    unit_source_url: str = NFELOUNITS_URL,
    win_total_source_url: str = WIN_TOTAL_RATINGS_URL,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download, normalize, persist and validate both sources."""

    validate_database_file(database_file)
    unit_source = download_csv_with_retries(
        source_url=unit_source_url,
        source_name="nfelounits_units",
    )
    win_total_source = download_csv_with_retries(
        source_url=win_total_source_url,
        source_name="wt_ratings",
    )
    unit_ratings = normalize_nfelounits_units(unit_source)
    win_total_ratings = normalize_win_total_ratings(win_total_source)
    fetched_at = pd.Timestamp.now(tz="UTC")

    with duckdb.connect(str(database_file)) as connection:
        connection.begin()
        try:
            persist_external_team_strengths(
                connection,
                unit_ratings=unit_ratings,
                win_total_ratings=win_total_ratings,
                fetched_at=fetched_at,
            )
            validate_external_team_strengths(
                connection,
                expected_unit_rows=len(unit_ratings),
                expected_win_total_rows=len(win_total_ratings),
            )
            connection.commit()
        except Exception:
            connection.rollback()
            raise

    return unit_ratings, win_total_ratings


def main() -> None:
    """Build both external team-strength sources."""

    units, win_totals = build_external_team_strengths()
    logger.info(
        "External team-strength build completed: %s unit rows, "
        "%s win-total rows.",
        len(units),
        len(win_totals),
    )


if __name__ == "__main__":
    main()
