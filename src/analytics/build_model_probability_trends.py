"""Archive published win probabilities and build current prediction trends."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import duckdb


LOGGER = logging.getLogger(__name__)
ARCHIVE_TABLE = "analytics.published_game_prediction_archive"
CURRENT_TABLE = "analytics.current_game_probability_trends"
PREDICTION_TABLE = "analytics.current_game_predictions"


@dataclass(frozen=True)
class ProbabilityTrendConfig:
    """Shared trend classification settings, intentionally outside the UI."""

    neutral_threshold_pp: float = 0.5


DEFAULT_TREND_CONFIG = ProbabilityTrendConfig()


def _table_exists(connection: duckdb.DuckDBPyConnection, full_name: str) -> bool:
    schema, table = full_name.split(".", maxsplit=1)
    return bool(
        connection.execute(
            """
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_schema = ? AND table_name = ?
            """,
            [schema, table],
        ).fetchone()[0]
    )


def _ensure_archive_table(connection: duckdb.DuckDBPyConnection) -> None:
    connection.execute("CREATE SCHEMA IF NOT EXISTS analytics")
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {ARCHIVE_TABLE} (
            published_snapshot_id VARCHAR,
            game_id VARCHAR,
            season INTEGER,
            week INTEGER,
            team VARCHAR,
            outcome_side VARCHAR,
            win_probability DOUBLE,
            prediction_generated_at TIMESTAMP,
            archived_at TIMESTAMPTZ,
            publication_status VARCHAR
        )
        """
    )


def archive_previous_published_predictions(database_file: Path) -> int:
    """Archive the prediction state restored from the latest published release.

    This must run before a publishing refresh overwrites current predictions.
    The resulting operational database is published only after all quality gates,
    so an unsuccessful refresh cannot promote these or its new predictions.
    """

    with duckdb.connect(str(database_file)) as connection:
        if not _table_exists(connection, PREDICTION_TABLE):
            LOGGER.warning("No prior published predictions are available to archive.")
            return 0
        _ensure_archive_table(connection)
        snapshot_value = connection.execute(
            f"SELECT MAX(prediction_generated_at) FROM {PREDICTION_TABLE}"
        ).fetchone()[0]
        if snapshot_value is None:
            LOGGER.warning("Prior predictions have no generation timestamp; archive skipped.")
            return 0
        snapshot_id = f"published-{snapshot_value.isoformat()}"
        archived_at = datetime.now(timezone.utc)
        connection.execute("BEGIN TRANSACTION")
        try:
            connection.execute(
                f"DELETE FROM {ARCHIVE_TABLE} WHERE published_snapshot_id = ?",
                [snapshot_id],
            )
            connection.execute(
                f"""
                INSERT INTO {ARCHIVE_TABLE}
                SELECT ?, game_id, season, week, home_team, 'HOME',
                       home_win_probability, prediction_generated_at, ?, 'PUBLISHED'
                FROM {PREDICTION_TABLE}
                UNION ALL
                SELECT ?, game_id, season, week, away_team, 'AWAY',
                       away_win_probability, prediction_generated_at, ?, 'PUBLISHED'
                FROM {PREDICTION_TABLE}
                """,
                [snapshot_id, archived_at, snapshot_id, archived_at],
            )
            count = connection.execute(
                f"SELECT COUNT(*) FROM {ARCHIVE_TABLE} WHERE published_snapshot_id = ?",
                [snapshot_id],
            ).fetchone()[0]
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise
    LOGGER.info("Archived %s prior published team probabilities (%s).", count, snapshot_id)
    return int(count)


def build_current_game_probability_trends(
    database_file: Path,
    config: ProbabilityTrendConfig = DEFAULT_TREND_CONFIG,
) -> int:
    """Build one prepared current-vs-previous trend record per matchup."""

    if config.neutral_threshold_pp < 0:
        raise ValueError("neutral_threshold_pp must be non-negative")
    with duckdb.connect(str(database_file)) as connection:
        if not _table_exists(connection, PREDICTION_TABLE):
            raise RuntimeError(f"Missing required table: {PREDICTION_TABLE}")
        _ensure_archive_table(connection)
        threshold = float(config.neutral_threshold_pp)
        connection.execute(
            f"""
            CREATE OR REPLACE TABLE {CURRENT_TABLE} AS
            WITH prior_ranked AS (
                SELECT *,
                       ROW_NUMBER() OVER (
                           PARTITION BY game_id, team, outcome_side
                           ORDER BY prediction_generated_at DESC, archived_at DESC
                       ) AS row_number
                FROM {ARCHIVE_TABLE}
                WHERE publication_status = 'PUBLISHED'
            ), prior AS (
                SELECT * EXCLUDE (row_number)
                FROM prior_ranked
                WHERE row_number = 1
            ), team_trends AS (
                SELECT current.game_id, current.season, current.week,
                       current.home_team AS team, 'HOME' AS outcome_side,
                       current.home_win_probability AS current_probability,
                       prior.win_probability AS previous_probability,
                       100.0 * (current.home_win_probability - prior.win_probability)
                           AS probability_change_pp,
                       prior.prediction_generated_at AS previous_prediction_generated_at,
                       current.prediction_generated_at AS current_prediction_generated_at
                FROM {PREDICTION_TABLE} AS current
                LEFT JOIN prior ON prior.game_id = current.game_id
                    AND prior.team = current.home_team AND prior.outcome_side = 'HOME'
                UNION ALL
                SELECT current.game_id, current.season, current.week,
                       current.away_team AS team, 'AWAY' AS outcome_side,
                       current.away_win_probability AS current_probability,
                       prior.win_probability AS previous_probability,
                       100.0 * (current.away_win_probability - prior.win_probability)
                           AS probability_change_pp,
                       prior.prediction_generated_at AS previous_prediction_generated_at,
                       current.prediction_generated_at AS current_prediction_generated_at
                FROM {PREDICTION_TABLE} AS current
                LEFT JOIN prior ON prior.game_id = current.game_id
                    AND prior.team = current.away_team AND prior.outcome_side = 'AWAY'
            ), classified AS (
                SELECT *,
                    CASE
                        WHEN previous_probability IS NULL THEN 'NEW'
                        WHEN ABS(probability_change_pp) < ? THEN 'UNCHANGED'
                        WHEN probability_change_pp > 0 THEN 'INCREASE'
                        ELSE 'DECREASE'
                    END AS trend_direction,
                    ?::DOUBLE AS neutral_threshold_pp
                FROM team_trends
            )
            SELECT
                game_id, season, week,
                MAX(CASE WHEN outcome_side = 'HOME' THEN team END) AS home_team,
                MAX(CASE WHEN outcome_side = 'AWAY' THEN team END) AS away_team,
                MAX(CASE WHEN outcome_side = 'HOME' THEN previous_probability END)
                    AS home_previous_win_probability,
                MAX(CASE WHEN outcome_side = 'AWAY' THEN previous_probability END)
                    AS away_previous_win_probability,
                MAX(CASE WHEN outcome_side = 'HOME' THEN probability_change_pp END)
                    AS home_probability_change_pp,
                MAX(CASE WHEN outcome_side = 'AWAY' THEN probability_change_pp END)
                    AS away_probability_change_pp,
                MAX(CASE WHEN outcome_side = 'HOME' THEN trend_direction END)
                    AS home_probability_trend,
                MAX(CASE WHEN outcome_side = 'AWAY' THEN trend_direction END)
                    AS away_probability_trend,
                MAX(previous_prediction_generated_at) AS previous_prediction_generated_at,
                MAX(current_prediction_generated_at) AS current_prediction_generated_at,
                MAX(neutral_threshold_pp) AS neutral_threshold_pp
            FROM classified
            GROUP BY game_id, season, week
            """,
            [threshold, threshold],
        )
        invalid = connection.execute(
            f"""
            SELECT COUNT(*) FROM {CURRENT_TABLE}
            WHERE home_probability_trend NOT IN ('NEW','UNCHANGED','INCREASE','DECREASE')
               OR away_probability_trend NOT IN ('NEW','UNCHANGED','INCREASE','DECREASE')
               OR (home_probability_change_pp IS NOT NULL
                   AND ABS(home_probability_change_pp + away_probability_change_pp) > 0.000001)
            """
        ).fetchone()[0]
        if invalid:
            raise RuntimeError("Current probability trend validation failed.")
        count = connection.execute(f"SELECT COUNT(*) FROM {CURRENT_TABLE}").fetchone()[0]
    LOGGER.info("Built %s current game probability trend rows.", count)
    return int(count)
