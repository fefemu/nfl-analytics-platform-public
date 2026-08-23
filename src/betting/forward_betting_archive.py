"""Persist immutable future betting-board snapshots and prospective CLV."""

from datetime import datetime, timezone

import duckdb
import pandas as pd

from src.betting.build_current_betting_board import BOARD_COLUMNS

ARCHIVE_TABLE = "analytics.forward_betting_board_archive"
CLV_VIEW = "analytics.forward_tip_clv"
ARCHIVE_COLUMNS = (
    "archive_key", "refresh_run_id", *BOARD_COLUMNS, "is_tip_candidate", "archived_at",
)


def prepare_forward_archive_rows(
    board: pd.DataFrame,
    refresh_run_id: str,
    archived_at: datetime | None = None,
) -> pd.DataFrame:
    """Keep only pregame rows and create stable immutable identifiers."""
    missing = sorted(set(BOARD_COLUMNS) - set(board.columns))
    if missing:
        raise ValueError("Current betting board is missing columns: " + ", ".join(missing))
    if not str(refresh_run_id).strip():
        raise ValueError("Refresh run ID must not be empty.")
    rows = board.loc[:, BOARD_COLUMNS].copy()
    rows["fetched_at"] = pd.to_datetime(rows["fetched_at"], utc=True)
    rows["commence_time"] = pd.to_datetime(rows["commence_time"], utc=True)
    rows = rows.loc[rows["commence_time"] > rows["fetched_at"]].copy()
    if rows.empty:
        raise RuntimeError("No future betting-board rows are available for archival.")
    identity = rows[[
        "snapshot_id", "game_id", "market_key", "outcome_type", "point",
        "best_bookmaker_key",
    ]].fillna("<NULL>").astype(str).agg("|".join, axis=1)
    rows.insert(0, "refresh_run_id", str(refresh_run_id))
    rows.insert(0, "archive_key", identity.map(lambda value: __import__("hashlib").sha256(value.encode()).hexdigest()))
    rows["is_tip_candidate"] = rows["positive_expected_value"].astype(bool)
    if archived_at is None:
        archived_at = datetime.now(timezone.utc)
    rows["archived_at"] = pd.Timestamp(archived_at)
    return rows.loc[:, ARCHIVE_COLUMNS].sort_values(
        ["fetched_at", "commence_time", "game_id", "market_key", "outcome_type"]
    ).reset_index(drop=True)


def persist_forward_archive(
    connection: duckdb.DuckDBPyConnection,
    rows: pd.DataFrame,
) -> int:
    """Append only previously unseen board identities and rebuild the CLV view."""
    connection.execute("CREATE SCHEMA IF NOT EXISTS analytics")
    connection.register("_forward_archive_rows", rows)
    try:
        connection.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {ARCHIVE_TABLE} AS
            SELECT * FROM _forward_archive_rows WHERE FALSE;
            INSERT INTO {ARCHIVE_TABLE}
            SELECT source.* FROM _forward_archive_rows AS source
            WHERE NOT EXISTS (
                SELECT 1 FROM {ARCHIVE_TABLE} AS target
                WHERE target.archive_key = source.archive_key
            );
            """
        )
    finally:
        connection.unregister("_forward_archive_rows")
    connection.execute(
        f"""
        CREATE OR REPLACE VIEW {CLV_VIEW} AS
        SELECT
            tip.archive_key,
            tip.refresh_run_id,
            tip.snapshot_id AS tip_snapshot_id,
            tip.fetched_at AS tip_fetched_at,
            tip.game_id,
            tip.season,
            tip.week,
            tip.commence_time,
            tip.home_team,
            tip.away_team,
            tip.market_key,
            tip.market_line,
            tip.outcome_name,
            tip.outcome_type,
            tip.point,
            tip.best_bookmaker_title AS tip_bookmaker,
            tip.best_decimal_odds AS tip_decimal_odds,
            tip.model_probability,
            tip.expected_value_percent,
            later.snapshot_id AS comparison_snapshot_id,
            later.fetched_at AS comparison_fetched_at,
            later.best_bookmaker_title AS comparison_bookmaker,
            later.best_decimal_odds AS comparison_decimal_odds,
            CASE WHEN later.best_decimal_odds IS NOT NULL THEN
                100.0 * (1.0 / later.best_decimal_odds - 1.0 / tip.best_decimal_odds)
            END AS clv_probability_percentage_points,
            later.fetched_at IS NOT NULL AS has_later_market_comparison
        FROM {ARCHIVE_TABLE} AS tip
        LEFT JOIN LATERAL (
            SELECT candidate.*
            FROM {ARCHIVE_TABLE} AS candidate
            WHERE candidate.game_id = tip.game_id
              AND candidate.market_key = tip.market_key
              AND candidate.outcome_type = tip.outcome_type
              AND candidate.point IS NOT DISTINCT FROM tip.point
              AND candidate.fetched_at > tip.fetched_at
              AND candidate.fetched_at < tip.commence_time
            ORDER BY candidate.fetched_at DESC
            LIMIT 1
        ) AS later ON TRUE
        WHERE tip.is_tip_candidate
        """
    )
    return int(connection.execute(f"SELECT COUNT(*) FROM {ARCHIVE_TABLE}").fetchone()[0])


def validate_forward_archive(connection: duckdb.DuckDBPyConnection) -> None:
    """Reject duplicates, post-kickoff rows and invalid candidate flags."""
    invalid = connection.execute(
        f"""
        SELECT COUNT(*) FROM {ARCHIVE_TABLE}
        WHERE commence_time <= fetched_at
           OR archive_key IS NULL
           OR refresh_run_id IS NULL
           OR archived_at IS NULL
           OR is_tip_candidate <> positive_expected_value
        """
    ).fetchone()[0]
    duplicates = connection.execute(
        f"SELECT COUNT(*) FROM (SELECT archive_key FROM {ARCHIVE_TABLE} GROUP BY 1 HAVING COUNT(*) > 1)"
    ).fetchone()[0]
    if invalid or duplicates:
        raise RuntimeError("Forward betting archive validation failed.")
