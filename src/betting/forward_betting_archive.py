"""Persist immutable betting snapshots and derive pre-kickoff market movement."""

from datetime import datetime, timezone

import duckdb
import pandas as pd

from src.betting.build_current_betting_board import BOARD_COLUMNS

ARCHIVE_TABLE = "analytics.forward_betting_board_archive"
MARKET_MOVEMENT_VIEW = "analytics.forward_tip_market_movement"
LEGACY_CLV_VIEW = "analytics.forward_tip_clv"
ARCHIVE_COLUMNS = (
    "archive_key", "refresh_run_id", *BOARD_COLUMNS, "is_tip_candidate", "archived_at",
)
IMMUTABLE_PAYLOAD_COLUMNS = (*BOARD_COLUMNS, "is_tip_candidate")


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
    for column in (
        "fetched_at",
        "commence_time",
        "prediction_generated_at",
        "betting_board_generated_at",
    ):
        rows[column] = pd.to_datetime(rows[column], utc=True, errors="coerce")
    if archived_at is None:
        archived_at = datetime.now(timezone.utc)
    lock_time = pd.Timestamp(archived_at)
    if lock_time.tzinfo is None:
        lock_time = lock_time.tz_localize("UTC")
    else:
        lock_time = lock_time.tz_convert("UTC")
    rows["archived_at"] = lock_time
    rows = rows.loc[
        rows["commence_time"].notna()
        & rows["fetched_at"].notna()
        & rows["prediction_generated_at"].notna()
        & rows["betting_board_generated_at"].notna()
        & (rows["commence_time"] > rows["fetched_at"])
        & (rows["commence_time"] > rows["prediction_generated_at"])
        & (rows["commence_time"] > rows["betting_board_generated_at"])
        & (rows["commence_time"] > rows["archived_at"])
    ].copy()
    if rows.empty:
        raise RuntimeError("No future betting-board rows are available for archival.")
    identity = rows[[
        "snapshot_id", "game_id", "market_key", "outcome_type", "point",
        "best_bookmaker_key",
    ]].fillna("<NULL>").astype(str).agg("|".join, axis=1)
    rows.insert(0, "refresh_run_id", str(refresh_run_id))
    rows.insert(0, "archive_key", identity.map(lambda value: __import__("hashlib").sha256(value.encode()).hexdigest()))
    rows["is_tip_candidate"] = rows["positive_expected_value"].astype(bool)
    return rows.loc[:, ARCHIVE_COLUMNS].sort_values(
        ["fetched_at", "commence_time", "game_id", "market_key", "outcome_type"]
    ).reset_index(drop=True)


def persist_forward_archive(
    connection: duckdb.DuckDBPyConnection,
    rows: pd.DataFrame,
) -> int:
    """Append unseen board identities and rebuild the market-movement view."""
    connection.execute("CREATE SCHEMA IF NOT EXISTS analytics")
    connection.register("_forward_archive_rows", rows)
    try:
        connection.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {ARCHIVE_TABLE} AS
            SELECT * FROM _forward_archive_rows WHERE FALSE
            """
        )
        distinct_payload = " OR ".join(
            f'target."{column}" IS DISTINCT FROM source."{column}"'
            for column in IMMUTABLE_PAYLOAD_COLUMNS
        )
        conflicts = connection.execute(
            f"""
            SELECT COUNT(*)
            FROM _forward_archive_rows AS source
            JOIN {ARCHIVE_TABLE} AS target
              ON target.archive_key = source.archive_key
            WHERE {distinct_payload}
            """
        ).fetchone()[0]
        if conflicts:
            raise RuntimeError(
                "Forward archive key already exists with different locked values."
            )
        connection.execute(
            f"""
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
    connection.execute(f"DROP VIEW IF EXISTS {LEGACY_CLV_VIEW}")
    connection.execute(
        f"""
        CREATE OR REPLACE VIEW {MARKET_MOVEMENT_VIEW} AS
        WITH ranked_entries AS (
            SELECT archive.*,
                   ROW_NUMBER() OVER (
                       PARTITION BY game_id, market_key, outcome_type
                       ORDER BY fetched_at, archived_at, expected_value_percent DESC,
                                point DESC NULLS LAST, archive_key
                   ) AS entry_rank
            FROM {ARCHIVE_TABLE} AS archive
            WHERE is_tip_candidate
        ),
        entries AS (
            SELECT * FROM ranked_entries WHERE entry_rank = 1
        )
        SELECT
            entry.archive_key AS entry_archive_key,
            entry.refresh_run_id AS entry_refresh_run_id,
            entry.snapshot_id AS entry_snapshot_id,
            entry.fetched_at AS entry_fetched_at,
            entry.game_id,
            entry.season,
            entry.week,
            entry.commence_time,
            entry.home_team,
            entry.away_team,
            entry.market_key,
            entry.market_name,
            entry.outcome_name,
            entry.outcome_type,
            entry.market_line AS entry_market_line,
            entry.point AS entry_point,
            entry.best_bookmaker_key AS entry_bookmaker_key,
            entry.best_bookmaker_title AS entry_bookmaker,
            entry.best_american_price AS entry_american_price,
            entry.best_decimal_odds AS entry_decimal_odds,
            1.0 / entry.best_decimal_odds AS entry_implied_probability,
            entry.model_name AS entry_model_name,
            entry.model_version AS entry_model_version,
            entry.prediction_mode AS entry_prediction_mode,
            entry.prediction_generated_at AS entry_prediction_generated_at,
            entry.model_probability AS entry_model_probability,
            entry.expected_value_percent AS entry_expected_value_percent,
            later.archive_key AS latest_archive_key,
            later.snapshot_id AS latest_snapshot_id,
            later.fetched_at AS latest_fetched_at,
            later.market_line AS latest_market_line,
            later.point AS latest_point,
            later.best_bookmaker_key AS latest_bookmaker_key,
            later.best_bookmaker_title AS latest_bookmaker,
            later.best_american_price AS latest_american_price,
            later.best_decimal_odds AS latest_decimal_odds,
            CASE WHEN later.best_decimal_odds IS NOT NULL
                THEN 1.0 / later.best_decimal_odds
            END AS latest_implied_probability,
            CASE WHEN later.best_decimal_odds IS NOT NULL THEN
                100.0 * (1.0 / later.best_decimal_odds - 1.0 / entry.best_decimal_odds)
            END AS price_movement_implied_probability_pp,
            CASE
                WHEN later.point IS NULL OR entry.point IS NULL THEN NULL
                WHEN entry.market_key = 'spreads' THEN entry.point - later.point
                WHEN entry.market_key = 'totals' AND LOWER(entry.outcome_type) = 'over'
                    THEN later.point - entry.point
                WHEN entry.market_key = 'totals' AND LOWER(entry.outcome_type) = 'under'
                    THEN entry.point - later.point
            END AS entry_line_advantage_points,
            CASE WHEN later.fetched_at IS NOT NULL
                THEN DATE_DIFF('minute', later.fetched_at, entry.commence_time)
            END AS latest_minutes_before_kickoff,
            CASE
                WHEN later.fetched_at IS NULL THEN 'NO_LATER_SNAPSHOT'
                WHEN COALESCE(
                    CASE
                        WHEN entry.market_key = 'spreads' THEN entry.point - later.point
                        WHEN entry.market_key = 'totals' AND LOWER(entry.outcome_type) = 'over'
                            THEN later.point - entry.point
                        WHEN entry.market_key = 'totals' AND LOWER(entry.outcome_type) = 'under'
                            THEN entry.point - later.point
                    END, 0.0
                ) > 0.000001 THEN 'POSITIVE'
                WHEN COALESCE(
                    CASE
                        WHEN entry.market_key = 'spreads' THEN entry.point - later.point
                        WHEN entry.market_key = 'totals' AND LOWER(entry.outcome_type) = 'over'
                            THEN later.point - entry.point
                        WHEN entry.market_key = 'totals' AND LOWER(entry.outcome_type) = 'under'
                            THEN entry.point - later.point
                    END, 0.0
                ) < -0.000001 THEN 'NEGATIVE'
                WHEN 100.0 * (1.0 / later.best_decimal_odds - 1.0 / entry.best_decimal_odds) > 0.000001
                    THEN 'POSITIVE'
                WHEN 100.0 * (1.0 / later.best_decimal_odds - 1.0 / entry.best_decimal_odds) < -0.000001
                    THEN 'NEGATIVE'
                ELSE 'UNCHANGED'
            END AS market_movement_direction,
            later.fetched_at IS NOT NULL AS has_latest_pregame_comparison,
            'LATEST_PRE_KICKOFF' AS comparison_type,
            FALSE AS is_closing_snapshot,
            FALSE AS is_clv
        FROM entries AS entry
        LEFT JOIN LATERAL (
            SELECT candidate.*
            FROM {ARCHIVE_TABLE} AS candidate
            WHERE candidate.game_id = entry.game_id
              AND candidate.market_key = entry.market_key
              AND candidate.outcome_type = entry.outcome_type
              AND candidate.fetched_at > entry.fetched_at
              AND candidate.fetched_at < entry.commence_time
            ORDER BY candidate.fetched_at DESC, candidate.archived_at DESC,
                     candidate.expected_value_percent DESC, candidate.point DESC NULLS LAST,
                     candidate.archive_key
            LIMIT 1
        ) AS later ON TRUE
        """
    )
    return int(connection.execute(f"SELECT COUNT(*) FROM {ARCHIVE_TABLE}").fetchone()[0])


def validate_forward_archive(connection: duckdb.DuckDBPyConnection) -> None:
    """Reject invalid archive rows and market-movement comparisons."""
    invalid = connection.execute(
        f"""
        SELECT COUNT(*) FROM {ARCHIVE_TABLE}
        WHERE commence_time <= fetched_at
           OR commence_time <= prediction_generated_at
           OR commence_time <= betting_board_generated_at
           OR commence_time <= archived_at
           OR archive_key IS NULL
           OR refresh_run_id IS NULL
           OR archived_at IS NULL
           OR is_tip_candidate <> positive_expected_value
        """
    ).fetchone()[0]
    duplicates = connection.execute(
        f"SELECT COUNT(*) FROM (SELECT archive_key FROM {ARCHIVE_TABLE} GROUP BY 1 HAVING COUNT(*) > 1)"
    ).fetchone()[0]
    invalid_movement = connection.execute(
        f"""
        SELECT COUNT(*) FROM {MARKET_MOVEMENT_VIEW}
        WHERE comparison_type <> 'LATEST_PRE_KICKOFF'
           OR is_closing_snapshot
           OR is_clv
           OR market_movement_direction NOT IN (
               'POSITIVE', 'NEGATIVE', 'UNCHANGED', 'NO_LATER_SNAPSHOT'
           )
           OR (has_latest_pregame_comparison AND (
               latest_fetched_at <= entry_fetched_at
               OR latest_fetched_at >= commence_time
           ))
           OR (NOT has_latest_pregame_comparison AND (
               latest_fetched_at IS NOT NULL
               OR market_movement_direction <> 'NO_LATER_SNAPSHOT'
           ))
        """
    ).fetchone()[0]
    duplicate_entries = connection.execute(
        f"""
        SELECT COUNT(*) FROM (
            SELECT game_id, market_key, outcome_type
            FROM {MARKET_MOVEMENT_VIEW}
            GROUP BY ALL HAVING COUNT(*) > 1
        )
        """
    ).fetchone()[0]
    if invalid or duplicates or invalid_movement or duplicate_entries:
        raise RuntimeError("Forward betting archive validation failed.")
