-- =====================================================
-- NFL Analytics Platform
-- File: 004_odds_data_quality_checks.sql
--
-- Purpose:
--     Execute data quality checks against the latest
--     normalized Odds API snapshot in DuckDB.
--
-- Author:
--     Ferenc Kaizer
--
-- Version:
--     0.1.0
-- =====================================================


WITH latest_snapshot AS (
    SELECT snapshot_id
    FROM raw.odds_snapshots
    ORDER BY fetched_at DESC
    LIMIT 1
),

quality_checks AS (
    SELECT
        'snapshot_event_count_matches' AS check_name,
        ABS(
            snapshot.event_count
            - (
                SELECT COUNT(*)
                FROM raw.odds_events AS event
                WHERE event.snapshot_id = snapshot.snapshot_id
            )
        ) AS issue_count
    FROM raw.odds_snapshots AS snapshot
    INNER JOIN latest_snapshot AS latest
        ON snapshot.snapshot_id = latest.snapshot_id

    UNION ALL

    SELECT
        'missing_event_identifier',
        COUNT(*)
    FROM raw.odds_events AS event
    INNER JOIN latest_snapshot AS latest
        ON event.snapshot_id = latest.snapshot_id
    WHERE event.event_id IS NULL

    UNION ALL

    SELECT
        'invalid_team_assignment',
        COUNT(*)
    FROM raw.odds_events AS event
    INNER JOIN latest_snapshot AS latest
        ON event.snapshot_id = latest.snapshot_id
    WHERE event.home_team IS NULL
       OR event.away_team IS NULL
       OR event.home_team = event.away_team

    UNION ALL

    SELECT
        'duplicate_event',
        COUNT(*)
    FROM (
        SELECT
            event.snapshot_id,
            event.event_id
        FROM raw.odds_events AS event
        INNER JOIN latest_snapshot AS latest
            ON event.snapshot_id = latest.snapshot_id
        GROUP BY
            event.snapshot_id,
            event.event_id
        HAVING COUNT(*) > 1
    )

    UNION ALL

    SELECT
        'orphan_market_record',
        COUNT(*)
    FROM raw.odds_markets AS market
    INNER JOIN latest_snapshot AS latest
        ON market.snapshot_id = latest.snapshot_id
    LEFT JOIN raw.odds_events AS event
        ON market.snapshot_id = event.snapshot_id
       AND market.event_id = event.event_id
    WHERE event.event_id IS NULL

    UNION ALL

    SELECT
        'unsupported_market',
        COUNT(*)
    FROM raw.odds_markets AS market
    INNER JOIN latest_snapshot AS latest
        ON market.snapshot_id = latest.snapshot_id
    WHERE market.market_key NOT IN (
        'h2h',
        'spreads',
        'totals'
    )

    UNION ALL

    SELECT
        'missing_market_value',
        COUNT(*)
    FROM raw.odds_markets AS market
    INNER JOIN latest_snapshot AS latest
        ON market.snapshot_id = latest.snapshot_id
    WHERE market.bookmaker_key IS NULL
       OR market.market_key IS NULL
       OR market.outcome_name IS NULL
       OR market.price IS NULL

    UNION ALL

    SELECT
        'invalid_american_price',
        COUNT(*)
    FROM raw.odds_markets AS market
    INNER JOIN latest_snapshot AS latest
        ON market.snapshot_id = latest.snapshot_id
    WHERE market.price > -100
      AND market.price < 100

    UNION ALL

    SELECT
        'invalid_moneyline_point',
        COUNT(*)
    FROM raw.odds_markets AS market
    INNER JOIN latest_snapshot AS latest
        ON market.snapshot_id = latest.snapshot_id
    WHERE market.market_key = 'h2h'
      AND market.point IS NOT NULL

    UNION ALL

    SELECT
        'missing_spread_or_total_point',
        COUNT(*)
    FROM raw.odds_markets AS market
    INNER JOIN latest_snapshot AS latest
        ON market.snapshot_id = latest.snapshot_id
    WHERE market.market_key IN (
        'spreads',
        'totals'
    )
      AND market.point IS NULL

    UNION ALL

    SELECT
        'invalid_outcome_count',
        COUNT(*)
    FROM (
        SELECT
            market.snapshot_id,
            market.event_id,
            market.bookmaker_key,
            market.market_key
        FROM raw.odds_markets AS market
        INNER JOIN latest_snapshot AS latest
            ON market.snapshot_id = latest.snapshot_id
        GROUP BY
            market.snapshot_id,
            market.event_id,
            market.bookmaker_key,
            market.market_key
        HAVING COUNT(*) <> 2
    )
)

SELECT
    check_name,
    issue_count,
    CASE
        WHEN issue_count = 0 THEN 'PASS'
        ELSE 'FAIL'
    END AS status
FROM quality_checks
ORDER BY check_name;