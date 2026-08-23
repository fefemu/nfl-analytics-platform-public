-- =====================================================
-- NFL Analytics Platform
-- File: 008_current_market_board_quality_checks.sql
--
-- Purpose:
--     Validate the latest schedule-linked NFL
--     betting market board.
--
-- Author:
--     Ferenc Kaizer
--
-- Version:
--     0.1.0
-- =====================================================


WITH quality_checks AS (
    SELECT
        'contains_exactly_one_snapshot' AS check_name,
        CASE
            WHEN COUNT(DISTINCT snapshot_id) = 1
                THEN 0
            ELSE 1
        END AS issue_count
    FROM analytics.current_market_board

    UNION ALL

    SELECT
        'not_latest_snapshot',
        COUNT(*)
    FROM analytics.current_market_board
    WHERE snapshot_id IS DISTINCT FROM (
        SELECT snapshot_id
        FROM analytics.best_odds_by_line
        ORDER BY fetched_at DESC, snapshot_id DESC
        LIMIT 1
    )

    UNION ALL

    SELECT
        'missing_identifier',
        COUNT(*)
    FROM analytics.current_market_board
    WHERE snapshot_id IS NULL
       OR game_id IS NULL
       OR odds_event_id IS NULL

    UNION ALL

    SELECT
        'duplicate_equivalent_offer',
        COUNT(*)
    FROM (
        SELECT
            snapshot_id,
            game_id,
            market_key,
            outcome_type,
            point
        FROM analytics.current_market_board
        GROUP BY
            snapshot_id,
            game_id,
            market_key,
            outcome_type,
            point
        HAVING COUNT(*) > 1
    )

    UNION ALL

    SELECT
        'invalid_market_name',
        COUNT(*)
    FROM analytics.current_market_board
    WHERE market_name NOT IN (
        'Moneyline',
        'Spread',
        'Totals'
    )

    UNION ALL

    SELECT
        'invalid_best_price',
        COUNT(*)
    FROM analytics.current_market_board
    WHERE best_bookmaker_key IS NULL
       OR best_decimal_odds <= 1.0
       OR best_american_price IS NULL

    UNION ALL

    SELECT
        'invalid_consensus_probability',
        COUNT(*)
    FROM analytics.current_market_board
    WHERE consensus_no_vig_probability <= 0.0
       OR consensus_no_vig_probability >= 1.0

    UNION ALL

    SELECT
        'missing_schedule_context',
        COUNT(*)
    FROM analytics.current_market_board
    WHERE season IS NULL
       OR game_type IS NULL
       OR week IS NULL
       OR gameday IS NULL
       OR home_team IS NULL
       OR away_team IS NULL
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