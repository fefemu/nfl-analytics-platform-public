-- =====================================================
-- NFL Analytics Platform
-- File: 007_odds_event_bridge_quality_checks.sql
--
-- Purpose:
--     Validate Odds API event mappings to nflverse
--     schedule game identifiers.
--
-- Author:
--     Ferenc Kaizer
--
-- Version:
--     0.1.0
-- =====================================================


WITH quality_checks AS (
    SELECT
        'row_count_matches_odds_events' AS check_name,
        ABS(
            (
                SELECT COUNT(*)
                FROM raw.odds_events
            )
            - COUNT(*)
        ) AS issue_count
    FROM analytics.odds_event_schedule_bridge

    UNION ALL

    SELECT
        'event_not_matched',
        COUNT(*)
    FROM analytics.odds_event_schedule_bridge
    WHERE match_status <> 'MATCHED'

    UNION ALL

    SELECT
        'missing_bridge_identifier',
        COUNT(*)
    FROM analytics.odds_event_schedule_bridge
    WHERE snapshot_id IS NULL
       OR odds_event_id IS NULL
       OR game_id IS NULL

    UNION ALL

    SELECT
        'duplicate_odds_event',
        COUNT(*)
    FROM (
        SELECT
            snapshot_id,
            odds_event_id
        FROM analytics.odds_event_schedule_bridge
        GROUP BY
            snapshot_id,
            odds_event_id
        HAVING COUNT(*) > 1
    )

    UNION ALL

    SELECT
        'duplicate_schedule_game',
        COUNT(*)
    FROM (
        SELECT
            snapshot_id,
            game_id
        FROM analytics.odds_event_schedule_bridge
        GROUP BY
            snapshot_id,
            game_id
        HAVING COUNT(*) > 1
    )

    UNION ALL

    SELECT
        'incorrect_game_date',
        COUNT(*)
    FROM analytics.odds_event_schedule_bridge
    WHERE eastern_game_date IS DISTINCT FROM gameday

    UNION ALL

    SELECT
        'incorrect_team_match',
        COUNT(*)
    FROM analytics.odds_event_schedule_bridge AS bridge
    INNER JOIN processed.schedule AS schedule
        ON bridge.game_id = schedule.game_id
    WHERE bridge.home_team_code
            IS DISTINCT FROM schedule.home_team
       OR bridge.away_team_code
            IS DISTINCT FROM schedule.away_team
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