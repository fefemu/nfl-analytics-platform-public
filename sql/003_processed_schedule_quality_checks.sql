-- =====================================================
-- NFL Analytics Platform
-- File: 003_processed_schedule_quality_checks.sql
--
-- Purpose:
--     Execute data quality checks against the
--     processed schedule table stored in DuckDB.
--
-- Author:
--     Ferenc Kaizer
--
-- Version:
--     0.1.0
-- =====================================================


WITH quality_checks AS (
    SELECT
        'row_count_matches_raw' AS check_name,
        ABS(
            (SELECT COUNT(*) FROM raw.schedule)
            - COUNT(*)
        ) AS issue_count
    FROM processed.schedule

    UNION ALL

    SELECT
        'missing_game_id',
        COUNT(*)
    FROM processed.schedule
    WHERE game_id IS NULL

    UNION ALL

    SELECT
        'duplicate_game_id',
        COUNT(*)
    FROM (
        SELECT game_id
        FROM processed.schedule
        GROUP BY game_id
        HAVING COUNT(*) > 1
    )

    UNION ALL

    SELECT
        'invalid_team_assignment',
        COUNT(*)
    FROM processed.schedule
    WHERE home_team IS NULL
       OR away_team IS NULL
       OR home_team = away_team

    UNION ALL

    SELECT
        'inconsistent_score_pair',
        COUNT(*)
    FROM processed.schedule
    WHERE (home_score IS NULL AND away_score IS NOT NULL)
       OR (home_score IS NOT NULL AND away_score IS NULL)

    UNION ALL

    SELECT
        'invalid_completed_flag',
        COUNT(*)
    FROM processed.schedule
    WHERE is_completed IS DISTINCT FROM (
        home_score IS NOT NULL
        AND away_score IS NOT NULL
    )

    UNION ALL

    SELECT
        'invalid_total_points',
        COUNT(*)
    FROM processed.schedule
    WHERE total_points IS DISTINCT FROM
        home_score + away_score

    UNION ALL

    SELECT
        'invalid_point_differential',
        COUNT(*)
    FROM processed.schedule
    WHERE point_differential IS DISTINCT FROM
        home_score - away_score

    UNION ALL

    SELECT
        'invalid_game_result_flags',
        COUNT(*)
    FROM processed.schedule
    WHERE is_completed
      AND (
          CAST(home_win AS INTEGER)
          + CAST(away_win AS INTEGER)
          + CAST(is_tie AS INTEGER)
      ) <> 1

    UNION ALL

    SELECT
        'invalid_regular_season_flag',
        COUNT(*)
    FROM processed.schedule
    WHERE is_regular_season IS DISTINCT FROM
        (game_type = 'REG')

    UNION ALL

    SELECT
        'invalid_playoff_flag',
        COUNT(*)
    FROM processed.schedule
    WHERE is_playoff IS DISTINCT FROM
        (game_type IN ('WC', 'DIV', 'CON', 'SB'))
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