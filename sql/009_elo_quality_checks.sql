-- =====================================================
-- NFL Analytics Platform
-- File: 009_elo_quality_checks.sql
--
-- Purpose:
--     Validate historical Elo game predictions
--     and current franchise ratings.
--
-- Author:
--     Ferenc Kaizer
--
-- Version:
--     0.1.0
-- =====================================================


WITH checks AS (
    SELECT
        'missing_game_id' AS check_name,
        COUNT(*) AS issue_count
    FROM analytics.elo_game_predictions
    WHERE game_id IS NULL

    UNION ALL

    SELECT
        'duplicate_game_id',
        COUNT(*)
    FROM (
        SELECT game_id
        FROM analytics.elo_game_predictions
        GROUP BY game_id
        HAVING COUNT(*) > 1
    )

    UNION ALL

    SELECT
        'invalid_probability',
        COUNT(*)
    FROM analytics.elo_game_predictions
    WHERE home_win_probability NOT BETWEEN 0.0 AND 1.0
       OR away_win_probability NOT BETWEEN 0.0 AND 1.0
       OR ABS(
            home_win_probability
            + away_win_probability
            - 1.0
       ) > 0.000000001

    UNION ALL

    SELECT
        'invalid_actual_home_score',
        COUNT(*)
    FROM analytics.elo_game_predictions
    WHERE actual_home_score NOT IN (0.0, 0.5, 1.0)

    UNION ALL

    SELECT
        'invalid_home_rating_update',
        COUNT(*)
    FROM analytics.elo_game_predictions
    WHERE ABS(
        home_rating_post
        - home_rating_pre
        - home_rating_change
    ) > 0.000000001

    UNION ALL

    SELECT
        'invalid_away_rating_update',
        COUNT(*)
    FROM analytics.elo_game_predictions
    WHERE ABS(
        away_rating_post
        - away_rating_pre
        + home_rating_change
    ) > 0.000000001

    UNION ALL

    SELECT
        'neutral_game_with_home_advantage',
        COUNT(*)
    FROM analytics.elo_game_predictions
    WHERE is_neutral = TRUE
      AND home_advantage <> 0.0

    UNION ALL

    SELECT
        'history_row_count_matches_source',
        CASE
            WHEN (
                SELECT COUNT(*)
                FROM analytics.elo_game_predictions
            ) = (
                SELECT COUNT(*)
                FROM processed.schedule
                WHERE is_completed = TRUE
                  AND game_type IN (
                      'REG',
                      'WC',
                      'DIV',
                      'CON',
                      'SB'
                  )
            )
            THEN 0
            ELSE 1
        END

    UNION ALL

    SELECT
        'duplicate_current_team',
        COUNT(*)
    FROM (
        SELECT team
        FROM analytics.current_elo_ratings
        GROUP BY team
        HAVING COUNT(*) > 1
    )

    UNION ALL

    SELECT
        'invalid_current_rating',
        COUNT(*)
    FROM analytics.current_elo_ratings
    WHERE elo_rating IS NULL
       OR games_played < 1
       OR elo_rank < 1

    UNION ALL

    SELECT
        'current_team_count_is_32',
        CASE
            WHEN (
                SELECT COUNT(*)
                FROM analytics.current_elo_ratings
            ) = 32
            THEN 0
            ELSE 1
        END
)

SELECT
    check_name,
    issue_count,
    CASE
        WHEN issue_count = 0 THEN 'PASS'
        ELSE 'FAIL'
    END AS status
FROM checks
ORDER BY check_name;