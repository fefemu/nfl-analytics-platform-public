-- =========================================================
-- NFL Analytics Platform
-- Game QB Features Quality Checks
-- =========================================================

WITH quality_checks AS (

    -- -----------------------------------------------------
    -- Pregame feature table structure
    -- -----------------------------------------------------

    SELECT
        'duplicate_feature_game' AS check_name,
        COUNT(*) AS issue_count

    FROM (
        SELECT game_id
        FROM analytics.game_qb_features
        GROUP BY game_id
        HAVING COUNT(*) > 1
    )

    UNION ALL

    SELECT
        'feature_row_count_mismatch',
        ABS(
            (
                SELECT COUNT(*)
                FROM analytics.game_qb_features
            )
            -
            (
                SELECT COUNT(*)
                FROM processed.schedule AS schedule
                WHERE schedule.is_completed = TRUE
                  AND EXISTS (
                        SELECT 1
                        FROM processed.qb_game_performance
                            AS performance
                        WHERE performance.game_id
                            = schedule.game_id
                  )
            )
        )

    UNION ALL

    SELECT
        'invalid_feature_team_assignment',
        COUNT(*)

    FROM analytics.game_qb_features

    WHERE home_team IS NULL
       OR away_team IS NULL
       OR home_team = away_team

    UNION ALL

    SELECT
        'invalid_rating_availability_flag',
        COUNT(*)

    FROM analytics.game_qb_features

    WHERE
        home_listed_qb_rating_available
            IS DISTINCT FROM (
                home_listed_qb_rating IS NOT NULL
            )

        OR away_listed_qb_rating_available
            IS DISTINCT FROM (
                away_listed_qb_rating IS NOT NULL
            )

        OR both_listed_qb_ratings_available
            IS DISTINCT FROM (
                home_listed_qb_rating IS NOT NULL
                AND away_listed_qb_rating IS NOT NULL
            )

    UNION ALL

    SELECT
        'invalid_listed_qb_rating_difference',
        COUNT(*)

    FROM analytics.game_qb_features

    WHERE
        (
            both_listed_qb_ratings_available = TRUE
            AND (
                listed_qb_rating_difference IS NULL
                OR ABS(
                    listed_qb_rating_difference
                    - (
                        home_listed_qb_rating
                        - away_listed_qb_rating
                    )
                ) > 0.000000001
            )
        )

        OR (
            both_listed_qb_ratings_available = FALSE
            AND (
                listed_qb_rating_difference IS NOT NULL
                OR listed_qb_rating_difference_standard_error
                    IS NOT NULL
            )
        )

        OR listed_qb_rating_difference_standard_error < 0

    UNION ALL

    SELECT
        'postgame_column_in_feature_table',
        COUNT(*)

    FROM information_schema.columns

    WHERE table_schema = 'analytics'
      AND table_name = 'game_qb_features'
      AND (
            column_name LIKE '%actual_primary%'
            OR column_name LIKE '%postgame%'
            OR column_name LIKE '%final_score%'
      )

    UNION ALL

    SELECT
        'orphan_feature_game',
        COUNT(*)

    FROM analytics.game_qb_features AS features

    LEFT JOIN processed.schedule AS schedule
        ON features.game_id = schedule.game_id

    WHERE schedule.game_id IS NULL


    -- -----------------------------------------------------
    -- Postgame audit table structure
    -- -----------------------------------------------------

    UNION ALL

    SELECT
        'duplicate_audit_game',
        COUNT(*)

    FROM (
        SELECT game_id
        FROM analytics.game_qb_audit
        GROUP BY game_id
        HAVING COUNT(*) > 1
    )

    UNION ALL

    SELECT
        'audit_row_count_mismatch',
        ABS(
            (
                SELECT COUNT(*)
                FROM analytics.game_qb_audit
            )
            -
            (
                SELECT COUNT(*)
                FROM analytics.game_qb_features
            )
        )

    UNION ALL

    SELECT
        'missing_actual_primary_qb',
        COUNT(*)

    FROM analytics.game_qb_audit

    WHERE home_actual_primary_qb_id IS NULL
       OR away_actual_primary_qb_id IS NULL

    UNION ALL

    SELECT
        'invalid_listed_actual_match_flag',
        COUNT(*)

    FROM analytics.game_qb_audit

    WHERE
        home_listed_qb_matches_actual_primary
            IS DISTINCT FROM (
                home_listed_qb_id
                = home_actual_primary_qb_id
            )

        OR away_listed_qb_matches_actual_primary
            IS DISTINCT FROM (
                away_listed_qb_id
                = away_actual_primary_qb_id
            )

        OR both_listed_qbs_match_actual_primary
            IS DISTINCT FROM (
                home_listed_qb_id
                    = home_actual_primary_qb_id
                AND away_listed_qb_id
                    = away_actual_primary_qb_id
            )

    UNION ALL

    SELECT
        'invalid_actual_primary_rating_difference',
        COUNT(*)

    FROM analytics.game_qb_audit

    WHERE
        (
            home_actual_primary_qb_pregame_rating IS NOT NULL
            AND away_actual_primary_qb_pregame_rating IS NOT NULL
            AND (
                actual_primary_qb_rating_difference IS NULL
                OR ABS(
                    actual_primary_qb_rating_difference
                    - (
                        home_actual_primary_qb_pregame_rating
                        - away_actual_primary_qb_pregame_rating
                    )
                ) > 0.000000001
            )
        )

        OR (
            (
                home_actual_primary_qb_pregame_rating IS NULL
                OR away_actual_primary_qb_pregame_rating IS NULL
            )
            AND actual_primary_qb_rating_difference IS NOT NULL
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