-- =========================================================
-- NFL Analytics Platform
-- Game Schedule Features Quality Checks
-- =========================================================

WITH quality_checks AS (

    SELECT
        'duplicate_game_id' AS check_name,
        COUNT(*) AS issue_count

    FROM (
        SELECT game_id
        FROM analytics.game_schedule_features
        GROUP BY game_id
        HAVING COUNT(*) > 1
    )

    UNION ALL

    SELECT
        'row_count_matches_schedule',
        ABS(
            (
                SELECT COUNT(*)
                FROM analytics.game_schedule_features
            )
            -
            (
                SELECT COUNT(*)
                FROM processed.schedule
            )
        )

    UNION ALL

    SELECT
        'orphan_feature_record',
        COUNT(*)

    FROM analytics.game_schedule_features AS features

    LEFT JOIN processed.schedule AS schedule
        ON features.game_id = schedule.game_id

    WHERE schedule.game_id IS NULL

    UNION ALL

    SELECT
        'missing_feature_record',
        COUNT(*)

    FROM processed.schedule AS schedule

    LEFT JOIN analytics.game_schedule_features AS features
        ON schedule.game_id = features.game_id

    WHERE features.game_id IS NULL

    UNION ALL

    SELECT
        'invalid_game_identity',
        COUNT(*)

    FROM analytics.game_schedule_features AS features

    INNER JOIN processed.schedule AS schedule
        ON features.game_id = schedule.game_id

    WHERE features.season
            IS DISTINCT FROM schedule.season
       OR features.game_type
            IS DISTINCT FROM schedule.game_type
       OR features.week
            IS DISTINCT FROM schedule.week
       OR features.game_date
            IS DISTINCT FROM schedule.gameday
       OR features.home_team
            IS DISTINCT FROM schedule.home_team
       OR features.away_team
            IS DISTINCT FROM schedule.away_team

    UNION ALL

    SELECT
        'invalid_rest_days',
        COUNT(*)

    FROM analytics.game_schedule_features AS features

    INNER JOIN processed.schedule AS schedule
        ON features.game_id = schedule.game_id

    WHERE features.home_rest_days
            IS DISTINCT FROM schedule.home_rest
       OR features.away_rest_days
            IS DISTINCT FROM schedule.away_rest
       OR features.rest_days_difference
            IS DISTINCT FROM (
                schedule.home_rest
                - schedule.away_rest
            )

    UNION ALL

    SELECT
        'invalid_short_week_flags',
        COUNT(*)

    FROM analytics.game_schedule_features

    WHERE home_short_week
            IS DISTINCT FROM (
                home_rest_days <= 6
            )
       OR away_short_week
            IS DISTINCT FROM (
                away_rest_days <= 6
            )
       OR short_week_difference
            IS DISTINCT FROM (
                CAST(home_short_week AS INTEGER)
                - CAST(away_short_week AS INTEGER)
            )

    UNION ALL

    SELECT
        'invalid_extended_rest_flags',
        COUNT(*)

    FROM analytics.game_schedule_features

    WHERE home_extended_rest
            IS DISTINCT FROM (
                home_rest_days >= 9
            )
       OR away_extended_rest
            IS DISTINCT FROM (
                away_rest_days >= 9
            )
       OR extended_rest_difference
            IS DISTINCT FROM (
                CAST(home_extended_rest AS INTEGER)
                - CAST(away_extended_rest AS INTEGER)
            )

    UNION ALL

    SELECT
        'invalid_post_bye_flags',
        COUNT(*)

    FROM analytics.game_schedule_features

    WHERE home_post_bye
            IS DISTINCT FROM (
                home_rest_days >= 13
            )
       OR away_post_bye
            IS DISTINCT FROM (
                away_rest_days >= 13
            )
       OR post_bye_difference
            IS DISTINCT FROM (
                CAST(home_post_bye AS INTEGER)
                - CAST(away_post_bye AS INTEGER)
            )

    UNION ALL

    SELECT
        'invalid_difference_range',
        COUNT(*)

    FROM analytics.game_schedule_features

    WHERE short_week_difference NOT BETWEEN -1 AND 1
       OR extended_rest_difference NOT BETWEEN -1 AND 1
       OR post_bye_difference NOT BETWEEN -1 AND 1

    UNION ALL

    SELECT
        'invalid_rest_flag_hierarchy',
        COUNT(*)

    FROM analytics.game_schedule_features

    WHERE (
            home_post_bye
            AND NOT home_extended_rest
          )
       OR (
            away_post_bye
            AND NOT away_extended_rest
          )
       OR (
            home_short_week
            AND home_extended_rest
          )
       OR (
            away_short_week
            AND away_extended_rest
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