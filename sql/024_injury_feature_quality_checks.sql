-- =========================================================
-- NFL Analytics Platform
-- Injury Context, Impact and Modeling Feature Quality Checks
-- =========================================================

WITH quality_checks AS (

    SELECT
        'injury_context_row_count_is_45318'
            AS check_name,

        CASE
            WHEN (
                SELECT COUNT(*)
                FROM analytics.player_game_injury_context
            ) = 45318
            THEN 0
            ELSE 1
        END AS issue_count

    UNION ALL

    SELECT
        'injury_context_duplicate_business_keys',
        COUNT(*)

    FROM (
        SELECT
            game_id,
            team,
            gsis_id
        FROM analytics.player_game_injury_context
        GROUP BY
            game_id,
            team,
            gsis_id
        HAVING COUNT(*) > 1
    )

    UNION ALL

    SELECT
        'injury_context_null_business_keys',
        COUNT(*)

    FROM analytics.player_game_injury_context

    WHERE game_id IS NULL
       OR season IS NULL
       OR gameday IS NULL
       OR team IS NULL
       OR opponent IS NULL
       OR gsis_id IS NULL
       OR is_home IS NULL

    UNION ALL

    SELECT
        'injury_context_prior_history_leakage',
        COUNT(*)

    FROM analytics.player_game_injury_context

    WHERE has_prior_snap_history
      AND prior_snap_history_gameday >= gameday

    UNION ALL

    SELECT
        'injury_context_invalid_history_source',
        COUNT(*)

    FROM analytics.player_game_injury_context

    WHERE snap_history_source
            NOT IN (
                'TEAM',
                'CAREER',
                'NONE'
            )
       OR (
            has_prior_snap_history
            AND snap_history_source = 'NONE'
          )
       OR (
            NOT has_prior_snap_history
            AND snap_history_source != 'NONE'
          )

    UNION ALL

    SELECT
        'injury_context_invalid_prior_snap_shares',
        COUNT(*)

    FROM analytics.player_game_injury_context

    WHERE prior_offense_snap_share_last_4
            NOT BETWEEN 0.0 AND 1.0
       OR prior_defense_snap_share_last_4
            NOT BETWEEN 0.0 AND 1.0
       OR prior_special_teams_snap_share_last_4
            NOT BETWEEN 0.0 AND 1.0
       OR prior_offense_snap_share_last_8
            NOT BETWEEN 0.0 AND 1.0
       OR prior_defense_snap_share_last_8
            NOT BETWEEN 0.0 AND 1.0
       OR prior_special_teams_snap_share_last_8
            NOT BETWEEN 0.0 AND 1.0

    UNION ALL

    SELECT
        'player_injury_impact_row_count_matches_context',

        CASE
            WHEN (
                SELECT COUNT(*)
                FROM analytics.player_injury_impact
            ) = (
                SELECT COUNT(*)
                FROM analytics.player_game_injury_context
            )
            THEN 0
            ELSE 1
        END

    UNION ALL

    SELECT
        'player_injury_impact_duplicate_business_keys',
        COUNT(*)

    FROM (
        SELECT
            game_id,
            team,
            gsis_id
        FROM analytics.player_injury_impact
        GROUP BY
            game_id,
            team,
            gsis_id
        HAVING COUNT(*) > 1
    )

    UNION ALL

    SELECT
        'player_injury_impact_invalid_scores',
        COUNT(*)

    FROM analytics.player_injury_impact

    WHERE availability_severity_score
            NOT BETWEEN 0.0 AND 1.0
       OR player_importance_score
            NOT BETWEEN 0.0 AND 1.0
       OR injury_impact_score
            NOT BETWEEN 0.0 AND 1.0
       OR non_qb_injury_impact_score
            NOT BETWEEN 0.0 AND 1.0
       OR offense_injury_impact_score
            NOT BETWEEN 0.0 AND 1.0
       OR defense_injury_impact_score
            NOT BETWEEN 0.0 AND 1.0
       OR special_teams_injury_impact_score
            NOT BETWEEN 0.0 AND 1.0

    UNION ALL

    SELECT
        'player_injury_impact_qb_double_counting',
        COUNT(*)

    FROM analytics.player_injury_impact

    WHERE is_qb
      AND (
            non_qb_injury_impact_score != 0.0
            OR offense_injury_impact_score != 0.0
          )

    UNION ALL

    SELECT
        'player_injury_positive_impact_count_is_21134',

        CASE
            WHEN (
                SELECT COUNT(*)
                FROM analytics.player_injury_impact
                WHERE injury_impact_score > 0.0
            ) = 21134
            THEN 0
            ELSE 1
        END

    UNION ALL

    SELECT
        'team_injury_burden_row_count_is_4454',

        CASE
            WHEN (
                SELECT COUNT(*)
                FROM analytics.team_game_injury_burden
            ) = 4454
            THEN 0
            ELSE 1
        END

    UNION ALL

    SELECT
        'team_injury_burden_duplicate_business_keys',
        COUNT(*)

    FROM (
        SELECT
            game_id,
            team
        FROM analytics.team_game_injury_burden
        GROUP BY
            game_id,
            team
        HAVING COUNT(*) > 1
    )

    UNION ALL

    SELECT
        'team_injury_burden_missing_source_rows_is_21',

        CASE
            WHEN (
                SELECT COUNT(*)
                FROM analytics.team_game_injury_burden
                WHERE NOT has_injury_report_data
            ) = 21
            THEN 0
            ELSE 1
        END

    UNION ALL

    SELECT
        'team_injury_burden_invalid_missing_data_semantics',
        COUNT(*)

    FROM analytics.team_game_injury_burden

    WHERE NOT has_injury_report_data
      AND (
            injury_report_player_count != 0
            OR out_player_count != 0
            OR doubtful_player_count != 0
            OR questionable_player_count != 0
            OR total_injury_burden != 0.0
            OR qb_injury_burden != 0.0
            OR non_qb_injury_burden != 0.0
            OR offense_injury_burden != 0.0
            OR defense_injury_burden != 0.0
            OR special_teams_injury_burden != 0.0
            OR depth_chart_match_rate IS NOT NULL
            OR snap_history_match_rate IS NOT NULL
          )

    UNION ALL

    SELECT
        'team_injury_burden_invalid_scores',
        COUNT(*)

    FROM analytics.team_game_injury_burden

    WHERE total_injury_burden < 0.0
       OR qb_injury_burden < 0.0
       OR non_qb_injury_burden < 0.0
       OR offense_injury_burden < 0.0
       OR defense_injury_burden < 0.0
       OR special_teams_injury_burden < 0.0
       OR depth_chart_match_rate
            NOT BETWEEN 0.0 AND 1.0
       OR snap_history_match_rate
            NOT BETWEEN 0.0 AND 1.0

    UNION ALL

    SELECT
        'team_injury_burden_schedule_coverage',
        COUNT(*)

    FROM (
        SELECT
            schedule.game_id,
            schedule.home_team AS team
        FROM processed.schedule AS schedule
        WHERE schedule.season
                BETWEEN 2018 AND 2025

        UNION ALL

        SELECT
            schedule.game_id,
            schedule.away_team AS team
        FROM processed.schedule AS schedule
        WHERE schedule.season
                BETWEEN 2018 AND 2025
    ) AS schedule_team_games

    LEFT JOIN analytics.team_game_injury_burden
        AS burden
        ON schedule_team_games.game_id
            = burden.game_id
       AND schedule_team_games.team
            = burden.team

    WHERE burden.game_id IS NULL

    UNION ALL

    SELECT
        'game_injury_feature_row_count_is_2227',

        CASE
            WHEN (
                SELECT COUNT(*)
                FROM analytics.game_injury_features
            ) = 2227
            THEN 0
            ELSE 1
        END

    UNION ALL

    SELECT
        'game_injury_feature_duplicate_games',
        COUNT(*)

    FROM (
        SELECT game_id
        FROM analytics.game_injury_features
        GROUP BY game_id
        HAVING COUNT(*) > 1
    )

    UNION ALL

    SELECT
        'game_injury_incomplete_coverage_count_is_14',

        CASE
            WHEN (
                SELECT COUNT(*)
                FROM analytics.game_injury_features
                WHERE NOT has_complete_injury_data
            ) = 14
            THEN 0
            ELSE 1
        END

    UNION ALL

    SELECT
        'game_injury_invalid_coverage_flags',
        COUNT(*)

    FROM analytics.game_injury_features

    WHERE has_complete_injury_data
            IS DISTINCT FROM (
                home_has_injury_report_data
                AND away_has_injury_report_data
            )

    UNION ALL

    SELECT
        'game_injury_invalid_burden_differences',
        COUNT(*)

    FROM analytics.game_injury_features

    WHERE ABS(
            total_injury_burden_difference
            - (
                home_total_injury_burden
                - away_total_injury_burden
            )
          ) > 0.000000001
       OR ABS(
            qb_injury_burden_difference
            - (
                home_qb_injury_burden
                - away_qb_injury_burden
            )
          ) > 0.000000001
       OR ABS(
            non_qb_injury_burden_difference
            - (
                home_non_qb_injury_burden
                - away_non_qb_injury_burden
            )
          ) > 0.000000001
       OR ABS(
            offense_injury_burden_difference
            - (
                home_offense_injury_burden
                - away_offense_injury_burden
            )
          ) > 0.000000001
       OR ABS(
            defense_injury_burden_difference
            - (
                home_defense_injury_burden
                - away_defense_injury_burden
            )
          ) > 0.000000001
       OR ABS(
            special_teams_injury_burden_difference
            - (
                home_special_teams_injury_burden
                - away_special_teams_injury_burden
            )
          ) > 0.000000001

    UNION ALL

    SELECT
        'modeling_dataset_injury_coverage_matches_source',
        COUNT(*)

    FROM analytics.game_modeling_dataset
        AS modeling

    INNER JOIN analytics.game_injury_features
        AS injury
        ON modeling.game_id = injury.game_id

    WHERE modeling.has_complete_injury_data
            IS DISTINCT FROM
                injury.has_complete_injury_data
       OR ABS(
            modeling.non_qb_injury_burden_difference
            - injury.non_qb_injury_burden_difference
          ) > 0.000000001
       OR ABS(
            modeling.offense_injury_burden_difference
            - injury.offense_injury_burden_difference
          ) > 0.000000001
       OR ABS(
            modeling.defense_injury_burden_difference
            - injury.defense_injury_burden_difference
          ) > 0.000000001
       OR ABS(
            modeling.special_teams_injury_burden_difference
            - injury.special_teams_injury_burden_difference
          ) > 0.000000001
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