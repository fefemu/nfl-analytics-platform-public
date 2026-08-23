-- =========================================================
-- NFL Analytics Platform
-- Game Modeling Dataset Quality Checks
-- =========================================================

WITH quality_checks AS (

    SELECT
        'duplicate_game_id' AS check_name,
        COUNT(*) AS issue_count

    FROM (
        SELECT game_id
        FROM analytics.game_modeling_dataset
        GROUP BY game_id
        HAVING COUNT(*) > 1
    )

    UNION ALL

    SELECT
        'row_count_matches_sources',
        ABS(
            (
                SELECT COUNT(*)
                FROM analytics.game_modeling_dataset
            )
            -
            (
                SELECT COUNT(*)

                FROM processed.schedule AS schedule

                INNER JOIN analytics.elo_game_predictions
                    AS elo
                    ON schedule.game_id = elo.game_id

                INNER JOIN analytics.rolling_team_features
                    AS home_features
                    ON schedule.game_id
                        = home_features.game_id
                   AND home_features.is_home = TRUE

                INNER JOIN analytics.rolling_team_features
                    AS away_features
                    ON schedule.game_id
                        = away_features.game_id
                   AND away_features.is_home = FALSE

                INNER JOIN analytics.game_qb_features
                    AS qb
                    ON schedule.game_id = qb.game_id

                WHERE schedule.is_completed = TRUE
            )
        )

    UNION ALL

    SELECT
        'invalid_team_assignment',
        COUNT(*)

    FROM analytics.game_modeling_dataset

    WHERE home_team IS NULL
       OR away_team IS NULL
       OR home_team = away_team

    UNION ALL

    SELECT
        'missing_elo_feature',
        COUNT(*)

    FROM analytics.game_modeling_dataset

    WHERE elo_home_advantage IS NULL
       OR home_elo_rating IS NULL
       OR away_elo_rating IS NULL
       OR elo_rating_difference IS NULL
       OR elo_home_win_probability IS NULL

    UNION ALL

    SELECT
        'invalid_elo_calculation',
        COUNT(*)

    FROM analytics.game_modeling_dataset

    WHERE elo_home_win_probability NOT BETWEEN 0.0 AND 1.0

       OR ABS(
            elo_rating_difference
            - (
                home_elo_rating
                - away_elo_rating
            )
       ) > 0.000000001

    UNION ALL

    SELECT
        'invalid_window_metadata',
        COUNT(*)

    FROM analytics.game_modeling_dataset

    WHERE home_short_window_games NOT BETWEEN 0 AND 4
       OR away_short_window_games NOT BETWEEN 0 AND 4
       OR home_long_window_games NOT BETWEEN 0 AND 8
       OR away_long_window_games NOT BETWEEN 0 AND 8

       OR home_short_window_complete
            IS DISTINCT FROM (
                home_short_window_games = 4
            )

       OR away_short_window_complete
            IS DISTINCT FROM (
                away_short_window_games = 4
            )

       OR home_long_window_complete
            IS DISTINCT FROM (
                home_long_window_games = 8
            )

       OR away_long_window_complete
            IS DISTINCT FROM (
                away_long_window_games = 8
            )

       OR both_short_windows_complete
            IS DISTINCT FROM (
                home_short_window_games = 4
                AND away_short_window_games = 4
            )

       OR both_long_windows_complete
            IS DISTINCT FROM (
                home_long_window_games = 8
                AND away_long_window_games = 8
            )

    UNION ALL

    SELECT
        'invalid_qb_availability',
        COUNT(*)

    FROM analytics.game_modeling_dataset

    WHERE home_listed_qb_rating_available
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
        'invalid_qb_rating_difference',
        COUNT(*)

    FROM analytics.game_modeling_dataset

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
            AND listed_qb_rating_difference IS NOT NULL
        )

        OR listed_qb_rating_difference_standard_error < 0

    UNION ALL

    SELECT
        'invalid_core_rolling_difference',
        COUNT(*)

    FROM analytics.game_modeling_dataset

    WHERE
        (
            home_offensive_epa_per_play_last_4 IS NOT NULL
            AND away_offensive_epa_per_play_last_4 IS NOT NULL
            AND (
                offensive_epa_per_play_difference_last_4 IS NULL
                OR ABS(
                    offensive_epa_per_play_difference_last_4
                    - (
                        home_offensive_epa_per_play_last_4
                        - away_offensive_epa_per_play_last_4
                    )
                ) > 0.000000001
            )
        )

        OR (
            home_offensive_epa_per_play_last_8 IS NOT NULL
            AND away_offensive_epa_per_play_last_8 IS NOT NULL
            AND (
                offensive_epa_per_play_difference_last_8 IS NULL
                OR ABS(
                    offensive_epa_per_play_difference_last_8
                    - (
                        home_offensive_epa_per_play_last_8
                        - away_offensive_epa_per_play_last_8
                    )
                ) > 0.000000001
            )
        )

        OR (
            home_defensive_epa_allowed_per_play_last_4
                IS NOT NULL
            AND away_defensive_epa_allowed_per_play_last_4
                IS NOT NULL
            AND (
                defensive_epa_allowed_per_play_difference_last_4
                    IS NULL
                OR ABS(
                    defensive_epa_allowed_per_play_difference_last_4
                    - (
                        home_defensive_epa_allowed_per_play_last_4
                        - away_defensive_epa_allowed_per_play_last_4
                    )
                ) > 0.000000001
            )
        )

        OR (
            home_defensive_epa_allowed_per_play_last_8
                IS NOT NULL
            AND away_defensive_epa_allowed_per_play_last_8
                IS NOT NULL
            AND (
                defensive_epa_allowed_per_play_difference_last_8
                    IS NULL
                OR ABS(
                    defensive_epa_allowed_per_play_difference_last_8
                    - (
                        home_defensive_epa_allowed_per_play_last_8
                        - away_defensive_epa_allowed_per_play_last_8
                    )
                ) > 0.000000001
            )
        )

    UNION ALL

    SELECT
        'invalid_target_calculation',
        COUNT(*)

    FROM analytics.game_modeling_dataset

    WHERE target_home_score IS NULL
       OR target_away_score IS NULL

       OR target_point_differential
            IS DISTINCT FROM (
                target_home_score
                - target_away_score
            )

       OR target_total_points
            IS DISTINCT FROM (
                target_home_score
                + target_away_score
            )

       OR target_home_result
            IS DISTINCT FROM (
                CASE
                    WHEN target_home_score > target_away_score
                        THEN 1.0
                    WHEN target_home_score < target_away_score
                        THEN 0.0
                    ELSE 0.5
                END
            )

       OR target_home_win
            IS DISTINCT FROM (
                CASE
                    WHEN target_home_score = target_away_score
                        THEN NULL
                    ELSE target_home_score > target_away_score
                END
            )

    UNION ALL

    SELECT
        'unprefixed_target_or_postgame_column',
        COUNT(*)

    FROM information_schema.columns

    WHERE table_schema = 'analytics'
      AND table_name = 'game_modeling_dataset'
      AND (
            column_name IN (
                'home_score',
                'away_score',
                'home_win',
                'point_differential',
                'total_points'
            )
            OR column_name LIKE '%actual_primary%'
            OR column_name LIKE '%postgame%'
      )

    UNION ALL

    SELECT
        'orphan_source_record',
        COUNT(*)

    FROM analytics.game_modeling_dataset AS dataset

    LEFT JOIN processed.schedule AS schedule
        ON dataset.game_id = schedule.game_id

    LEFT JOIN analytics.elo_game_predictions AS elo
        ON dataset.game_id = elo.game_id

    LEFT JOIN analytics.game_qb_features AS qb
        ON dataset.game_id = qb.game_id

    WHERE schedule.game_id IS NULL
       OR elo.game_id IS NULL
       OR qb.game_id IS NULL
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