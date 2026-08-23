-- =====================================================
-- NFL Analytics Platform
-- Rolling Team Features Quality Checks
--
-- Purpose:
--     Validate analytics.rolling_team_features
--     and detect possible target leakage.
--
-- Author:
--     Ferenc Kaizer
--
-- Version:
--     0.1.0
-- =====================================================


WITH expected_features AS (
    SELECT
        game_id,
        team,

        AVG(offensive_epa_per_play) OVER (
            PARTITION BY team, season
            ORDER BY
                game_date,
                game_id
            ROWS BETWEEN 4 PRECEDING
                     AND 1 PRECEDING
        ) AS expected_offensive_epa_last_4,

        AVG(offensive_epa_per_play) OVER (
            PARTITION BY team, season
            ORDER BY
                game_date,
                game_id
            ROWS BETWEEN 8 PRECEDING
                     AND 1 PRECEDING
        ) AS expected_offensive_epa_last_8

    FROM processed.team_game_efficiency
),

checks AS (

    -- -------------------------------------------------
    -- Business key must be unique
    -- -------------------------------------------------

    SELECT
        'duplicate_game_team' AS check_name,
        COUNT(*) AS issue_count
    FROM (
        SELECT
            game_id,
            team
        FROM analytics.rolling_team_features
        GROUP BY
            game_id,
            team
        HAVING COUNT(*) > 1
    )

    UNION ALL

    -- -------------------------------------------------
    -- Target and source row counts must match
    -- -------------------------------------------------

    SELECT
        'row_count_matches_source',
        CASE
            WHEN (
                SELECT COUNT(*)
                FROM analytics.rolling_team_features
            ) = (
                SELECT COUNT(*)
                FROM processed.team_game_efficiency
            )
            THEN 0
            ELSE 1
        END

    UNION ALL

    -- -------------------------------------------------
    -- Window sizes must match available history
    -- -------------------------------------------------

    SELECT
        'invalid_window_size',
        COUNT(*)
    FROM analytics.rolling_team_features
    WHERE season_games_played_before < 0
       OR short_window_games
          <> LEAST(season_games_played_before, 4)
       OR long_window_games
          <> LEAST(season_games_played_before, 8)

    UNION ALL

    -- -------------------------------------------------
    -- First game cannot contain current-season history
    -- -------------------------------------------------

    SELECT
        'first_game_contains_history',
        COUNT(*)
    FROM analytics.rolling_team_features
    WHERE season_games_played_before = 0
      AND (
            short_window_games <> 0
            OR long_window_games <> 0
            OR pregame_offensive_epa_per_play_last_4
               IS NOT NULL
            OR pregame_offensive_epa_per_play_last_8
               IS NOT NULL
      )

    UNION ALL

    -- -------------------------------------------------
    -- Later games must have core pregame features
    -- -------------------------------------------------

    SELECT
        'missing_available_history',
        COUNT(*)
    FROM analytics.rolling_team_features
    WHERE (
            short_window_games > 0
            AND (
                pregame_offensive_epa_per_play_last_4
                    IS NULL
                OR pregame_defensive_epa_allowed_per_play_last_4
                    IS NULL
                OR pregame_success_rate_last_4
                    IS NULL
            )
          )
       OR (
            long_window_games > 0
            AND (
                pregame_offensive_epa_per_play_last_8
                    IS NULL
                OR pregame_defensive_epa_allowed_per_play_last_8
                    IS NULL
                OR pregame_success_rate_last_8
                    IS NULL
            )
          )

    UNION ALL

    -- -------------------------------------------------
    -- Rolling rates must stay between zero and one
    -- -------------------------------------------------

    SELECT
        'invalid_rolling_rate',
        COUNT(*)
    FROM analytics.rolling_team_features
    WHERE (
            pregame_success_rate_last_4 IS NOT NULL
            AND pregame_success_rate_last_4
                NOT BETWEEN 0 AND 1
          )
       OR (
            pregame_explosive_play_rate_last_4 IS NOT NULL
            AND pregame_explosive_play_rate_last_4
                NOT BETWEEN 0 AND 1
          )
       OR (
            pregame_sack_rate_last_4 IS NOT NULL
            AND pregame_sack_rate_last_4
                NOT BETWEEN 0 AND 1
          )
       OR (
            pregame_turnover_rate_last_4 IS NOT NULL
            AND pregame_turnover_rate_last_4
                NOT BETWEEN 0 AND 1
          )
       OR (
            pregame_success_rate_last_8 IS NOT NULL
            AND pregame_success_rate_last_8
                NOT BETWEEN 0 AND 1
          )
       OR (
            pregame_explosive_play_rate_last_8 IS NOT NULL
            AND pregame_explosive_play_rate_last_8
                NOT BETWEEN 0 AND 1
          )
       OR (
            pregame_sack_rate_last_8 IS NOT NULL
            AND pregame_sack_rate_last_8
                NOT BETWEEN 0 AND 1
          )
       OR (
            pregame_turnover_rate_last_8 IS NOT NULL
            AND pregame_turnover_rate_last_8
                NOT BETWEEN 0 AND 1
          )

    UNION ALL

    -- -------------------------------------------------
    -- Recalculate short-window EPA independently
    -- -------------------------------------------------

    SELECT
        'incorrect_offensive_epa_last_4',
        COUNT(*)
    FROM analytics.rolling_team_features AS actual
    INNER JOIN expected_features AS expected
        ON actual.game_id = expected.game_id
       AND actual.team = expected.team
    WHERE actual.pregame_offensive_epa_per_play_last_4
          IS DISTINCT FROM
          expected.expected_offensive_epa_last_4

    UNION ALL

    -- -------------------------------------------------
    -- Recalculate long-window EPA independently
    -- -------------------------------------------------

    SELECT
        'incorrect_offensive_epa_last_8',
        COUNT(*)
    FROM analytics.rolling_team_features AS actual
    INNER JOIN expected_features AS expected
        ON actual.game_id = expected.game_id
       AND actual.team = expected.team
    WHERE actual.pregame_offensive_epa_per_play_last_8
          IS DISTINCT FROM
          expected.expected_offensive_epa_last_8

    UNION ALL

    -- -------------------------------------------------
    -- Postgame result fields must not leak into features
    -- -------------------------------------------------

    SELECT
        'postgame_result_column_leakage',
        COUNT(*)
    FROM information_schema.columns
    WHERE table_schema = 'analytics'
      AND table_name = 'rolling_team_features'
      AND column_name IN (
            'points_scored',
            'points_allowed',
            'point_differential',
            'team_win',
            'is_tie'
      )
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