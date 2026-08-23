-- =====================================================
-- NFL Analytics Platform
-- QB Ratings Quality Checks
--
-- Purpose:
--     Validate leakage-safe historical and current
--     quarterback ratings.
--
-- Author:
--     Ferenc Kaizer
--
-- Version:
--     0.1.0
-- =====================================================


WITH first_qb_ratings AS (
    SELECT
        qb_id,
        game_id,
        pregame_effective_dropbacks,
        pregame_qb_rating,
        ROW_NUMBER() OVER (
            PARTITION BY qb_id
            ORDER BY
                game_date,
                game_id
        ) AS qb_game_number
    FROM analytics.qb_rating_history
),

expected_current_metadata AS (
    SELECT
        qb_id,
        COUNT(*) AS expected_games_played,
        SUM(dropbacks) AS expected_career_dropbacks,
        MAX(game_date) AS expected_last_game_date
    FROM processed.qb_game_performance
    GROUP BY qb_id
),

checks AS (

    -- -------------------------------------------------
    -- History must preserve every QB-game source row
    -- -------------------------------------------------

    SELECT
        'history_row_count_matches_source'
            AS check_name,
        CASE
            WHEN (
                SELECT COUNT(*)
                FROM analytics.qb_rating_history
            ) = (
                SELECT COUNT(*)
                FROM processed.qb_game_performance
            )
            THEN 0
            ELSE 1
        END AS issue_count

    UNION ALL

    -- -------------------------------------------------
    -- History business key must be unique
    -- -------------------------------------------------

    SELECT
        'duplicate_history_key',
        COUNT(*)
    FROM (
        SELECT
            game_id,
            team,
            qb_id
        FROM analytics.qb_rating_history
        GROUP BY
            game_id,
            team,
            qb_id
        HAVING COUNT(*) > 1
    )

    UNION ALL

    -- -------------------------------------------------
    -- Current table must contain every distinct QB
    -- -------------------------------------------------

    SELECT
        'current_qb_count_matches_source',
        CASE
            WHEN (
                SELECT COUNT(*)
                FROM analytics.current_qb_ratings
            ) = (
                SELECT COUNT(DISTINCT qb_id)
                FROM processed.qb_game_performance
            )
            THEN 0
            ELSE 1
        END

    UNION ALL

    -- -------------------------------------------------
    -- Current QB identifier must be unique
    -- -------------------------------------------------

    SELECT
        'duplicate_current_qb',
        COUNT(*)
    FROM (
        SELECT qb_id
        FROM analytics.current_qb_ratings
        GROUP BY qb_id
        HAVING COUNT(*) > 1
    )

    UNION ALL

    -- -------------------------------------------------
    -- First appearance cannot contain own game result
    -- -------------------------------------------------

    SELECT
        'first_appearance_leakage',
        COUNT(*)
    FROM first_qb_ratings
    WHERE qb_game_number = 1
      AND (
            ABS(pregame_effective_dropbacks)
                > 0.000000001
            OR ABS(pregame_qb_rating - 100.0)
                > 0.000000001
      )

    UNION ALL

    -- -------------------------------------------------
    -- Historical values must be finite and valid
    -- -------------------------------------------------

    SELECT
        'invalid_historical_rating',
        COUNT(*)
    FROM analytics.qb_rating_history
    WHERE pregame_effective_dropbacks < 0
       OR pregame_prior_weight NOT BETWEEN 0 AND 1
       OR pregame_rating_standard_error < 0
       OR NOT isfinite(pregame_qb_rating)
       OR NOT isfinite(
            pregame_shrunk_adjusted_epa_per_dropback
          )
       OR pregame_league_epa_standard_deviation < 0.05

    UNION ALL

    -- -------------------------------------------------
    -- Shrunk estimate must lie between QB and league
    -- -------------------------------------------------

    SELECT
        'invalid_historical_shrinkage',
        COUNT(*)
    FROM analytics.qb_rating_history
    WHERE
        pregame_shrunk_adjusted_epa_per_dropback
        < LEAST(
            pregame_raw_adjusted_epa_per_dropback,
            pregame_league_mean_epa_per_dropback
          ) - 0.000000001
       OR
        pregame_shrunk_adjusted_epa_per_dropback
        > GREATEST(
            pregame_raw_adjusted_epa_per_dropback,
            pregame_league_mean_epa_per_dropback
          ) + 0.000000001

    UNION ALL

    -- -------------------------------------------------
    -- Rating index formula must be reproducible
    -- -------------------------------------------------

    SELECT
        'incorrect_historical_rating_formula',
        COUNT(*)
    FROM analytics.qb_rating_history
    WHERE ABS(
        pregame_qb_rating
        - (
            100.0
            + 15.0 * (
                pregame_shrunk_adjusted_epa_per_dropback
                - pregame_league_mean_epa_per_dropback
              )
              / pregame_league_epa_standard_deviation
          )
    ) > 0.000000001

    UNION ALL

    -- -------------------------------------------------
    -- Current metadata must match source history
    -- -------------------------------------------------

    SELECT
        'incorrect_current_metadata',
        COUNT(*)
    FROM analytics.current_qb_ratings AS current
    INNER JOIN expected_current_metadata AS expected
        ON current.qb_id = expected.qb_id
    WHERE current.games_played
          <> expected.expected_games_played
       OR current.career_dropbacks
          <> expected.expected_career_dropbacks
       OR current.last_game_date
          <> expected.expected_last_game_date
       OR current.days_since_last_game
          <> date_diff(
                'day',
                current.last_game_date,
                current.as_of_date
             )

    UNION ALL

    -- -------------------------------------------------
    -- Current rating values must be valid
    -- -------------------------------------------------

    SELECT
        'invalid_current_rating',
        COUNT(*)
    FROM analytics.current_qb_ratings
    WHERE qb_rank <= 0
       OR games_played <= 0
       OR career_dropbacks <= 0
       OR effective_dropbacks <= 0
       OR effective_dropbacks
          > career_dropbacks + 0.000000001
       OR days_since_last_game < 0
       OR prior_weight NOT BETWEEN 0 AND 1
       OR rating_standard_error < 0
       OR NOT isfinite(qb_rating)
       OR NOT isfinite(
            shrunk_adjusted_epa_per_dropback
          )
       OR league_epa_standard_deviation < 0.05

    UNION ALL

    -- -------------------------------------------------
    -- Ranking must follow descending rating order
    -- -------------------------------------------------

    SELECT
        'invalid_current_rank',
        COUNT(*)
    FROM (
        SELECT
            qb_rank,
            ROW_NUMBER() OVER (
                ORDER BY
                    qb_rating DESC,
                    qb_id
            ) AS expected_rank
        FROM analytics.current_qb_ratings
    )
    WHERE qb_rank <> expected_rank

    UNION ALL

    -- -------------------------------------------------
    -- Current snapshot date must be globally consistent
    -- -------------------------------------------------

    SELECT
        'invalid_current_as_of_date',
        COUNT(*)
    FROM analytics.current_qb_ratings
    WHERE as_of_date <> (
        SELECT MAX(game_date)
        FROM processed.qb_game_performance
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