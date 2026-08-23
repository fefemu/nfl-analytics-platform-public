-- =========================================================
-- NFL Analytics Platform
-- Current Production Totals Prediction Quality Checks
-- =========================================================

WITH quality_checks AS (

    SELECT
        'row_count_matches_probability_predictions'
            AS check_name,

        ABS(
            (
                SELECT COUNT(*)
                FROM analytics
                    .current_game_total_predictions
            )
            -
            (
                SELECT COUNT(*)
                FROM analytics
                    .current_game_predictions
            )
        ) AS issue_count

    UNION ALL

    SELECT
        'duplicate_game_id',
        COUNT(*)

    FROM (
        SELECT game_id
        FROM analytics.current_game_total_predictions
        GROUP BY game_id
        HAVING COUNT(*) > 1
    )

    UNION ALL

    SELECT
        'missing_probability_prediction_game',
        COUNT(*)

    FROM analytics.current_game_total_predictions
        AS totals

    LEFT JOIN analytics.current_game_predictions
        AS probability
        ON totals.game_id = probability.game_id

    WHERE probability.game_id IS NULL

    UNION ALL

    SELECT
        'missing_totals_prediction_game',
        COUNT(*)

    FROM analytics.current_game_predictions
        AS probability

    LEFT JOIN analytics.current_game_total_predictions
        AS totals
        ON probability.game_id = totals.game_id

    WHERE totals.game_id IS NULL

    UNION ALL

    SELECT
        'game_metadata_mismatch',
        COUNT(*)

    FROM analytics.current_game_total_predictions
        AS totals

    INNER JOIN analytics.current_game_predictions
        AS probability
        ON totals.game_id = probability.game_id

    WHERE totals.season
            IS DISTINCT FROM probability.season
       OR totals.game_type
            IS DISTINCT FROM probability.game_type
       OR totals.week
            IS DISTINCT FROM probability.week
       OR totals.gameday
            IS DISTINCT FROM probability.gameday
       OR totals.home_team
            IS DISTINCT FROM probability.home_team
       OR totals.away_team
            IS DISTINCT FROM probability.away_team
       OR totals.is_neutral
            IS DISTINCT FROM probability.is_neutral

    UNION ALL

    SELECT
        'invalid_predicted_total',
        COUNT(*)

    FROM analytics.current_game_total_predictions

    WHERE predicted_total_points IS NULL
       OR NOT isfinite(predicted_total_points)

    UNION ALL

    SELECT
        'invalid_primary_routing',
        COUNT(*)

    FROM analytics.current_game_total_predictions

    WHERE prediction_mode = 'RIDGE_TOTALS_PRIMARY'
      AND (
            model_name
                <> 'ridge_epa_weather_qb_league_64_totals'
         OR model_version <> '0.1.0'
         OR prediction_mode_reason
                <> 'complete_locked_totals_features'
         OR ridge_alpha <> 100.0
         OR NOT has_complete_primary_features
         OR NOT both_short_windows_complete
         OR NOT both_listed_qb_ratings_available
         OR offensive_epa_sum_last_4 IS NULL
         OR defensive_epa_allowed_sum_last_4
                IS NULL
         OR listed_qb_rating_sum IS NULL
      )

    UNION ALL

    SELECT
        'invalid_fallback_routing',
        COUNT(*)

    FROM analytics.current_game_total_predictions

    WHERE prediction_mode = 'RIDGE_TOTALS_FALLBACK'
      AND (
            model_name
                <> 'ridge_league_64_indoor_elo_totals'
         OR model_version <> '0.1.0'
         OR prediction_mode_reason
                <> 'missing_primary_rolling_or_qb_features'
         OR ridge_alpha <> 1.0
         OR has_complete_primary_features
      )

    UNION ALL

    SELECT
        'unknown_prediction_mode',
        COUNT(*)

    FROM analytics.current_game_total_predictions

    WHERE prediction_mode NOT IN (
            'RIDGE_TOTALS_PRIMARY',
            'RIDGE_TOTALS_FALLBACK'
          )

    UNION ALL

    SELECT
        'invalid_elo_rating_sum',
        COUNT(*)

    FROM analytics.current_game_total_predictions
        AS totals

    LEFT JOIN analytics.current_elo_ratings
        AS home_elo
        ON totals.home_team = home_elo.team

    LEFT JOIN analytics.current_elo_ratings
        AS away_elo
        ON totals.away_team = away_elo.team

    WHERE totals.elo_rating_sum IS NULL
       OR home_elo.elo_rating IS NULL
       OR away_elo.elo_rating IS NULL
       OR ABS(
            totals.elo_rating_sum
            - (
                home_elo.elo_rating
                + away_elo.elo_rating
              )
          ) > 0.000001

    UNION ALL

    SELECT
        'primary_training_count_mismatch',
        COUNT(*)

    FROM analytics.current_game_total_predictions

    WHERE primary_training_game_count
            IS DISTINCT FROM (
                SELECT COUNT(*)
                FROM analytics.game_modeling_dataset
                WHERE season < 2026
                  AND target_total_points IS NOT NULL
                  AND both_short_windows_complete
                  AND home_offensive_epa_per_play_last_4
                        IS NOT NULL
                  AND away_offensive_epa_per_play_last_4
                        IS NOT NULL
                  AND home_defensive_epa_allowed_per_play_last_4
                        IS NOT NULL
                  AND away_defensive_epa_allowed_per_play_last_4
                        IS NOT NULL
                  AND is_indoor IS NOT NULL
                  AND has_game_weather IS NOT NULL
                  AND cold_degrees_below_50 IS NOT NULL
                  AND heat_degrees_above_80 IS NOT NULL
                  AND wind_mph_above_10 IS NOT NULL
                  AND home_listed_qb_rating IS NOT NULL
                  AND away_listed_qb_rating IS NOT NULL
                  AND league_average_total_last_64
                        IS NOT NULL
            )

    UNION ALL

    SELECT
        'fallback_training_count_mismatch',
        COUNT(*)

    FROM analytics.current_game_total_predictions

    WHERE fallback_training_game_count
            IS DISTINCT FROM (
                SELECT COUNT(*)
                FROM analytics.game_modeling_dataset
                WHERE season < 2026
                  AND target_total_points IS NOT NULL
                  AND home_elo_rating IS NOT NULL
                  AND away_elo_rating IS NOT NULL
                  AND is_indoor IS NOT NULL
                  AND league_average_total_last_64
                        IS NOT NULL
            )

    UNION ALL

    SELECT
        'invalid_required_metadata',
        COUNT(*)

    FROM analytics.current_game_total_predictions

    WHERE game_id IS NULL
       OR season IS NULL
       OR game_type IS NULL
       OR week IS NULL
       OR gameday IS NULL
       OR home_team IS NULL
       OR away_team IS NULL
       OR home_team = away_team
       OR is_neutral IS NULL
       OR elo_rating_sum IS NULL
       OR is_indoor IS NULL
       OR has_game_weather IS NULL
       OR league_average_total_last_64 IS NULL
       OR primary_training_game_count <= 0
       OR fallback_training_game_count <= 0
       OR prediction_generated_at IS NULL
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