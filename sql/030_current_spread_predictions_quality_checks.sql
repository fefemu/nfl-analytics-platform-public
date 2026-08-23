-- =========================================================
-- NFL Analytics Platform
-- Current Production Spread Prediction Quality Checks
-- =========================================================

WITH quality_checks AS (

    SELECT
        'row_count_matches_probability_predictions'
            AS check_name,

        ABS(
            (
                SELECT COUNT(*)
                FROM analytics
                    .current_game_spread_predictions
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
        FROM analytics.current_game_spread_predictions
        GROUP BY game_id
        HAVING COUNT(*) > 1
    )

    UNION ALL

    SELECT
        'missing_probability_prediction_game',
        COUNT(*)

    FROM analytics.current_game_spread_predictions
        AS spread

    LEFT JOIN analytics.current_game_predictions
        AS probability
        ON spread.game_id = probability.game_id

    WHERE probability.game_id IS NULL

    UNION ALL

    SELECT
        'missing_spread_prediction_game',
        COUNT(*)

    FROM analytics.current_game_predictions
        AS probability

    LEFT JOIN analytics.current_game_spread_predictions
        AS spread
        ON probability.game_id = spread.game_id

    WHERE spread.game_id IS NULL

    UNION ALL

    SELECT
        'game_metadata_mismatch',
        COUNT(*)

    FROM analytics.current_game_spread_predictions
        AS spread

    INNER JOIN analytics.current_game_predictions
        AS probability
        ON spread.game_id = probability.game_id

    WHERE spread.season
            IS DISTINCT FROM probability.season
       OR spread.game_type
            IS DISTINCT FROM probability.game_type
       OR spread.week
            IS DISTINCT FROM probability.week
       OR spread.gameday
            IS DISTINCT FROM probability.gameday
       OR spread.home_team
            IS DISTINCT FROM probability.home_team
       OR spread.away_team
            IS DISTINCT FROM probability.away_team
       OR spread.is_neutral
            IS DISTINCT FROM probability.is_neutral

    UNION ALL

    SELECT
        'invalid_margin_math',
        COUNT(*)

    FROM analytics.current_game_spread_predictions

    WHERE predicted_home_margin IS NULL
       OR predicted_away_margin IS NULL
       OR NOT isfinite(predicted_home_margin)
       OR NOT isfinite(predicted_away_margin)
       OR ABS(
            predicted_home_margin
            + predicted_away_margin
          ) > 0.000001

    UNION ALL

    SELECT
        'invalid_predicted_winner',
        COUNT(*)

    FROM analytics.current_game_spread_predictions

    WHERE predicted_winner NOT IN (
            home_team,
            away_team
          )
       OR (
            predicted_home_margin >= 0.0
            AND predicted_winner <> home_team
          )
       OR (
            predicted_home_margin < 0.0
            AND predicted_winner <> away_team
          )

    UNION ALL

    SELECT
        'invalid_primary_routing',
        COUNT(*)

    FROM analytics.current_game_spread_predictions

    WHERE prediction_mode = 'EXTERNAL_NFELO_QB_RIDGE'
      AND (
            model_name <> 'external_nfelo_external_qb_spread'
         OR model_version <> '0.2.0'
         OR prediction_mode_reason
                <> 'complete_external_nfelo_qb_features'
         OR ridge_alpha <> 10.0
         OR external_nfelo_rating_difference IS NULL
         OR external_nfelo_qb_adjustment_difference IS NULL
      )

    UNION ALL

    SELECT
        'invalid_fallback_routing',
        COUNT(*)

    FROM analytics.current_game_spread_predictions

    WHERE FALSE

    UNION ALL

    SELECT
        'unknown_prediction_mode',
        COUNT(*)

    FROM analytics.current_game_spread_predictions

    WHERE prediction_mode NOT IN (
            'EXTERNAL_NFELO_QB_RIDGE'
          )

    UNION ALL

    SELECT
        'primary_training_count_mismatch',
        COUNT(*)

    FROM analytics.current_game_spread_predictions

    WHERE primary_training_game_count
            IS DISTINCT FROM (
                SELECT COUNT(*)
                FROM analytics.game_modeling_dataset AS dataset
                INNER JOIN processed.external_nfelo_game_ratings AS external
                    ON dataset.game_id = external.normalized_game_id
                WHERE dataset.season < 2026
                  AND dataset.target_point_differential
                        IS NOT NULL
                  AND external.starting_nfelo_home IS NOT NULL
                  AND external.starting_nfelo_away IS NOT NULL
                  AND external.home_538_qb_adj IS NOT NULL
                  AND external.away_538_qb_adj IS NOT NULL
            )

    UNION ALL

    SELECT
        'fallback_training_count_mismatch',
        COUNT(*)

    FROM analytics.current_game_spread_predictions

    WHERE fallback_training_game_count
            IS DISTINCT FROM (
                SELECT COUNT(*)
                FROM analytics.game_modeling_dataset AS dataset
                INNER JOIN processed.external_nfelo_game_ratings AS external
                    ON dataset.game_id = external.normalized_game_id
                WHERE dataset.season < 2026
                  AND dataset.target_point_differential
                        IS NOT NULL
                  AND external.starting_nfelo_home IS NOT NULL
                  AND external.starting_nfelo_away IS NOT NULL
                  AND external.home_538_qb_adj IS NOT NULL
                  AND external.away_538_qb_adj IS NOT NULL
            )

    UNION ALL

    SELECT
        'invalid_required_metadata',
        COUNT(*)

    FROM analytics.current_game_spread_predictions

    WHERE game_id IS NULL
       OR season IS NULL
       OR home_team IS NULL
       OR away_team IS NULL
       OR home_team = away_team
       OR external_nfelo_rating_difference IS NULL
       OR external_nfelo_qb_adjustment_difference IS NULL
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
