WITH quality_checks AS (
    SELECT 'duplicate_game_id' AS check_name,
           COUNT(*) AS issue_count
    FROM (
        SELECT game_id FROM analytics.current_game_predictions
        GROUP BY game_id HAVING COUNT(*) > 1
    )

    UNION ALL
    SELECT 'invalid_probability_math', COUNT(*)
    FROM analytics.current_game_predictions
    WHERE home_win_probability NOT BETWEEN 0.0 AND 1.0
       OR away_win_probability NOT BETWEEN 0.0 AND 1.0
       OR ABS(home_win_probability + away_win_probability - 1.0) > 0.000001

    UNION ALL
    SELECT 'invalid_routing', COUNT(*)
    FROM analytics.current_game_predictions
    WHERE prediction_mode NOT IN ('EXTERNAL_NFELO_BLEND', 'EXTERNAL_ELO_QB_FALLBACK')
       OR (prediction_mode = 'EXTERNAL_NFELO_BLEND' AND (
              prediction_mode_reason <> 'complete_external_primary_features'
           OR primary_logistic_home_win_probability IS NULL
           OR published_nfelo_home_probability IS NULL
           OR ABS(applied_primary_logistic_weight + applied_published_nfelo_weight - 1.0) > 0.000001
           OR NOT has_complete_production_features))
       OR (prediction_mode = 'EXTERNAL_ELO_QB_FALLBACK' AND (
              prediction_mode_reason <> 'incomplete_external_primary_features'
           OR fallback_logistic_home_win_probability IS NULL
           OR primary_logistic_home_win_probability IS NOT NULL
           OR has_complete_production_features))

    UNION ALL
    SELECT 'invalid_external_features', COUNT(*)
    FROM analytics.current_game_predictions
    WHERE external_nfelo_rating_difference IS NULL
       OR external_nfelo_qb_adjustment_difference IS NULL
       OR NOT has_complete_fallback_features

    UNION ALL
    SELECT 'invalid_metadata', COUNT(*)
    FROM analytics.current_game_predictions
    WHERE model_name <> 'external_nfelo_probability_routing'
       OR model_version <> '0.3.0'
       OR predicted_winner NOT IN (home_team, away_team)
       OR prediction_generated_at IS NULL
)
SELECT check_name, issue_count,
       CASE WHEN issue_count = 0 THEN 'PASS' ELSE 'FAIL' END AS status
FROM quality_checks ORDER BY check_name;
