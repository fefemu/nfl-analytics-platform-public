WITH quality_checks AS (
    SELECT 'missing_prediction_game' AS check_name, COUNT(*) AS issue_count
    FROM analytics.current_game_predictions p
    LEFT JOIN analytics.current_game_prediction_data_science_view v USING (game_id)
    WHERE v.game_id IS NULL
    UNION ALL
    SELECT 'unknown_prediction_mode', COUNT(*)
    FROM analytics.current_game_prediction_data_science_view
    WHERE prediction_mode NOT IN ('EXTERNAL_NFELO_BLEND', 'EXTERNAL_ELO_QB_FALLBACK')
    UNION ALL
    SELECT 'invalid_probability_metadata', COUNT(*)
    FROM analytics.current_game_prediction_data_science_view
    WHERE home_win_probability NOT BETWEEN 0.0 AND 1.0
       OR away_win_probability NOT BETWEEN 0.0 AND 1.0
       OR (prediction_mode = 'EXTERNAL_NFELO_BLEND' AND (
              published_nfelo_home_probability NOT BETWEEN 0.0 AND 1.0
           OR primary_logistic_home_win_probability NOT BETWEEN 0.0 AND 1.0
           OR ABS(applied_primary_logistic_weight + applied_published_nfelo_weight - 1.0) > 0.000001))
       OR (prediction_mode = 'EXTERNAL_ELO_QB_FALLBACK'
           AND fallback_logistic_home_win_probability NOT BETWEEN 0.0 AND 1.0)
    UNION ALL
    SELECT 'invalid_feature_math', COUNT(*)
    FROM analytics.current_game_prediction_data_science_view
    WHERE (feature_name IS NOT NULL AND (
              ABS(log_odds_contribution - standardized_feature_value * coefficient) > 0.000001
           OR contribution_rank <= 0))
       OR prediction_generated_at IS NULL
)
SELECT check_name, issue_count,
       CASE WHEN issue_count = 0 THEN 'PASS' ELSE 'FAIL' END AS status
FROM quality_checks ORDER BY check_name;
