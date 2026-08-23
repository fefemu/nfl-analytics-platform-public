WITH quality_checks AS (
    SELECT 'row_count_mismatch' AS check_name,
           ABS((SELECT COUNT(*) FROM analytics.current_game_predictions)
             - (SELECT COUNT(*) FROM analytics.current_game_prediction_explanations)) AS issue_count
    UNION ALL
    SELECT 'missing_prediction_game', COUNT(*)
    FROM analytics.current_game_predictions p
    LEFT JOIN analytics.current_game_prediction_explanations e USING (game_id)
    WHERE e.game_id IS NULL
    UNION ALL
    SELECT 'invalid_probability_math', COUNT(*)
    FROM analytics.current_game_prediction_explanations
    WHERE home_win_probability NOT BETWEEN 0.0 AND 1.0
       OR away_win_probability NOT BETWEEN 0.0 AND 1.0
       OR ABS(home_win_probability + away_win_probability - 1.0) > 0.000001
       OR favorite_win_probability <> GREATEST(home_win_probability, away_win_probability)
    UNION ALL
    SELECT 'invalid_routing', COUNT(*)
    FROM analytics.current_game_prediction_explanations
    WHERE prediction_mode NOT IN ('EXTERNAL_NFELO_BLEND', 'EXTERNAL_ELO_QB_FALLBACK')
       OR external_nfelo_rating_difference IS NULL
       OR external_nfelo_qb_adjustment_difference IS NULL
       OR NOT has_complete_fallback_features
    UNION ALL
    SELECT 'prediction_value_mismatch', COUNT(*)
    FROM analytics.current_game_predictions p
    JOIN analytics.current_game_prediction_explanations e USING (game_id)
    WHERE ABS(p.home_win_probability - e.home_win_probability) > 0.000001
       OR p.prediction_mode <> e.prediction_mode
)
SELECT check_name, issue_count,
       CASE WHEN issue_count = 0 THEN 'PASS' ELSE 'FAIL' END AS status
FROM quality_checks ORDER BY check_name;
