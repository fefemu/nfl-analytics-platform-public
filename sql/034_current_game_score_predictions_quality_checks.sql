WITH quality_checks AS (
    SELECT 'row_count_matches_spread' AS check_name,
           ABS((SELECT COUNT(*) FROM analytics.current_game_spread_predictions) - COUNT(*)) AS issue_count
    FROM analytics.current_game_score_predictions
    UNION ALL
    SELECT 'row_count_matches_totals',
           ABS((SELECT COUNT(*) FROM analytics.current_game_total_predictions) - COUNT(*))
    FROM analytics.current_game_score_predictions
    UNION ALL
    SELECT 'invalid_score_identity', COUNT(*)
    FROM analytics.current_game_score_predictions
    WHERE ABS(implied_home_score + implied_away_score - predicted_total_points) > 0.000001
       OR ABS(implied_home_score - implied_away_score - predicted_home_margin) > 0.000001
       OR implied_home_score < 0.0 OR implied_away_score < 0.0
    UNION ALL
    SELECT 'invalid_winner', COUNT(*)
    FROM analytics.current_game_score_predictions
    WHERE implied_score_winner NOT IN (home_team, away_team)
       OR (predicted_home_margin >= 0.0 AND implied_score_winner <> home_team)
       OR (predicted_home_margin < 0.0 AND implied_score_winner <> away_team)
    UNION ALL
    SELECT 'invalid_metadata', COUNT(*)
    FROM analytics.current_game_score_predictions
    WHERE spread_model_name IS NULL OR totals_model_name IS NULL
       OR spread_prediction_mode IS NULL OR totals_prediction_mode IS NULL
       OR spread_prediction_generated_at IS NULL
       OR totals_prediction_generated_at IS NULL
       OR score_prediction_generated_at IS NULL
)
SELECT check_name, issue_count,
       CASE WHEN issue_count = 0 THEN 'PASS' ELSE 'FAIL' END AS status
FROM quality_checks ORDER BY check_name;
