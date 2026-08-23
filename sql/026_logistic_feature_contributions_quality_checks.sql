WITH primary_games AS (
    SELECT game_id FROM analytics.current_game_predictions
    WHERE prediction_mode = 'EXTERNAL_NFELO_BLEND'
), quality_checks AS (
    SELECT 'missing_primary_game' AS check_name, COUNT(*) AS issue_count
    FROM primary_games p
    LEFT JOIN analytics.current_game_logistic_feature_contributions c USING (game_id)
    WHERE c.game_id IS NULL
    UNION ALL
    SELECT 'unexpected_fallback_contribution', COUNT(*)
    FROM analytics.current_game_logistic_feature_contributions c
    JOIN analytics.current_game_predictions p USING (game_id)
    WHERE p.prediction_mode <> 'EXTERNAL_NFELO_BLEND'
    UNION ALL
    SELECT 'duplicate_feature', COUNT(*)
    FROM (
        SELECT game_id, feature_name
        FROM analytics.current_game_logistic_feature_contributions
        GROUP BY game_id, feature_name HAVING COUNT(*) > 1
    )
    UNION ALL
    SELECT 'invalid_contribution_math', COUNT(*)
    FROM analytics.current_game_logistic_feature_contributions
    WHERE ABS(log_odds_contribution - standardized_feature_value * coefficient) > 0.000001
       OR contribution_rank <= 0
       OR prediction_generated_at IS NULL
)
SELECT check_name, issue_count,
       CASE WHEN issue_count = 0 THEN 'PASS' ELSE 'FAIL' END AS status
FROM quality_checks ORDER BY check_name;
