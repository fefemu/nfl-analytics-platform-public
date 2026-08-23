WITH quality_checks AS (
    SELECT 'row_count_mismatch' AS check_name,
           ABS((SELECT COUNT(*) FROM analytics.current_game_predictions)
             - (SELECT COUNT(*) FROM analytics.current_game_prediction_narratives)) AS issue_count
    UNION ALL
    SELECT 'missing_prediction_game', COUNT(*)
    FROM analytics.current_game_predictions p
    LEFT JOIN analytics.current_game_prediction_narratives n USING (game_id)
    WHERE n.game_id IS NULL
    UNION ALL
    SELECT 'invalid_required_text', COUNT(*)
    FROM analytics.current_game_prediction_narratives
    WHERE headline_en IS NULL OR headline_hu IS NULL
       OR summary_en IS NULL OR summary_hu IS NULL
       OR model_context_en IS NULL OR model_context_hu IS NULL
       OR prediction_generated_at IS NULL
    UNION ALL
    SELECT 'invalid_primary_factor', COUNT(*)
    FROM analytics.current_game_prediction_narratives n
    JOIN analytics.current_game_predictions p USING (game_id)
    WHERE p.prediction_mode = 'EXTERNAL_NFELO_BLEND'
      AND (n.top_factor_feature IS NULL OR n.top_factor_direction IS NULL)
)
SELECT check_name, issue_count,
       CASE WHEN issue_count = 0 THEN 'PASS' ELSE 'FAIL' END AS status
FROM quality_checks ORDER BY check_name;
