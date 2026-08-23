WITH quality_checks AS (
    SELECT 'duplicate_offer' AS check_name, COUNT(*) AS issue_count
    FROM (
        SELECT game_id, market_line, outcome_type
        FROM analytics.current_totals_value
        GROUP BY game_id, market_line, outcome_type HAVING COUNT(*) > 1
    )
    UNION ALL
    SELECT 'missing_totals_prediction_game', COUNT(*)
    FROM analytics.current_game_total_predictions p
    LEFT JOIN analytics.current_totals_value v USING (game_id)
    WHERE v.game_id IS NULL
    UNION ALL
    SELECT 'invalid_probability_math', COUNT(*)
    FROM analytics.current_totals_value
    WHERE win_probability NOT BETWEEN 0.0 AND 1.0
       OR push_probability NOT BETWEEN 0.0 AND 1.0
       OR loss_probability NOT BETWEEN 0.0 AND 1.0
       OR ABS(win_probability + push_probability + loss_probability - 1.0) > 0.000001
       OR no_push_win_probability NOT BETWEEN 0.0 AND 1.0
    UNION ALL
    SELECT 'invalid_ev_math', COUNT(*)
    FROM analytics.current_totals_value
    WHERE ABS(expected_value_per_unit
          - (win_probability * (best_decimal_odds - 1.0) - loss_probability)) > 0.000001
       OR ABS(expected_value_percent - 100.0 * expected_value_per_unit) > 0.000001
       OR positive_expected_value <> (expected_value_per_unit > 0.0)
    UNION ALL
    SELECT 'invalid_market_metadata', COUNT(*)
    FROM analytics.current_totals_value
    WHERE market_key <> 'totals' OR market_name <> 'Totals'
       OR outcome_type NOT IN ('over', 'under')
       OR market_line IS NULL OR best_decimal_odds <= 1.0
       OR bookmaker_count <= 0 OR calibration_sample_count <= 0
       OR prediction_generated_at IS NULL
)
SELECT check_name, issue_count,
       CASE WHEN issue_count = 0 THEN 'PASS' ELSE 'FAIL' END AS status
FROM quality_checks ORDER BY check_name;
