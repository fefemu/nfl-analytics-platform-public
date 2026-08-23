WITH source_count AS (
    SELECT
        (SELECT COUNT(*) FROM analytics.current_moneyline_value)
      + (SELECT COUNT(*) FROM analytics.current_spread_value)
      + (SELECT COUNT(*) FROM analytics.current_totals_value) AS expected_rows
), quality_checks AS (
    SELECT 'source_row_count_mismatch' AS check_name,
           ABS((SELECT expected_rows FROM source_count) - COUNT(*)) AS issue_count
    FROM analytics.current_betting_board
    UNION ALL
    SELECT 'missing_market', 3 - COUNT(DISTINCT market_key)
    FROM analytics.current_betting_board
    WHERE market_key IN ('h2h', 'spreads', 'totals')
    UNION ALL
    SELECT 'invalid_probability', COUNT(*)
    FROM analytics.current_betting_board
    WHERE model_probability NOT BETWEEN 0.0 AND 1.0
       OR push_probability NOT BETWEEN 0.0 AND 1.0
       OR loss_probability NOT BETWEEN 0.0 AND 1.0
       OR ABS(model_probability + push_probability + loss_probability - 1.0) > 0.000001
    UNION ALL
    SELECT 'invalid_ev_flag', COUNT(*)
    FROM analytics.current_betting_board
    WHERE positive_expected_value <> (expected_value_per_unit > 0.0)
       OR ABS(expected_value_percent - 100.0 * expected_value_per_unit) > 0.000001
    UNION ALL
    SELECT 'invalid_metadata', COUNT(*)
    FROM analytics.current_betting_board
    WHERE game_id IS NULL OR home_team IS NULL OR away_team IS NULL
       OR home_team = away_team OR best_bookmaker_key IS NULL
       OR best_decimal_odds <= 1.0 OR model_name IS NULL
       OR prediction_mode IS NULL OR prediction_generated_at IS NULL
       OR betting_board_generated_at IS NULL
)
SELECT check_name, issue_count,
       CASE WHEN issue_count = 0 THEN 'PASS' ELSE 'FAIL' END AS status
FROM quality_checks ORDER BY check_name;
