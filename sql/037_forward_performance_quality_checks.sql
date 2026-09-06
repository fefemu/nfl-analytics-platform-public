WITH checks AS (
    SELECT 'duplicate_settlement_key' AS check_name, COUNT(*) AS violation_count
    FROM (
        SELECT entry_archive_key
        FROM analytics.forward_tip_settlement
        GROUP BY 1 HAVING COUNT(*) > 1
    )
    UNION ALL
    SELECT 'invalid_settlement_arithmetic', COUNT(*)
    FROM analytics.forward_tip_settlement
    WHERE settlement_status NOT IN ('PENDING', 'SETTLED')
       OR (settlement_status = 'PENDING' AND (result IS NOT NULL OR profit_units IS NOT NULL))
       OR (settlement_status = 'SETTLED' AND result NOT IN ('WIN', 'LOSS', 'PUSH'))
       OR (result = 'LOSS' AND profit_units <> -1.0)
       OR (result = 'PUSH' AND profit_units <> 0.0)
       OR (result = 'WIN' AND ABS(profit_units - (entry_decimal_odds - 1.0)) > 0.000001)
    UNION ALL
    SELECT 'invalid_probability_scoring', COUNT(*)
    FROM analytics.forward_tip_settlement
    WHERE (is_probability_score_eligible AND (market_key <> 'h2h' OR result = 'PUSH'))
       OR (NOT is_probability_score_eligible AND (brier_loss IS NOT NULL OR log_loss IS NOT NULL))
       OR brier_loss NOT BETWEEN 0.0 AND 1.0
       OR log_loss < 0.0
    UNION ALL
    SELECT 'summary_count_mismatch', COUNT(*)
    FROM analytics.forward_performance_summary
    WHERE tracked_count <> settled_count + pending_count
       OR settled_count <> win_count + loss_count + push_count
       OR (settled_count = 0 AND roi_percent IS NOT NULL)
       OR (settled_count > 0 AND ABS(roi_percent - 100.0 * total_profit_units / settled_count) > 0.000001)
)
SELECT * FROM checks WHERE violation_count > 0;
