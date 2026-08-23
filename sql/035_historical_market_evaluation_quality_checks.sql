WITH quality_checks AS (
    SELECT 'duplicate_market_game' AS check_name, COUNT(*) AS issue_count
    FROM (
        SELECT game_id, market_key
        FROM analytics.historical_betting_ledger
        GROUP BY game_id, market_key
        HAVING COUNT(*) > 1
    )

    UNION ALL

    SELECT 'invalid_settlement', COUNT(*)
    FROM analytics.historical_betting_ledger
    WHERE result NOT IN ('WIN', 'LOSS', 'PUSH')
       OR decimal_odds <= 1.0
       OR (result = 'WIN' AND profit_per_unit <= 0.0)
       OR (result = 'LOSS' AND profit_per_unit <> -1.0)
       OR (result = 'PUSH' AND profit_per_unit <> 0.0)

    UNION ALL

    SELECT 'invalid_market_or_season', COUNT(*)
    FROM analytics.historical_betting_ledger
    WHERE market_key NOT IN ('h2h', 'spreads', 'totals')
       OR pricing_basis NOT IN ('SYNTHETIC_CLOSE_FAIR', 'CLOSING_PRICE')
       OR season NOT BETWEEN 2021 AND 2024

    UNION ALL

    SELECT 'missing_market_summary', COUNT(*)
    FROM (VALUES ('h2h'), ('spreads'), ('totals')) AS expected(market_key)
    LEFT JOIN analytics.historical_betting_performance AS actual
      ON expected.market_key = actual.market_key
     AND actual.edge_bucket = 'ALL'
    WHERE actual.market_key IS NULL

    UNION ALL

    SELECT 'summary_bet_count_mismatch', COUNT(*)
    FROM (
        SELECT market_key, COUNT(*) AS ledger_count
        FROM analytics.historical_betting_ledger
        GROUP BY market_key
    ) AS ledger
    JOIN analytics.historical_betting_performance AS summary
      ON ledger.market_key = summary.market_key
     AND summary.edge_bucket = 'ALL'
    WHERE ledger.ledger_count <> summary.bet_count
)
SELECT check_name, issue_count,
       CASE WHEN issue_count = 0 THEN 'PASS' ELSE 'FAIL' END AS status
FROM quality_checks
ORDER BY check_name;
