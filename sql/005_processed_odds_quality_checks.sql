-- =====================================================
-- NFL Analytics Platform
-- File: 005_processed_odds_quality_checks.sql
--
-- Purpose:
--     Validate analytics-ready odds calculations,
--     probabilities and market groupings.
--
-- Author:
--     Ferenc Kaizer
--
-- Version:
--     0.1.0
-- =====================================================


WITH quality_checks AS (
    SELECT
        'row_count_matches_raw' AS check_name,
        ABS(
            (
                SELECT COUNT(*)
                FROM raw.odds_markets
            )
            - COUNT(*)
        ) AS issue_count
    FROM processed.odds_market_outcomes

    UNION ALL

    SELECT
        'missing_calculated_value',
        COUNT(*)
    FROM processed.odds_market_outcomes
    WHERE decimal_odds IS NULL
       OR implied_probability IS NULL
       OR bookmaker_margin IS NULL
       OR no_vig_probability IS NULL

    UNION ALL

    SELECT
        'invalid_decimal_odds',
        COUNT(*)
    FROM processed.odds_market_outcomes
    WHERE decimal_odds <= 1.0

    UNION ALL

    SELECT
        'invalid_implied_probability',
        COUNT(*)
    FROM processed.odds_market_outcomes
    WHERE implied_probability <= 0.0
       OR implied_probability >= 1.0

    UNION ALL

    SELECT
        'invalid_no_vig_probability',
        COUNT(*)
    FROM processed.odds_market_outcomes
    WHERE no_vig_probability <= 0.0
       OR no_vig_probability >= 1.0

    UNION ALL

    SELECT
        'no_vig_group_not_equal_to_one',
        COUNT(*)
    FROM (
        SELECT
            snapshot_id,
            event_id,
            bookmaker_key,
            market_key,
            market_line,
            SUM(no_vig_probability) AS probability_sum
        FROM processed.odds_market_outcomes
        GROUP BY
            snapshot_id,
            event_id,
            bookmaker_key,
            market_key,
            market_line
        HAVING ABS(probability_sum - 1.0) > 0.000001
    )

    UNION ALL

    SELECT
        'unknown_outcome_type',
        COUNT(*)
    FROM processed.odds_market_outcomes
    WHERE outcome_type = 'other'

    UNION ALL

    SELECT
        'invalid_moneyline_market_line',
        COUNT(*)
    FROM processed.odds_market_outcomes
    WHERE market_key = 'h2h'
      AND market_line IS NOT NULL

    UNION ALL

    SELECT
        'missing_spread_or_total_market_line',
        COUNT(*)
    FROM processed.odds_market_outcomes
    WHERE market_key IN (
        'spreads',
        'totals'
    )
      AND market_line IS NULL

    UNION ALL

    SELECT
        'american_to_decimal_mismatch',
        COUNT(*)
    FROM processed.odds_market_outcomes
    WHERE ABS(
        decimal_odds
        - CASE
            WHEN american_price >= 100
                THEN 1.0 + american_price / 100.0
            WHEN american_price <= -100
                THEN 1.0
                    + 100.0 / ABS(american_price)
            ELSE NULL
        END
    ) > 0.000001

    UNION ALL

    SELECT
        'implied_probability_mismatch',
        COUNT(*)
    FROM processed.odds_market_outcomes
    WHERE ABS(
        implied_probability
        - 1.0 / decimal_odds
    ) > 0.000001
)

SELECT
    check_name,
    issue_count,
    CASE
        WHEN issue_count = 0 THEN 'PASS'
        ELSE 'FAIL'
    END AS status
FROM quality_checks
ORDER BY check_name;