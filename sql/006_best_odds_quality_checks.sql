-- =====================================================
-- NFL Analytics Platform
-- File: 006_best_odds_quality_checks.sql
--
-- Purpose:
--     Validate best-price selection and bookmaker
--     consensus calculations.
--
-- Author:
--     Ferenc Kaizer
--
-- Version:
--     0.1.0
-- =====================================================


WITH quality_checks AS (
    SELECT
        'row_count_matches_offer_groups' AS check_name,
        ABS(
            (
                SELECT COUNT(*)
                FROM (
                    SELECT DISTINCT
                        snapshot_id,
                        event_id,
                        market_key,
                        outcome_type,
                        point
                    FROM processed.odds_market_outcomes
                )
            )
            - COUNT(*)
        ) AS issue_count
    FROM analytics.best_odds_by_line

    UNION ALL

    SELECT
        'duplicate_offer_group',
        COUNT(*)
    FROM (
        SELECT
            snapshot_id,
            event_id,
            market_key,
            outcome_type,
            point
        FROM analytics.best_odds_by_line
        GROUP BY
            snapshot_id,
            event_id,
            market_key,
            outcome_type,
            point
        HAVING COUNT(*) > 1
    )

    UNION ALL

    SELECT
        'incorrect_best_price',
        COUNT(*)
    FROM analytics.best_odds_by_line AS target
    WHERE target.best_decimal_odds IS DISTINCT FROM (
        SELECT MAX(source.decimal_odds)
        FROM processed.odds_market_outcomes AS source
        WHERE source.snapshot_id = target.snapshot_id
          AND source.event_id = target.event_id
          AND source.market_key = target.market_key
          AND source.outcome_type = target.outcome_type
          AND source.point IS NOT DISTINCT FROM target.point
    )

    UNION ALL

    SELECT
        'incorrect_bookmaker_count',
        COUNT(*)
    FROM analytics.best_odds_by_line AS target
    WHERE target.bookmaker_count <> (
        SELECT COUNT(*)
        FROM processed.odds_market_outcomes AS source
        WHERE source.snapshot_id = target.snapshot_id
          AND source.event_id = target.event_id
          AND source.market_key = target.market_key
          AND source.outcome_type = target.outcome_type
          AND source.point IS NOT DISTINCT FROM target.point
    )

    UNION ALL

    SELECT
        'invalid_consensus_probability',
        COUNT(*)
    FROM analytics.best_odds_by_line
    WHERE consensus_no_vig_probability <= 0.0
       OR consensus_no_vig_probability >= 1.0
       OR minimum_no_vig_probability <= 0.0
       OR maximum_no_vig_probability >= 1.0

    UNION ALL

    SELECT
        'consensus_outside_market_range',
        COUNT(*)
    FROM analytics.best_odds_by_line
    WHERE consensus_no_vig_probability
            < minimum_no_vig_probability
       OR consensus_no_vig_probability
            > maximum_no_vig_probability

    UNION ALL

    SELECT
        'negative_price_improvement',
        COUNT(*)
    FROM analytics.best_odds_by_line
    WHERE decimal_price_improvement < -0.000001

    UNION ALL

    SELECT
        'missing_best_bookmaker',
        COUNT(*)
    FROM analytics.best_odds_by_line
    WHERE best_bookmaker_key IS NULL
       OR best_decimal_odds IS NULL
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