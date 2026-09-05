WITH quality_checks AS (
    SELECT 'duplicate_archive_key' AS check_name, COUNT(*) AS issue_count
    FROM (
        SELECT archive_key
        FROM analytics.forward_betting_board_archive
        GROUP BY archive_key
        HAVING COUNT(*) > 1
    )
    UNION ALL
    SELECT 'post_kickoff_archive_row', COUNT(*)
    FROM analytics.forward_betting_board_archive
    WHERE commence_time <= fetched_at
       OR commence_time <= prediction_generated_at
       OR commence_time <= betting_board_generated_at
       OR commence_time <= archived_at
    UNION ALL
    SELECT 'invalid_tip_candidate_flag', COUNT(*)
    FROM analytics.forward_betting_board_archive
    WHERE is_tip_candidate <> positive_expected_value
    UNION ALL
    SELECT 'invalid_refresh_run', COUNT(*)
    FROM analytics.refresh_run_history
    WHERE status NOT IN ('RUNNING', 'SUCCESS', 'FAILED')
       OR started_at IS NULL
       OR (status IN ('SUCCESS', 'FAILED') AND completed_at IS NULL)
       OR (status = 'SUCCESS' AND archived_market_row_count IS NULL)
       OR (status = 'FAILED' AND error_message IS NULL)
    UNION ALL
    SELECT 'invalid_market_movement_comparison_time', COUNT(*)
    FROM analytics.forward_tip_market_movement
    WHERE has_latest_pregame_comparison
      AND (latest_fetched_at <= entry_fetched_at
       OR latest_fetched_at >= commence_time)
    UNION ALL
    SELECT 'invalid_market_movement_label', COUNT(*)
    FROM analytics.forward_tip_market_movement
    WHERE comparison_type <> 'LATEST_PRE_KICKOFF'
       OR is_closing_snapshot
       OR is_clv
       OR market_movement_direction NOT IN (
           'POSITIVE', 'NEGATIVE', 'UNCHANGED', 'NO_LATER_SNAPSHOT'
       )
    UNION ALL
    SELECT 'duplicate_market_movement_entry', COUNT(*)
    FROM (
        SELECT game_id, market_key, outcome_type
        FROM analytics.forward_tip_market_movement
        GROUP BY ALL
        HAVING COUNT(*) > 1
    )
)
SELECT check_name, issue_count,
       CASE WHEN issue_count = 0 THEN 'PASS' ELSE 'FAIL' END AS status
FROM quality_checks
ORDER BY check_name;
