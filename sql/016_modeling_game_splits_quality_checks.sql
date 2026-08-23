-- =========================================================
-- NFL Analytics Platform
-- Modeling Game Splits Quality Checks
-- =========================================================

WITH quality_checks AS (

    SELECT
        'duplicate_game_id' AS check_name,
        COUNT(*) AS issue_count

    FROM (
        SELECT game_id
        FROM analytics.modeling_game_splits
        GROUP BY game_id
        HAVING COUNT(*) > 1
    )

    UNION ALL

    SELECT
        'row_count_matches_dataset',
        ABS(
            (
                SELECT COUNT(*)
                FROM analytics.modeling_game_splits
            )
            -
            (
                SELECT COUNT(*)
                FROM analytics.game_modeling_dataset
                WHERE season BETWEEN 2018 AND 2025
            )
        )

    UNION ALL

    SELECT
        'invalid_split_name',
        COUNT(*)

    FROM analytics.modeling_game_splits

    WHERE split_name NOT IN (
        'train',
        'validation',
        'holdout'
    )

    UNION ALL

    SELECT
        'invalid_split_order',
        COUNT(*)

    FROM analytics.modeling_game_splits

    WHERE split_order IS DISTINCT FROM (
        CASE
            WHEN split_name = 'train' THEN 1
            WHEN split_name = 'validation' THEN 2
            WHEN split_name = 'holdout' THEN 3
            ELSE NULL
        END
    )

    UNION ALL

    SELECT
        'invalid_season_assignment',
        COUNT(*)

    FROM analytics.modeling_game_splits

    WHERE split_name IS DISTINCT FROM (
        CASE
            WHEN season BETWEEN 2018 AND 2022
                THEN 'train'
            WHEN season BETWEEN 2023 AND 2024
                THEN 'validation'
            WHEN season = 2025
                THEN 'holdout'
            ELSE NULL
        END
    )

    UNION ALL

    SELECT
        'missing_required_split',
        ABS(
            3
            -
            (
                SELECT COUNT(DISTINCT split_name)
                FROM analytics.modeling_game_splits
                WHERE split_name IN (
                    'train',
                    'validation',
                    'holdout'
                )
            )
        )

    UNION ALL

    SELECT
        'invalid_split_chronology',
        COUNT(*)

    FROM (
        SELECT
            (
                SELECT MAX(game_date)
                FROM analytics.modeling_game_splits
                WHERE split_name = 'train'
            ) AS train_last_date,

            (
                SELECT MIN(game_date)
                FROM analytics.modeling_game_splits
                WHERE split_name = 'validation'
            ) AS validation_first_date,

            (
                SELECT MAX(game_date)
                FROM analytics.modeling_game_splits
                WHERE split_name = 'validation'
            ) AS validation_last_date,

            (
                SELECT MIN(game_date)
                FROM analytics.modeling_game_splits
                WHERE split_name = 'holdout'
            ) AS holdout_first_date
    )

    WHERE train_last_date >= validation_first_date
       OR validation_last_date >= holdout_first_date

    UNION ALL

    SELECT
        'invalid_binary_target_eligibility',
        COUNT(*)

    FROM analytics.modeling_game_splits AS splits

    INNER JOIN analytics.game_modeling_dataset AS dataset
        ON splits.game_id = dataset.game_id

    WHERE splits.is_binary_target_eligible
        IS DISTINCT FROM (
            dataset.target_home_win IS NOT NULL
        )

    UNION ALL

    SELECT
        'invalid_history_eligibility',
        COUNT(*)

    FROM analytics.modeling_game_splits AS splits

    INNER JOIN analytics.game_modeling_dataset AS dataset
        ON splits.game_id = dataset.game_id

    WHERE splits.has_complete_short_history
            IS DISTINCT FROM (
                dataset.both_short_windows_complete
            )

       OR splits.has_complete_long_history
            IS DISTINCT FROM (
                dataset.both_long_windows_complete
            )

    UNION ALL

    SELECT
        'invalid_qb_eligibility',
        COUNT(*)

    FROM analytics.modeling_game_splits AS splits

    INNER JOIN analytics.game_modeling_dataset AS dataset
        ON splits.game_id = dataset.game_id

    WHERE splits.has_both_qb_ratings
        IS DISTINCT FROM (
            dataset.both_listed_qb_ratings_available
        )

    UNION ALL

    SELECT
        'invalid_model_eligibility',
        COUNT(*)

    FROM analytics.modeling_game_splits AS splits

    INNER JOIN analytics.game_modeling_dataset AS dataset
        ON splits.game_id = dataset.game_id

    WHERE splits.is_core_model_eligible
            IS DISTINCT FROM (
                dataset.target_home_win IS NOT NULL
                AND dataset.both_short_windows_complete
                AND dataset.both_listed_qb_ratings_available
            )

       OR splits.is_extended_model_eligible
            IS DISTINCT FROM (
                dataset.target_home_win IS NOT NULL
                AND dataset.both_long_windows_complete
                AND dataset.both_listed_qb_ratings_available
            )

    UNION ALL

    SELECT
        'orphan_split_record',
        COUNT(*)

    FROM analytics.modeling_game_splits AS splits

    LEFT JOIN analytics.game_modeling_dataset AS dataset
        ON splits.game_id = dataset.game_id

    WHERE dataset.game_id IS NULL
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