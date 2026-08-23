-- =========================================================
-- NFL Analytics Platform
-- Snap Count and Player Usage Quality Checks
-- =========================================================

WITH quality_checks AS (

    SELECT
        'raw_snap_count_row_count_is_205354'
            AS check_name,

        CASE
            WHEN (
                SELECT COUNT(*)
                FROM raw.player_snap_counts
            ) = 205354
            THEN 0
            ELSE 1
        END AS issue_count

    UNION ALL

    SELECT
        'raw_snap_count_seasons_are_2018_2025',

        CASE
            WHEN (
                SELECT COUNT(DISTINCT season)
                FROM raw.player_snap_counts
            ) = 8
             AND (
                SELECT MIN(season)
                FROM raw.player_snap_counts
            ) = 2018
             AND (
                SELECT MAX(season)
                FROM raw.player_snap_counts
            ) = 2025
            THEN 0
            ELSE 1
        END

    UNION ALL

    SELECT
        'raw_snap_count_null_business_keys',
        COUNT(*)

    FROM raw.player_snap_counts

    WHERE game_id IS NULL
       OR season IS NULL
       OR week IS NULL
       OR pfr_player_id IS NULL
       OR team IS NULL
       OR opponent IS NULL

    UNION ALL

    SELECT
        'raw_snap_count_duplicate_player_team_games',
        COUNT(*)

    FROM (
        SELECT
            game_id,
            team,
            pfr_player_id
        FROM raw.player_snap_counts
        GROUP BY
            game_id,
            team,
            pfr_player_id
        HAVING COUNT(*) > 1
    )

    UNION ALL

    SELECT
        'raw_snap_count_invalid_source_shares',
        COUNT(*)

    FROM raw.player_snap_counts

    WHERE offense_pct NOT BETWEEN 0.0 AND 1.0
       OR defense_pct NOT BETWEEN 0.0 AND 1.0
       OR st_pct NOT BETWEEN 0.0 AND 1.01

    UNION ALL

    SELECT
        'raw_snap_count_rounded_shares_above_one_is_3',

        CASE
            WHEN (
                SELECT COUNT(*)
                FROM raw.player_snap_counts
                WHERE offense_pct > 1.0
                   OR defense_pct > 1.0
                   OR st_pct > 1.0
            ) = 3
            THEN 0
            ELSE 1
        END

    UNION ALL

    SELECT
        'player_directory_row_count_is_25035',

        CASE
            WHEN (
                SELECT COUNT(*)
                FROM raw.player_directory
            ) = 25035
            THEN 0
            ELSE 1
        END

    UNION ALL

    SELECT
        'player_directory_null_gsis_ids',
        COUNT(*)

    FROM raw.player_directory

    WHERE gsis_id IS NULL

    UNION ALL

    SELECT
        'player_directory_duplicate_gsis_ids',
        COUNT(*)

    FROM (
        SELECT gsis_id
        FROM raw.player_directory
        GROUP BY gsis_id
        HAVING COUNT(*) > 1
    )

    UNION ALL

    SELECT
        'player_directory_duplicate_non_null_pfr_ids',
        COUNT(*)

    FROM (
        SELECT pfr_id
        FROM raw.player_directory
        WHERE pfr_id IS NOT NULL
        GROUP BY pfr_id
        HAVING COUNT(*) > 1
    )

    UNION ALL

    SELECT
        'player_directory_duplicate_non_null_espn_ids',
        COUNT(*)

    FROM (
        SELECT espn_id
        FROM raw.player_directory
        WHERE espn_id IS NOT NULL
        GROUP BY espn_id
        HAVING COUNT(*) > 1
    )

    UNION ALL

    SELECT
        'processed_snap_count_matches_raw_count',

        CASE
            WHEN (
                SELECT COUNT(*)
                FROM processed.player_game_snap_counts
            ) = (
                SELECT COUNT(*)
                FROM raw.player_snap_counts
            )
            THEN 0
            ELSE 1
        END

    UNION ALL

    SELECT
        'processed_snap_count_null_business_keys',
        COUNT(*)

    FROM processed.player_game_snap_counts

    WHERE game_id IS NULL
       OR season IS NULL
       OR gameday IS NULL
       OR team IS NULL
       OR opponent IS NULL
       OR player_key IS NULL
       OR is_home IS NULL

    UNION ALL

    SELECT
        'processed_snap_count_duplicate_business_keys',
        COUNT(*)

    FROM (
        SELECT
            game_id,
            team,
            player_key
        FROM processed.player_game_snap_counts
        GROUP BY
            game_id,
            team,
            player_key
        HAVING COUNT(*) > 1
    )

    UNION ALL

    SELECT
        'processed_snap_count_invalid_normalized_shares',
        COUNT(*)

    FROM processed.player_game_snap_counts

    WHERE offense_snap_share
            NOT BETWEEN 0.0 AND 1.0
       OR defense_snap_share
            NOT BETWEEN 0.0 AND 1.0
       OR special_teams_snap_share
            NOT BETWEEN 0.0 AND 1.0

    UNION ALL

    SELECT
        'processed_snap_count_directory_fallback_rows_is_206',

        CASE
            WHEN (
                SELECT COUNT(*)
                FROM processed.player_game_snap_counts
                WHERE NOT has_player_directory_match
            ) = 206
            THEN 0
            ELSE 1
        END

    UNION ALL

    SELECT
        'processed_snap_count_directory_fallback_players_is_24',

        CASE
            WHEN (
                SELECT COUNT(DISTINCT player_key)
                FROM processed.player_game_snap_counts
                WHERE NOT has_player_directory_match
            ) = 24
            THEN 0
            ELSE 1
        END

    UNION ALL

    SELECT
        'snap_history_row_count_matches_processed',

        CASE
            WHEN (
                SELECT COUNT(*)
                FROM analytics.player_snap_share_history
            ) = (
                SELECT COUNT(*)
                FROM processed.player_game_snap_counts
            )
            THEN 0
            ELSE 1
        END

    UNION ALL

    SELECT
        'snap_history_duplicate_business_keys',
        COUNT(*)

    FROM (
        SELECT
            game_id,
            team,
            player_key
        FROM analytics.player_snap_share_history
        GROUP BY
            game_id,
            team,
            player_key
        HAVING COUNT(*) > 1
    )

    UNION ALL

    SELECT
        'snap_history_invalid_availability_date',
        COUNT(*)

    FROM analytics.player_snap_share_history

    WHERE available_after_gameday
            IS DISTINCT FROM gameday

    UNION ALL

    SELECT
        'snap_history_invalid_rolling_shares',
        COUNT(*)

    FROM analytics.player_snap_share_history

    WHERE career_offense_snap_share_last_4
            NOT BETWEEN 0.0 AND 1.0
       OR career_defense_snap_share_last_4
            NOT BETWEEN 0.0 AND 1.0
       OR career_special_teams_snap_share_last_4
            NOT BETWEEN 0.0 AND 1.0
       OR career_offense_snap_share_last_8
            NOT BETWEEN 0.0 AND 1.0
       OR career_defense_snap_share_last_8
            NOT BETWEEN 0.0 AND 1.0
       OR career_special_teams_snap_share_last_8
            NOT BETWEEN 0.0 AND 1.0
       OR team_offense_snap_share_last_4
            NOT BETWEEN 0.0 AND 1.0
       OR team_defense_snap_share_last_4
            NOT BETWEEN 0.0 AND 1.0
       OR team_special_teams_snap_share_last_4
            NOT BETWEEN 0.0 AND 1.0
       OR team_offense_snap_share_last_8
            NOT BETWEEN 0.0 AND 1.0
       OR team_defense_snap_share_last_8
            NOT BETWEEN 0.0 AND 1.0
       OR team_special_teams_snap_share_last_8
            NOT BETWEEN 0.0 AND 1.0
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