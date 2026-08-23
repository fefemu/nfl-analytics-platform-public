-- =========================================================
-- NFL Analytics Platform
-- Depth Chart Quality Checks
-- =========================================================

WITH quality_checks AS (

    SELECT
        'raw_legacy_row_count_is_258942' AS check_name,

        CASE
            WHEN (
                SELECT COUNT(*)
                FROM raw.depth_charts_legacy
            ) = 258942
            THEN 0
            ELSE 1
        END AS issue_count

    UNION ALL

    SELECT
        'raw_espn_row_count_is_951797',

        CASE
            WHEN (
                SELECT COUNT(*)
                FROM raw.depth_charts_espn
            ) = 951797
            THEN 0
            ELSE 1
        END

    UNION ALL

    SELECT
        'raw_legacy_season_coverage_is_2018_2024',

        CASE
            WHEN (
                SELECT COUNT(DISTINCT season)
                FROM raw.depth_charts_legacy
            ) = 7
             AND (
                SELECT MIN(season)
                FROM raw.depth_charts_legacy
            ) = 2018
             AND (
                SELECT MAX(season)
                FROM raw.depth_charts_legacy
            ) = 2024
            THEN 0
            ELSE 1
        END

    UNION ALL

    SELECT
        'raw_espn_source_season_coverage_is_2025_2026',

        CASE
            WHEN (
                SELECT COUNT(DISTINCT source_season)
                FROM raw.depth_charts_espn
            ) = 2
             AND (
                SELECT MIN(source_season)
                FROM raw.depth_charts_espn
            ) = 2025
             AND (
                SELECT MAX(source_season)
                FROM raw.depth_charts_espn
            ) = 2026
            THEN 0
            ELSE 1
        END

    UNION ALL

    SELECT
        'raw_legacy_null_business_keys',
        COUNT(*)

    FROM raw.depth_charts_legacy

    WHERE season IS NULL
       OR club_code IS NULL
       OR game_type IS NULL
       OR gsis_id IS NULL
       OR formation IS NULL
       OR depth_position IS NULL
       OR depth_team IS NULL

    UNION ALL

    SELECT
        'raw_legacy_invalid_depth_rank',
        COUNT(*)

    FROM raw.depth_charts_legacy

    WHERE TRY_CAST(depth_team AS INTEGER)
            NOT BETWEEN 1 AND 3
       OR TRY_CAST(depth_team AS INTEGER) IS NULL

    UNION ALL

    SELECT
        'raw_legacy_invalid_formation',
        COUNT(*)

    FROM raw.depth_charts_legacy

    WHERE formation NOT IN (
            'Offense',
            'Defense',
            'Special Teams'
          )

    UNION ALL

    SELECT
        'raw_legacy_unexpected_null_week',
        COUNT(*)

    FROM raw.depth_charts_legacy

    WHERE week IS NULL
      AND game_type != 'SBBYE'

    UNION ALL

    SELECT
        'raw_espn_null_business_keys',
        COUNT(*)

    FROM raw.depth_charts_espn

    WHERE source_season IS NULL
       OR dt IS NULL
       OR team IS NULL
       OR espn_id IS NULL
       OR pos_grp IS NULL
       OR pos_name IS NULL
       OR pos_slot IS NULL
       OR pos_rank IS NULL

    UNION ALL

    SELECT
        'raw_espn_invalid_depth_rank',
        COUNT(*)

    FROM raw.depth_charts_espn

    WHERE pos_rank < 1

    UNION ALL

    SELECT
        'raw_espn_duplicate_business_role',
        COUNT(*)

    FROM (
        SELECT
            dt,
            team,
            espn_id,
            pos_grp,
            pos_name,
            pos_slot,
            pos_rank
        FROM raw.depth_charts_espn
        GROUP BY
            dt,
            team,
            espn_id,
            pos_grp,
            pos_name,
            pos_slot,
            pos_rank
        HAVING COUNT(*) > 1
    )

    UNION ALL

    SELECT
        'legacy_processed_row_count_is_225962',

        CASE
            WHEN (
                SELECT COUNT(*)
                FROM processed.player_game_depth_chart_legacy
            ) = 225962
            THEN 0
            ELSE 1
        END

    UNION ALL

    SELECT
        'espn_processed_row_count_is_95445',

        CASE
            WHEN (
                SELECT COUNT(*)
                FROM processed.player_game_depth_chart_espn
            ) = 95445
            THEN 0
            ELSE 1
        END

    UNION ALL

    SELECT
        'unified_row_count_is_321407',

        CASE
            WHEN (
                SELECT COUNT(*)
                FROM processed.player_game_depth_chart
            ) = 321407
            THEN 0
            ELSE 1
        END

    UNION ALL

    SELECT
        'unified_row_count_matches_sources',

        CASE
            WHEN (
                SELECT COUNT(*)
                FROM processed.player_game_depth_chart
            ) = (
                (
                    SELECT COUNT(*)
                    FROM processed.player_game_depth_chart_legacy
                )
                +
                (
                    SELECT COUNT(*)
                    FROM processed.player_game_depth_chart_espn
                )
            )
            THEN 0
            ELSE 1
        END

    UNION ALL

    SELECT
        'unified_team_game_count_is_4998',

        CASE
            WHEN (
                SELECT COUNT(*)
                FROM (
                    SELECT DISTINCT
                        game_id,
                        team
                    FROM processed.player_game_depth_chart
                )
            ) = 4998
            THEN 0
            ELSE 1
        END

    UNION ALL

    SELECT
        'unified_duplicate_role_records',
        COUNT(*)

    FROM (
        SELECT
            game_id,
            team,
            player_key,
            formation,
            depth_position,
            COALESCE(
                position_slot,
                -1
            ) AS position_slot_key
        FROM processed.player_game_depth_chart
        GROUP BY
            game_id,
            team,
            player_key,
            formation,
            depth_position,
            position_slot_key
        HAVING COUNT(*) > 1
    )

    UNION ALL

    SELECT
        'unified_null_business_keys',
        COUNT(*)

    FROM processed.player_game_depth_chart

    WHERE game_id IS NULL
       OR season IS NULL
       OR game_type IS NULL
       OR week IS NULL
       OR gameday IS NULL
       OR team IS NULL
       OR opponent IS NULL
       OR player_key IS NULL
       OR formation IS NULL
       OR depth_position IS NULL
       OR depth_rank IS NULL
       OR depth_tier IS NULL
       OR source_generation IS NULL

    UNION ALL

    SELECT
        'unified_invalid_source_generation',
        COUNT(*)

    FROM processed.player_game_depth_chart

    WHERE source_generation NOT IN (
            'legacy_nfl',
            'espn'
          )

    UNION ALL

    SELECT
        'unified_invalid_formation',
        COUNT(*)

    FROM processed.player_game_depth_chart

    WHERE formation NOT IN (
            'Offense',
            'Defense',
            'Special Teams'
          )

    UNION ALL

    SELECT
        'unified_invalid_depth_rank',
        COUNT(*)

    FROM processed.player_game_depth_chart

    WHERE depth_rank < 1
       OR source_record_count < 1
       OR source_rank_count < 1

    UNION ALL

    SELECT
        'unified_depth_rank_flags_inconsistent',
        COUNT(*)

    FROM processed.player_game_depth_chart

    WHERE is_starter
            IS DISTINCT FROM (
                depth_rank = 1
            )
       OR is_primary_backup
            IS DISTINCT FROM (
                depth_rank = 2
            )
       OR is_reserve
            IS DISTINCT FROM (
                depth_rank >= 3
            )

    UNION ALL

    SELECT
        'unified_depth_tier_inconsistent',
        COUNT(*)

    FROM processed.player_game_depth_chart

    WHERE depth_tier
            IS DISTINCT FROM
                CASE
                    WHEN is_starter
                        THEN 'STARTER'
                    WHEN is_primary_backup
                        THEN 'PRIMARY_BACKUP'
                    ELSE 'RESERVE'
                END

    UNION ALL

    SELECT
        'unified_formation_flags_inconsistent',
        COUNT(*)

    FROM processed.player_game_depth_chart

    WHERE is_offense_role
            IS DISTINCT FROM (
                formation = 'Offense'
            )
       OR is_defense_role
            IS DISTINCT FROM (
                formation = 'Defense'
            )
       OR is_special_teams_role
            IS DISTINCT FROM (
                formation = 'Special Teams'
            )

    UNION ALL

    SELECT
        'unified_identifier_flags_inconsistent',
        COUNT(*)

    FROM processed.player_game_depth_chart

    WHERE has_gsis_id
            IS DISTINCT FROM (
                gsis_id IS NOT NULL
            )
       OR player_identifier_source
            IS DISTINCT FROM
                CASE
                    WHEN gsis_id IS NOT NULL
                        THEN 'GSIS'
                    ELSE 'ESPN'
                END

    UNION ALL

    SELECT
        'unified_missing_gsis_count_is_382',

        CASE
            WHEN (
                SELECT COUNT(*)
                FROM processed.player_game_depth_chart
                WHERE NOT has_gsis_id
            ) = 382
            THEN 0
            ELSE 1
        END

    UNION ALL

    SELECT
        'missing_gsis_without_espn_fallback',
        COUNT(*)

    FROM processed.player_game_depth_chart

    WHERE gsis_id IS NULL
      AND (
            espn_id IS NULL
            OR player_key IS NULL
            OR player_identifier_source != 'ESPN'
          )

    UNION ALL

    SELECT
        'legacy_contains_espn_only_attributes',
        COUNT(*)

    FROM processed.player_game_depth_chart

    WHERE source_generation = 'legacy_nfl'
      AND (
            espn_id IS NOT NULL
            OR position_slot IS NOT NULL
            OR source_snapshot_at IS NOT NULL
            OR has_timestamped_snapshot
          )

    UNION ALL

    SELECT
        'espn_missing_timestamped_snapshot',
        COUNT(*)

    FROM processed.player_game_depth_chart

    WHERE source_generation = 'espn'
      AND (
            source_snapshot_at IS NULL
            OR NOT has_timestamped_snapshot
          )

    UNION ALL

    SELECT
        'espn_future_snapshot_selected',
        COUNT(*)

    FROM processed.player_game_depth_chart

    WHERE source_generation = 'espn'
      AND CAST(source_snapshot_at AS DATE) > gameday

    UNION ALL

    SELECT
        'unified_schedule_attributes_mismatch',
        COUNT(*)

    FROM processed.player_game_depth_chart
        AS depth_chart

    INNER JOIN processed.schedule
        AS schedule
        ON depth_chart.game_id = schedule.game_id

    WHERE depth_chart.season
            IS DISTINCT FROM schedule.season
       OR depth_chart.game_type
            IS DISTINCT FROM schedule.game_type
       OR depth_chart.week
            IS DISTINCT FROM schedule.week
       OR depth_chart.gameday
            IS DISTINCT FROM schedule.gameday
       OR (
            depth_chart.is_home
            AND depth_chart.team
                IS DISTINCT FROM schedule.home_team
          )
       OR (
            depth_chart.is_home
            AND depth_chart.opponent
                IS DISTINCT FROM schedule.away_team
          )
       OR (
            NOT depth_chart.is_home
            AND depth_chart.team
                IS DISTINCT FROM schedule.away_team
          )
       OR (
            NOT depth_chart.is_home
            AND depth_chart.opponent
                IS DISTINCT FROM schedule.home_team
          )

    UNION ALL

    SELECT
        'unified_game_missing_from_schedule',
        COUNT(*)

    FROM processed.player_game_depth_chart
        AS depth_chart

    LEFT JOIN processed.schedule
        AS schedule
        ON depth_chart.game_id = schedule.game_id

    WHERE schedule.game_id IS NULL

    UNION ALL

    SELECT
        'unified_starter_role_count_is_140208',

        CASE
            WHEN (
                SELECT COUNT(*)
                FROM processed.player_game_depth_chart
                WHERE is_starter
            ) = 140208
            THEN 0
            ELSE 1
        END
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