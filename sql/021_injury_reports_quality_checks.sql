-- =========================================================
-- NFL Analytics Platform
-- Injury Report Quality Checks
-- =========================================================

WITH quality_checks AS (

    SELECT
        'raw_row_count_is_45337' AS check_name,

        CASE
            WHEN (
                SELECT COUNT(*)
                FROM raw.injury_reports
            ) = 45337
            THEN 0
            ELSE 1
        END AS issue_count

    UNION ALL

    SELECT
        'raw_season_coverage_is_2018_2025',

        CASE
            WHEN (
                SELECT COUNT(DISTINCT season)
                FROM raw.injury_reports
            ) = 8
             AND (
                SELECT MIN(season)
                FROM raw.injury_reports
            ) = 2018
             AND (
                SELECT MAX(season)
                FROM raw.injury_reports
            ) = 2025
            THEN 0
            ELSE 1
        END

    UNION ALL

    SELECT
        'raw_null_business_keys',
        COUNT(*)

    FROM raw.injury_reports

    WHERE season IS NULL
       OR game_type IS NULL
       OR team IS NULL
       OR week IS NULL
       OR gsis_id IS NULL

    UNION ALL

    SELECT
        'raw_duplicate_snapshot_records',
        COUNT(*)

    FROM (
        SELECT
            season,
            game_type,
            team,
            week,
            gsis_id,
            date_modified
        FROM raw.injury_reports
        GROUP BY
            season,
            game_type,
            team,
            week,
            gsis_id,
            date_modified
        HAVING COUNT(*) > 1
    )

    UNION ALL

    SELECT
        'raw_invalid_report_status',
        COUNT(*)

    FROM raw.injury_reports

    WHERE report_status IS NOT NULL
      AND TRIM(
            CAST(report_status AS VARCHAR)
          ) NOT IN (
            '',
            'Out',
            'Doubtful',
            'Questionable',
            'Note'
          )

    UNION ALL

    SELECT
        'raw_invalid_practice_status',
        COUNT(*)

    FROM raw.injury_reports

    WHERE practice_status IS NOT NULL
      AND NOT REGEXP_MATCHES(
            CAST(practice_status AS VARCHAR),
            '^\s*$'
          )
      AND TRIM(
            CAST(practice_status AS VARCHAR)
          ) NOT IN (
            'Did Not Participate In Practice',
            'Limited Participation in Practice',
            'Full Participation in Practice',
            'Note'
          )

    UNION ALL

    SELECT
        'historical_timestamp_coverage_invalid',
        COUNT(*)

    FROM raw.injury_reports

    WHERE season BETWEEN 2018 AND 2024
      AND date_modified IS NULL

    UNION ALL

    SELECT
        '2025_contains_unexpected_timestamp',
        COUNT(*)

    FROM raw.injury_reports

    WHERE season = 2025
      AND date_modified IS NOT NULL

    UNION ALL

    SELECT
        'processed_row_count_is_45318',

        CASE
            WHEN (
                SELECT COUNT(*)
                FROM processed.player_game_injury_status
            ) = 45318
            THEN 0
            ELSE 1
        END

    UNION ALL

    SELECT
        'processed_duplicate_player_game',
        COUNT(*)

    FROM (
        SELECT
            game_id,
            team,
            gsis_id
        FROM processed.player_game_injury_status
        GROUP BY
            game_id,
            team,
            gsis_id
        HAVING COUNT(*) > 1
    )

    UNION ALL

    SELECT
        'processed_null_business_keys',
        COUNT(*)

    FROM processed.player_game_injury_status

    WHERE game_id IS NULL
       OR season IS NULL
       OR game_type IS NULL
       OR team IS NULL
       OR opponent IS NULL
       OR gsis_id IS NULL

    UNION ALL

    SELECT
        'processed_invalid_report_status',
        COUNT(*)

    FROM processed.player_game_injury_status

    WHERE report_status IS NOT NULL
      AND report_status NOT IN (
            'Out',
            'Doubtful',
            'Questionable'
          )

    UNION ALL

    SELECT
        'processed_invalid_practice_status',
        COUNT(*)

    FROM processed.player_game_injury_status

    WHERE practice_status IS NOT NULL
      AND practice_status NOT IN (
            'Did Not Participate In Practice',
            'Limited Participation in Practice',
            'Full Participation in Practice'
          )

    UNION ALL

    SELECT
        'processed_report_flags_inconsistent',
        COUNT(*)

    FROM processed.player_game_injury_status

    WHERE is_out
        IS DISTINCT FROM
            COALESCE(
                report_status = 'Out',
                FALSE
            )
       OR is_doubtful
        IS DISTINCT FROM
            COALESCE(
                report_status = 'Doubtful',
                FALSE
            )
       OR is_questionable
        IS DISTINCT FROM
            COALESCE(
                report_status = 'Questionable',
                FALSE
            )

    UNION ALL

    SELECT
        'processed_practice_flags_inconsistent',
        COUNT(*)

    FROM processed.player_game_injury_status

    WHERE did_not_practice
        IS DISTINCT FROM
            COALESCE(
                practice_status =
                    'Did Not Participate In Practice',
                FALSE
            )
       OR limited_practice
        IS DISTINCT FROM
            COALESCE(
                practice_status =
                    'Limited Participation in Practice',
                FALSE
            )
       OR full_practice
        IS DISTINCT FROM
            COALESCE(
                practice_status =
                    'Full Participation in Practice',
                FALSE
            )

    UNION ALL

    SELECT
        'invalid_snapshot_count',
        COUNT(*)

    FROM processed.player_game_injury_status

    WHERE source_snapshot_count < 1
       OR source_snapshot_count IS NULL

    UNION ALL

    SELECT
        'multiple_snapshot_player_count_is_2',

        CASE
            WHEN (
                SELECT COUNT(*)
                FROM processed.player_game_injury_status
                WHERE source_snapshot_count > 1
            ) = 2
            THEN 0
            ELSE 1
        END

    UNION ALL

    SELECT
        'processed_timestamp_flag_inconsistent',
        COUNT(*)

    FROM processed.player_game_injury_status

    WHERE has_source_timestamp
        IS DISTINCT FROM
            (
                source_date_modified IS NOT NULL
            )

    UNION ALL

    SELECT
        'nonstandard_status_not_cleaned',
        COUNT(*)

    FROM processed.player_game_injury_status

    WHERE report_status = 'Note'
       OR practice_status = 'Note'
       OR TRIM(
            COALESCE(
                practice_status,
                ''
            )
          ) = ''
          AND practice_status IS NOT NULL

    UNION ALL

    SELECT
        'processed_schedule_attributes_mismatch',
        COUNT(*)

    FROM processed.player_game_injury_status
        AS injury

    INNER JOIN processed.schedule
        AS schedule
        ON injury.game_id = schedule.game_id

    WHERE injury.season
            IS DISTINCT FROM schedule.season
       OR injury.game_type
            IS DISTINCT FROM schedule.game_type
       OR injury.week
            IS DISTINCT FROM schedule.week
       OR injury.gameday
            IS DISTINCT FROM schedule.gameday
       OR (
            injury.is_home
            AND injury.team
                IS DISTINCT FROM schedule.home_team
          )
       OR (
            NOT injury.is_home
            AND injury.team
                IS DISTINCT FROM schedule.away_team
          )

    UNION ALL

    SELECT
        'processed_game_missing_from_schedule',
        COUNT(*)

    FROM processed.player_game_injury_status
        AS injury

    LEFT JOIN processed.schedule
        AS schedule
        ON injury.game_id = schedule.game_id

    WHERE schedule.game_id IS NULL

    UNION ALL

    SELECT
        'known_unplayed_key_count_is_17',

        CASE
            WHEN (
                WITH injury_keys AS (
                    SELECT DISTINCT
                        season,
                        game_type,
                        team,
                        week,
                        gsis_id
                    FROM raw.injury_reports
                )
                SELECT COUNT(*)
                FROM injury_keys AS injury
                LEFT JOIN processed.schedule AS schedule
                    ON injury.season = schedule.season
                   AND injury.game_type = schedule.game_type
                   AND injury.week = schedule.week
                   AND (
                        injury.team = schedule.home_team
                        OR injury.team = schedule.away_team
                   )
                WHERE schedule.game_id IS NULL
                  AND injury.season = 2022
                  AND injury.game_type = 'REG'
                  AND injury.week = 17
                  AND injury.team IN (
                        'BUF',
                        'CIN'
                  )
            ) = 17
            THEN 0
            ELSE 1
        END

    UNION ALL

    SELECT
        'unexpected_unmatched_injury_key',
        COUNT(*)

    FROM (
        WITH injury_keys AS (
            SELECT DISTINCT
                season,
                game_type,
                team,
                week,
                gsis_id
            FROM raw.injury_reports
        )
        SELECT
            injury.season,
            injury.game_type,
            injury.team,
            injury.week,
            injury.gsis_id
        FROM injury_keys AS injury
        LEFT JOIN processed.schedule AS schedule
            ON injury.season = schedule.season
           AND injury.game_type = schedule.game_type
           AND injury.week = schedule.week
           AND (
                injury.team = schedule.home_team
                OR injury.team = schedule.away_team
           )
        WHERE schedule.game_id IS NULL
          AND NOT (
                injury.season = 2022
                AND injury.game_type = 'REG'
                AND injury.week = 17
                AND injury.team IN (
                    'BUF',
                    'CIN'
                )
          )
    )

    UNION ALL

    SELECT
        'latest_timestamped_snapshot_not_selected',
        COUNT(*)

    FROM processed.player_game_injury_status
        AS processed_injury

    INNER JOIN raw.injury_reports
        AS raw_injury
        ON processed_injury.season = raw_injury.season
       AND processed_injury.game_type = raw_injury.game_type
       AND processed_injury.team = raw_injury.team
       AND processed_injury.week = raw_injury.week
       AND processed_injury.gsis_id = raw_injury.gsis_id

    WHERE raw_injury.date_modified
            > processed_injury.source_date_modified
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