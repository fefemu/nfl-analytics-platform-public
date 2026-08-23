-- =====================================================
-- NFL Analytics Platform
-- QB Game Performance Quality Checks
--
-- Purpose:
--     Validate processed.qb_game_performance.
--
-- Author:
--     Ferenc Kaizer
--
-- Version:
--     0.1.0
-- =====================================================


WITH team_game_summary AS (
    SELECT
        game_id,
        team,
        COUNT(*) AS qb_count,
        SUM(
            CASE
                WHEN is_primary_qb THEN 1
                ELSE 0
            END
        ) AS primary_qb_count,
        SUM(
            CASE
                WHEN is_listed_starter THEN 1
                ELSE 0
            END
        ) AS matched_starter_count,
        SUM(team_dropback_share) AS total_dropback_share,
        MAX(dropbacks) AS maximum_qb_dropbacks
    FROM processed.qb_game_performance
    GROUP BY
        game_id,
        team
),

checks AS (

    -- -------------------------------------------------
    -- Business key must be unique
    -- -------------------------------------------------

    SELECT
        'duplicate_game_team_qb' AS check_name,
        COUNT(*) AS issue_count
    FROM (
        SELECT
            game_id,
            team,
            qb_id
        FROM processed.qb_game_performance
        GROUP BY
            game_id,
            team,
            qb_id
        HAVING COUNT(*) > 1
    )

    UNION ALL

    -- -------------------------------------------------
    -- Required QB identity cannot be missing
    -- -------------------------------------------------

    SELECT
        'missing_qb_identity',
        COUNT(*)
    FROM processed.qb_game_performance
    WHERE game_id IS NULL
       OR team IS NULL
       OR opponent IS NULL
       OR qb_id IS NULL
       OR qb_name IS NULL

    UNION ALL

    -- -------------------------------------------------
    -- Every team-game must have one primary QB
    -- -------------------------------------------------

    SELECT
        'invalid_primary_qb_count',
        COUNT(*)
    FROM team_game_summary
    WHERE primary_qb_count <> 1

    UNION ALL

    -- -------------------------------------------------
    -- Primary QB must have the most dropbacks
    -- -------------------------------------------------

    SELECT
        'primary_qb_not_maximum_dropbacks',
        COUNT(*)
    FROM processed.qb_game_performance AS qb
    INNER JOIN team_game_summary AS summary
        ON qb.game_id = summary.game_id
       AND qb.team = summary.team
    WHERE qb.is_primary_qb
      AND qb.dropbacks
          <> summary.maximum_qb_dropbacks

    UNION ALL

    -- -------------------------------------------------
    -- Team QB shares must total one
    -- -------------------------------------------------

    SELECT
        'invalid_team_dropback_share',
        COUNT(*)
    FROM team_game_summary
    WHERE ABS(total_dropback_share - 1.0)
          > 0.000000001

    UNION ALL

    -- -------------------------------------------------
    -- Counts must be logically valid
    -- -------------------------------------------------

    SELECT
        'invalid_qb_count',
        COUNT(*)
    FROM processed.qb_game_performance
    WHERE dropbacks <= 0
       OR competitive_dropbacks < 0
       OR competitive_dropbacks > dropbacks
       OR throw_attempts < 0
       OR throw_attempts > dropbacks
       OR completions < 0
       OR completions > throw_attempts
       OR incompletions < 0
       OR incompletions > throw_attempts
       OR sacks < 0
       OR sacks > dropbacks
       OR qb_hits < 0
       OR scrambles < 0
       OR scrambles > dropbacks
       OR interceptions < 0
       OR interceptions > throw_attempts
       OR fumbles_lost < 0
       OR turnovers < 0

    UNION ALL

    -- -------------------------------------------------
    -- Rates must be between zero and one
    -- -------------------------------------------------

    SELECT
        'invalid_qb_rate',
        COUNT(*)
    FROM processed.qb_game_performance
    WHERE success_rate NOT BETWEEN 0 AND 1
       OR team_dropback_share NOT BETWEEN 0 AND 1
       OR sack_rate NOT BETWEEN 0 AND 1
       OR qb_hit_rate NOT BETWEEN 0 AND 1
       OR scramble_rate NOT BETWEEN 0 AND 1
       OR turnover_rate NOT BETWEEN 0 AND 1
       OR (
            completion_rate IS NOT NULL
            AND completion_rate NOT BETWEEN 0 AND 1
       )
       OR (
            interception_rate IS NOT NULL
            AND interception_rate NOT BETWEEN 0 AND 1
       )

    UNION ALL

    -- -------------------------------------------------
    -- Team assignments must be valid
    -- -------------------------------------------------

    SELECT
        'invalid_team_assignment',
        COUNT(*)
    FROM processed.qb_game_performance
    WHERE team = opponent
       OR team NOT IN (home_team, away_team)
       OR opponent NOT IN (home_team, away_team)
       OR (
            is_home
            AND team <> home_team
       )
       OR (
            NOT is_home
            AND team <> away_team
       )

    UNION ALL

    -- -------------------------------------------------
    -- Every QB game must match the schedule
    -- -------------------------------------------------

    SELECT
        'orphan_schedule_game',
        COUNT(*)
    FROM processed.qb_game_performance AS qb
    LEFT JOIN processed.schedule AS schedule
        ON qb.game_id = schedule.game_id
    WHERE schedule.game_id IS NULL

    UNION ALL

    -- -------------------------------------------------
    -- Monitor schedule-to-PBP starter coverage
    -- A small mismatch is expected and informative
    -- -------------------------------------------------

    SELECT
        'starter_match_below_95_percent',
        CASE
            WHEN (
                SELECT
                    AVG(
                        CASE
                            WHEN matched_starter_count > 0
                            THEN 1.0
                            ELSE 0.0
                        END
                    )
                FROM team_game_summary
            ) >= 0.95
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
FROM checks
ORDER BY check_name;