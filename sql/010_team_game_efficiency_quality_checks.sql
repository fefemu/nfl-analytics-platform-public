-- =====================================================
-- NFL Analytics Platform
-- Team-Game Efficiency Quality Checks
--
-- Purpose:
--     Validate processed.team_game_efficiency.
--
-- Author:
--     Ferenc Kaizer
--
-- Version:
--     0.1.0
-- =====================================================


WITH checks AS (

    -- -------------------------------------------------
    -- Business key must be unique
    -- -------------------------------------------------

    SELECT
        'duplicate_game_team' AS check_name,
        COUNT(*) AS issue_count
    FROM (
        SELECT
            game_id,
            team
        FROM processed.team_game_efficiency
        GROUP BY
            game_id,
            team
        HAVING COUNT(*) > 1
    )

    UNION ALL

    -- -------------------------------------------------
    -- Every game must have exactly two teams
    -- -------------------------------------------------

    SELECT
        'invalid_team_rows_per_game',
        COUNT(*)
    FROM (
        SELECT
            game_id
        FROM processed.team_game_efficiency
        GROUP BY game_id
        HAVING COUNT(*) <> 2
           OR COUNT(DISTINCT team) <> 2
    )

    UNION ALL

    -- -------------------------------------------------
    -- Team and opponent assignments must be valid
    -- -------------------------------------------------

    SELECT
        'invalid_team_assignment',
        COUNT(*)
    FROM processed.team_game_efficiency
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
    -- Required identifiers and results cannot be null
    -- -------------------------------------------------

    SELECT
        'missing_required_value',
        COUNT(*)
    FROM processed.team_game_efficiency
    WHERE game_id IS NULL
       OR team IS NULL
       OR opponent IS NULL
       OR points_scored IS NULL
       OR points_allowed IS NULL
       OR point_differential IS NULL
       OR offensive_plays IS NULL
       OR offensive_epa_per_play IS NULL
       OR defensive_epa_allowed_per_play IS NULL

    UNION ALL

    -- -------------------------------------------------
    -- Counts must be logically valid
    -- -------------------------------------------------

    SELECT
        'invalid_play_count',
        COUNT(*)
    FROM processed.team_game_efficiency
    WHERE offensive_plays <= 0
       OR dropbacks < 0
       OR designed_rushes < 0
       OR competitive_plays < 0
       OR early_down_plays < 0
       OR red_zone_plays < 0
       OR dropbacks + designed_rushes
          <> offensive_plays
       OR competitive_plays > offensive_plays
       OR early_down_plays > offensive_plays
       OR red_zone_plays > offensive_plays

    UNION ALL

    -- -------------------------------------------------
    -- Rates must remain between zero and one
    -- -------------------------------------------------

    SELECT
        'invalid_rate',
        COUNT(*)
    FROM processed.team_game_efficiency
    WHERE success_rate NOT BETWEEN 0 AND 1
       OR dropback_success_rate NOT BETWEEN 0 AND 1
       OR designed_rush_success_rate NOT BETWEEN 0 AND 1
       OR explosive_play_rate NOT BETWEEN 0 AND 1
       OR sack_rate NOT BETWEEN 0 AND 1
       OR turnover_rate NOT BETWEEN 0 AND 1
       OR defensive_success_rate_allowed NOT BETWEEN 0 AND 1
       OR explosive_play_rate_allowed NOT BETWEEN 0 AND 1
       OR sack_rate_generated NOT BETWEEN 0 AND 1
       OR turnover_rate_generated NOT BETWEEN 0 AND 1

    UNION ALL

    -- -------------------------------------------------
    -- Score-derived fields must be consistent
    -- -------------------------------------------------

    SELECT
        'invalid_game_result',
        COUNT(*)
    FROM processed.team_game_efficiency
    WHERE point_differential
          <> points_scored - points_allowed
       OR (
            is_tie
            AND point_differential <> 0
       )
       OR (
            NOT is_tie
            AND point_differential = 0
       )
       OR (
            point_differential > 0
            AND COALESCE(team_win, 0) <> 1
       )
       OR (
            point_differential < 0
            AND COALESCE(team_win, 0) <> 0
       )

    UNION ALL

    -- -------------------------------------------------
    -- Both team rows must contain mirrored scores
    -- -------------------------------------------------

    SELECT
        'inconsistent_opponent_score',
        COUNT(*)
    FROM processed.team_game_efficiency AS team
    INNER JOIN processed.team_game_efficiency AS opponent
        ON team.game_id = opponent.game_id
       AND team.team = opponent.opponent
       AND team.opponent = opponent.team
    WHERE team.points_scored
          <> opponent.points_allowed
       OR team.points_allowed
          <> opponent.points_scored
       OR team.point_differential
          <> -opponent.point_differential

    UNION ALL

    -- -------------------------------------------------
    -- Defensive metrics must mirror opponent offense
    -- -------------------------------------------------

    SELECT
        'inconsistent_defensive_metric',
        COUNT(*)
    FROM processed.team_game_efficiency AS team
    INNER JOIN processed.team_game_efficiency AS opponent
        ON team.game_id = opponent.game_id
       AND team.team = opponent.opponent
       AND team.opponent = opponent.team
    WHERE ABS(
            team.defensive_epa_allowed_per_play
            - opponent.offensive_epa_per_play
          ) > 0.000000001
       OR ABS(
            team.defensive_success_rate_allowed
            - opponent.success_rate
          ) > 0.000000001
       OR ABS(
            team.explosive_play_rate_allowed
            - opponent.explosive_play_rate
          ) > 0.000000001
       OR ABS(
            team.sack_rate_generated
            - opponent.sack_rate
          ) > 0.000000001
       OR ABS(
            team.turnover_rate_generated
            - opponent.turnover_rate
          ) > 0.000000001

    UNION ALL

    -- -------------------------------------------------
    -- Every target game must exist in the schedule
    -- -------------------------------------------------

    SELECT
        'orphan_schedule_game',
        COUNT(*)
    FROM processed.team_game_efficiency AS efficiency
    LEFT JOIN processed.schedule AS schedule
        ON efficiency.game_id = schedule.game_id
    WHERE schedule.game_id IS NULL
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