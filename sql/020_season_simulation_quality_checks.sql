-- =========================================================
-- NFL Analytics Platform
-- Current Season Simulation Quality Checks
-- =========================================================

WITH quality_checks AS (

    SELECT
        'summary_team_count_is_32' AS check_name,

        CASE
            WHEN (
                SELECT COUNT(*)
                FROM analytics.current_season_simulation_summary
            ) = 32
            THEN 0
            ELSE 1
        END AS issue_count

    UNION ALL

    SELECT
        'duplicate_summary_team',
        COUNT(*)

    FROM (
        SELECT
            season,
            team
        FROM analytics.current_season_simulation_summary
        GROUP BY
            season,
            team
        HAVING COUNT(*) > 1
    )

    UNION ALL

    SELECT
        'summary_uses_latest_prediction_season',
        COUNT(*)

    FROM analytics.current_season_simulation_summary

    WHERE season IS DISTINCT FROM (
        SELECT MAX(season)
        FROM analytics.current_game_predictions
        WHERE game_type = 'REG'
    )

    UNION ALL

    SELECT
        'invalid_team_games',
        COUNT(*)

    FROM analytics.current_season_simulation_summary

    WHERE games <> 17
       OR expected_wins < 0.0
       OR expected_losses < 0.0
       OR expected_ties < 0
       OR expected_wins > games
       OR expected_losses > games
       OR expected_ties > games

    UNION ALL

    SELECT
        'expected_record_does_not_sum_to_games',
        COUNT(*)

    FROM analytics.current_season_simulation_summary

    WHERE ABS(
        expected_wins
        + expected_losses
        + expected_ties
        - games
    ) > 0.000001

    UNION ALL

    SELECT
        'invalid_win_summary_statistics',
        COUNT(*)

    FROM analytics.current_season_simulation_summary

    WHERE median_wins NOT BETWEEN 0.0 AND games
       OR p10_wins NOT BETWEEN 0.0 AND games
       OR p90_wins NOT BETWEEN 0.0 AND games
       OR p10_wins > median_wins
       OR median_wins > p90_wins
       OR most_likely_wins NOT BETWEEN 0 AND games
       OR minimum_wins NOT BETWEEN 0 AND games
       OR maximum_wins NOT BETWEEN 0 AND games
       OR minimum_wins > maximum_wins
       OR expected_wins < minimum_wins
       OR expected_wins > maximum_wins
       OR expected_final_elo IS NULL

    UNION ALL

    SELECT
        'invalid_summary_metadata',
        COUNT(*)

    FROM analytics.current_season_simulation_summary

    WHERE simulation_count <= 0
       OR random_seed IS NULL
       OR model_name IS DISTINCT FROM 'elo'
       OR model_version IS DISTINCT FROM '1.0.0'
       OR simulation_generated_at IS NULL

    UNION ALL

    SELECT
        'distribution_contains_unknown_team',
        COUNT(*)

    FROM analytics.current_season_win_distribution
        AS distribution

    LEFT JOIN analytics.current_season_simulation_summary
        AS summary
        ON distribution.season = summary.season
       AND distribution.team = summary.team

    WHERE summary.team IS NULL

    UNION ALL

    SELECT
        'summary_team_missing_distribution',
        COUNT(*)

    FROM analytics.current_season_simulation_summary
        AS summary

    LEFT JOIN (
        SELECT DISTINCT
            season,
            team
        FROM analytics.current_season_win_distribution
    ) AS distribution
        ON summary.season = distribution.season
       AND summary.team = distribution.team

    WHERE distribution.team IS NULL

    UNION ALL

    SELECT
        'duplicate_team_win_total',
        COUNT(*)

    FROM (
        SELECT
            season,
            team,
            wins
        FROM analytics.current_season_win_distribution
        GROUP BY
            season,
            team,
            wins
        HAVING COUNT(*) > 1
    )

    UNION ALL

    SELECT
        'invalid_distribution_values',
        COUNT(*)

    FROM analytics.current_season_win_distribution

    WHERE wins NOT BETWEEN 0 AND 17
       OR simulation_count < 0
       OR total_simulations <= 0
       OR simulation_count > total_simulations
       OR probability NOT BETWEEN 0.0 AND 1.0
       OR ABS(
            probability
            -
            (
                CAST(simulation_count AS DOUBLE)
                / total_simulations
            )
          ) > 0.000001

    UNION ALL

    SELECT
        'distribution_probability_not_one',
        COUNT(*)

    FROM (
        SELECT
            season,
            team,
            SUM(probability) AS probability_sum
        FROM analytics.current_season_win_distribution
        GROUP BY
            season,
            team
        HAVING ABS(
            probability_sum - 1.0
        ) > 0.000001
    )

    UNION ALL

    SELECT
        'distribution_count_not_simulation_count',
        COUNT(*)

    FROM (
        SELECT
            season,
            team,
            MAX(total_simulations)
                AS total_simulations,
            SUM(simulation_count)
                AS distributed_simulations
        FROM analytics.current_season_win_distribution
        GROUP BY
            season,
            team
        HAVING distributed_simulations
            <> total_simulations
    )

    UNION ALL

    SELECT
        'expected_wins_do_not_match_distribution',
        COUNT(*)

    FROM analytics.current_season_simulation_summary
        AS summary

    INNER JOIN (
        SELECT
            season,
            team,
            SUM(
                wins * probability
            ) AS distribution_expected_wins
        FROM analytics.current_season_win_distribution
        GROUP BY
            season,
            team
    ) AS distribution
        ON summary.season = distribution.season
       AND summary.team = distribution.team

    WHERE ABS(
        summary.expected_wins
        - distribution.distribution_expected_wins
    ) > 0.000001

    UNION ALL

    SELECT
        'distribution_outside_summary_range',
        COUNT(*)

    FROM analytics.current_season_win_distribution
        AS distribution

    INNER JOIN analytics.current_season_simulation_summary
        AS summary
        ON distribution.season = summary.season
       AND distribution.team = summary.team

    WHERE distribution.wins
            < summary.minimum_wins
       OR distribution.wins
            > summary.maximum_wins

    UNION ALL

    SELECT
        'simulation_metadata_mismatch',
        COUNT(*)

    FROM analytics.current_season_win_distribution
        AS distribution

    INNER JOIN analytics.current_season_simulation_summary
        AS summary
        ON distribution.season = summary.season
       AND distribution.team = summary.team

    WHERE distribution.total_simulations
            IS DISTINCT FROM
                summary.simulation_count

       OR distribution.random_seed
            IS DISTINCT FROM
                summary.random_seed

       OR distribution.model_name
            IS DISTINCT FROM
                summary.model_name

       OR distribution.model_version
            IS DISTINCT FROM
                summary.model_version

       OR distribution.simulation_generated_at
            IS DISTINCT FROM
                summary.simulation_generated_at

    UNION ALL

    SELECT
        'league_record_mass_is_invalid',

        CASE
            WHEN ABS(
                (
                    SELECT SUM(
                        expected_wins
                        + expected_losses
                        + expected_ties
                    )
                    FROM analytics.current_season_simulation_summary
                )
                - 544.0
            ) <= 0.000001
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