-- =========================================================
-- NFL Analytics Platform
-- Frozen Versus Dynamic Elo Benchmark Quality Checks
-- =========================================================

WITH current_season AS (

    SELECT MAX(season) AS season
    FROM analytics.current_game_predictions
    WHERE game_type = 'REG'
),

initial_rating_rows AS (

    SELECT
        home_team AS team,
        home_rating_pregame AS initial_elo

    FROM analytics.current_game_predictions

    WHERE season = (
            SELECT season
            FROM current_season
          )
      AND game_type = 'REG'

    UNION ALL

    SELECT
        away_team AS team,
        away_rating_pregame AS initial_elo

    FROM analytics.current_game_predictions

    WHERE season = (
            SELECT season
            FROM current_season
          )
      AND game_type = 'REG'
),

initial_ratings AS (

    SELECT
        team,
        MIN(initial_elo) AS minimum_initial_elo,
        MAX(initial_elo) AS maximum_initial_elo

    FROM initial_rating_rows

    GROUP BY team
),

quality_checks AS (

    SELECT
        'single_summary_row' AS check_name,
        ABS(
            (
                SELECT COUNT(*)
                FROM analytics
                    .current_season_elo_benchmark_summary
            )
            - 1
        ) AS issue_count

    UNION ALL

    SELECT
        'team_coverage_matches_production_simulation',
        ABS(
            (
                SELECT COUNT(*)
                FROM analytics
                    .current_season_elo_benchmark_team_comparison
            )
            -
            (
                SELECT COUNT(*)
                FROM analytics
                    .current_season_simulation_summary
            )
        )

    UNION ALL

    SELECT
        'duplicate_team',
        COUNT(*)

    FROM (
        SELECT team
        FROM analytics
            .current_season_elo_benchmark_team_comparison
        GROUP BY team
        HAVING COUNT(*) > 1
    )

    UNION ALL

    SELECT
        'invalid_common_random_number_metadata',
        COUNT(*)

    FROM analytics
        .current_season_elo_benchmark_team_comparison
        AS comparison

    CROSS JOIN analytics
        .current_season_elo_benchmark_summary
        AS summary

    WHERE comparison.season
            IS DISTINCT FROM summary.season
       OR comparison.simulation_count
            IS DISTINCT FROM summary.simulation_count
       OR comparison.random_seed
            IS DISTINCT FROM summary.random_seed
       OR comparison.comparison_method
            IS DISTINCT FROM 'common_random_numbers'
       OR summary.comparison_method
            IS DISTINCT FROM 'common_random_numbers'
       OR summary.dynamic_mode
            IS DISTINCT FROM 'DYNAMIC_ELO'
       OR summary.frozen_mode
            IS DISTINCT FROM 'FROZEN_ELO'
       OR comparison.benchmark_generated_at
            IS DISTINCT FROM
                summary.benchmark_generated_at

    UNION ALL

    SELECT
        'production_dynamic_result_mismatch',
        COUNT(*)

    FROM analytics
        .current_season_elo_benchmark_team_comparison
        AS comparison

    INNER JOIN analytics
        .current_season_simulation_summary
        AS production
        ON comparison.team = production.team

    WHERE comparison.season
            IS DISTINCT FROM production.season
       OR comparison.simulation_count
            IS DISTINCT FROM
                production.simulation_count
       OR comparison.random_seed
            IS DISTINCT FROM production.random_seed
       OR ABS(
            comparison.dynamic_expected_wins
            - production.expected_wins
          ) > 0.000000001
       OR ABS(
            comparison.dynamic_expected_final_elo
            - production.expected_final_elo
          ) > 0.000000001

    UNION ALL

    SELECT
        'invalid_comparison_math',
        COUNT(*)

    FROM analytics
        .current_season_elo_benchmark_team_comparison

    WHERE ABS(
            expected_wins_delta
            -
            (
                dynamic_expected_wins
                - frozen_expected_wins
            )
          ) > 0.000000001
       OR ABS(
            absolute_expected_wins_delta
            - ABS(expected_wins_delta)
          ) > 0.000000001
       OR ABS(
            dynamic_win_range
            -
            (
                dynamic_p90_wins
                - dynamic_p10_wins
            )
          ) > 0.000000001
       OR ABS(
            frozen_win_range
            -
            (
                frozen_p90_wins
                - frozen_p10_wins
            )
          ) > 0.000000001
       OR ABS(
            win_range_delta
            -
            (
                dynamic_win_range
                - frozen_win_range
            )
          ) > 0.000000001
       OR ABS(
            expected_final_elo_delta
            -
            (
                dynamic_expected_final_elo
                - frozen_expected_final_elo
            )
          ) > 0.000000001

    UNION ALL

    SELECT
        'invalid_tail_probabilities',
        COUNT(*)

    FROM analytics
        .current_season_elo_benchmark_team_comparison

    WHERE dynamic_probability_14_plus_wins
            NOT BETWEEN 0.0 AND 1.0
       OR frozen_probability_14_plus_wins
            NOT BETWEEN 0.0 AND 1.0
       OR dynamic_probability_3_or_fewer_wins
            NOT BETWEEN 0.0 AND 1.0
       OR frozen_probability_3_or_fewer_wins
            NOT BETWEEN 0.0 AND 1.0
       OR dynamic_probability_2_or_fewer_wins
            NOT BETWEEN 0.0 AND 1.0
       OR frozen_probability_2_or_fewer_wins
            NOT BETWEEN 0.0 AND 1.0
       OR dynamic_probability_2_or_fewer_wins
            > dynamic_probability_3_or_fewer_wins
       OR frozen_probability_2_or_fewer_wins
            > frozen_probability_3_or_fewer_wins
       OR ABS(
            probability_14_plus_wins_delta
            -
            (
                dynamic_probability_14_plus_wins
                - frozen_probability_14_plus_wins
            )
          ) > 0.000000001
       OR ABS(
            probability_3_or_fewer_wins_delta
            -
            (
                dynamic_probability_3_or_fewer_wins
                - frozen_probability_3_or_fewer_wins
            )
          ) > 0.000000001
       OR ABS(
            probability_2_or_fewer_wins_delta
            -
            (
                dynamic_probability_2_or_fewer_wins
                - frozen_probability_2_or_fewer_wins
            )
          ) > 0.000000001

    UNION ALL

    SELECT
        'invalid_distribution_distance',
        COUNT(*)

    FROM analytics
        .current_season_elo_benchmark_team_comparison

    WHERE win_distribution_total_variation
            NOT BETWEEN 0.0 AND 1.0

    UNION ALL

    SELECT
        'total_expected_wins_not_preserved',
        CASE
            WHEN ABS(
                (
                    SELECT SUM(expected_wins_delta)
                    FROM analytics
                        .current_season_elo_benchmark_team_comparison
                )
            ) > 0.000000001
                THEN 1
            ELSE 0
        END

    UNION ALL

    SELECT
        'frozen_elo_does_not_match_initial_rating',
        COUNT(*)

    FROM analytics
        .current_season_elo_benchmark_team_comparison
        AS comparison

    INNER JOIN initial_ratings
        AS initial_ratings
        ON comparison.team = initial_ratings.team

    WHERE ABS(
            initial_ratings.minimum_initial_elo
            - initial_ratings.maximum_initial_elo
          ) > 0.000000001
       OR ABS(
            comparison.frozen_expected_final_elo
            - initial_ratings.minimum_initial_elo
          ) > 0.000000001

    UNION ALL

    SELECT
        'benchmark_summary_aggregate_mismatch',
        COUNT(*)

    FROM analytics
        .current_season_elo_benchmark_summary
        AS summary

    CROSS JOIN (
        SELECT
            COUNT(*) AS team_count,
            AVG(absolute_expected_wins_delta)
                AS mean_absolute_delta,
            MAX(absolute_expected_wins_delta)
                AS maximum_absolute_delta,
            AVG(win_distribution_total_variation)
                AS mean_total_variation,
            MAX(win_distribution_total_variation)
                AS maximum_total_variation

        FROM analytics
            .current_season_elo_benchmark_team_comparison
    ) AS calculated

    WHERE summary.team_count
            IS DISTINCT FROM calculated.team_count
       OR ABS(
            summary.mean_absolute_expected_wins_delta
            - calculated.mean_absolute_delta
          ) > 0.000000001
       OR ABS(
            summary.maximum_absolute_expected_wins_delta
            - calculated.maximum_absolute_delta
          ) > 0.000000001
       OR ABS(
            summary.mean_win_distribution_total_variation
            - calculated.mean_total_variation
          ) > 0.000000001
       OR ABS(
            summary.maximum_win_distribution_total_variation
            - calculated.maximum_total_variation
          ) > 0.000000001
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