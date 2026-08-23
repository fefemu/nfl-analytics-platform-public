-- =========================================================
-- NFL Analytics Platform
-- Model Governance and Blend Reporting Quality Checks
-- =========================================================

WITH quality_checks AS (

    SELECT
        'governance_scorecard_row_count_is_5'
            AS check_name,

        CASE
            WHEN (
                SELECT COUNT(*)
                FROM analytics.model_governance_scorecard
            ) = 5
            THEN 0
            ELSE 1
        END AS issue_count

    UNION ALL

    SELECT
        'governance_scorecard_duplicate_models',
        COUNT(*)

    FROM (
        SELECT
            model_name,
            model_version
        FROM analytics.model_governance_scorecard
        GROUP BY
            model_name,
            model_version
        HAVING COUNT(*) > 1
    )

    UNION ALL

    SELECT
        'governance_scorecard_invalid_coverage',
        COUNT(*)

    FROM analytics.model_governance_scorecard

    WHERE season_count != 6
       OR game_count != 1254

    UNION ALL

    SELECT
        'governance_scorecard_invalid_metrics',
        COUNT(*)

    FROM analytics.model_governance_scorecard

    WHERE accuracy NOT BETWEEN 0.0 AND 1.0
       OR brier_score NOT BETWEEN 0.0 AND 1.0
       OR log_loss < 0.0
       OR worst_season_brier NOT BETWEEN 0.0 AND 1.0
       OR worst_season_log_loss < 0.0
       OR brier_season_std < 0.0
       OR log_loss_season_std < 0.0

    UNION ALL

    SELECT
        'governance_season_result_row_count_is_30',

        CASE
            WHEN (
                SELECT COUNT(*)
                FROM analytics.model_governance_season_results
            ) = 30
            THEN 0
            ELSE 1
        END

    UNION ALL

    SELECT
        'governance_season_coverage_is_2020_2025',

        CASE
            WHEN (
                SELECT COUNT(DISTINCT validation_season)
                FROM analytics.model_governance_season_results
            ) = 6
             AND (
                SELECT MIN(validation_season)
                FROM analytics.model_governance_season_results
            ) = 2020
             AND (
                SELECT MAX(validation_season)
                FROM analytics.model_governance_season_results
            ) = 2025
            THEN 0
            ELSE 1
        END

    UNION ALL

    SELECT
        'governance_season_has_five_models_each',
        COUNT(*)

    FROM (
        SELECT
            validation_season
        FROM analytics.model_governance_season_results
        GROUP BY validation_season
        HAVING COUNT(*) != 5
    )

    UNION ALL

    SELECT
        'governance_season_duplicate_model_results',
        COUNT(*)

    FROM (
        SELECT
            validation_season,
            model_name,
            model_version
        FROM analytics.model_governance_season_results
        GROUP BY
            validation_season,
            model_name,
            model_version
        HAVING COUNT(*) > 1
    )

    UNION ALL

    SELECT
        'governance_season_invalid_metrics',
        COUNT(*)

    FROM analytics.model_governance_season_results

    WHERE game_count <= 0
       OR accuracy NOT BETWEEN 0.0 AND 1.0
       OR brier_score NOT BETWEEN 0.0 AND 1.0
       OR log_loss < 0.0

    UNION ALL

    SELECT
        'blend_weight_grid_row_count_is_42',

        CASE
            WHEN (
                SELECT COUNT(*)
                FROM analytics.model_blend_weight_grid
            ) = 42
            THEN 0
            ELSE 1
        END

    UNION ALL

    SELECT
        'blend_weight_grid_has_two_scopes',

        CASE
            WHEN (
                SELECT COUNT(DISTINCT selection_scope)
                FROM analytics.model_blend_weight_grid
            ) = 2
             AND (
                SELECT COUNT(*)
                FROM analytics.model_blend_weight_grid
                WHERE selection_scope = 'audit_2020_2024'
            ) = 21
             AND (
                SELECT COUNT(*)
                FROM analytics.model_blend_weight_grid
                WHERE selection_scope
                    = 'production_2020_2025'
            ) = 21
            THEN 0
            ELSE 1
        END

    UNION ALL

    SELECT
        'blend_weight_grid_invalid_weights',
        COUNT(*)

    FROM analytics.model_blend_weight_grid

    WHERE injury_weight NOT BETWEEN 0.0 AND 1.0
       OR elo_weight NOT BETWEEN 0.0 AND 1.0
       OR ABS(
            injury_weight
            + elo_weight
            - 1.0
          ) > 0.000000001

    UNION ALL

    SELECT
        'blend_weight_grid_duplicate_scope_weights',
        COUNT(*)

    FROM (
        SELECT
            selection_scope,
            injury_weight
        FROM analytics.model_blend_weight_grid
        GROUP BY
            selection_scope,
            injury_weight
        HAVING COUNT(*) > 1
    )

    UNION ALL

    SELECT
        'blend_scorecard_row_count_is_9',

        CASE
            WHEN (
                SELECT COUNT(*)
                FROM analytics.model_blend_scorecard
            ) = 9
            THEN 0
            ELSE 1
        END

    UNION ALL

    SELECT
        'blend_scorecard_has_three_models_per_period',
        COUNT(*)

    FROM (
        SELECT
            evaluation_period
        FROM analytics.model_blend_scorecard
        GROUP BY evaluation_period
        HAVING COUNT(*) != 3
    )

    UNION ALL

    SELECT
        'blend_scorecard_invalid_metrics',
        COUNT(*)

    FROM analytics.model_blend_scorecard

    WHERE game_count <= 0
       OR accuracy NOT BETWEEN 0.0 AND 1.0
       OR brier_score NOT BETWEEN 0.0 AND 1.0
       OR log_loss < 0.0
       OR injury_weight NOT BETWEEN 0.0 AND 1.0
       OR elo_weight NOT BETWEEN 0.0 AND 1.0
       OR ABS(
            injury_weight
            + elo_weight
            - 1.0
          ) > 0.000000001

    UNION ALL

    SELECT
        'production_registry_row_count_is_1',

        CASE
            WHEN (
                SELECT COUNT(*)
                FROM analytics.production_model_registry
            ) = 1
            THEN 0
            ELSE 1
        END

    UNION ALL

    SELECT
        'production_registry_matches_selected_blend',
        COUNT(*)

    FROM analytics.production_model_registry

    WHERE model_name
            != 'elo_injury_logistic_blend'
       OR model_version != '0.2.0'
       OR deployment_status
            != 'selected_for_2026_forward_test'
       OR logistic_component_name
            != 'logistic_elo_qb_unit_burdens'
       OR logistic_regularization_c
            != 0.1
       OR ABS(
            logistic_weight - 0.70
          ) > 0.000000001
       OR ABS(
            elo_weight - 0.30
          ) > 0.000000001
       OR requires_complete_injury_data
            IS DISTINCT FROM TRUE
       OR incomplete_injury_fallback_model
            != 'elo'
       OR forward_test_season != 2026

    UNION ALL

    SELECT
        'production_registry_weight_matches_best_grid',

        CASE
            WHEN (
                SELECT logistic_weight
                FROM analytics.production_model_registry
            ) = (
                SELECT injury_weight
                FROM analytics.model_blend_weight_grid
                WHERE selection_scope
                    = 'production_2020_2025'
                ORDER BY
                    brier_score,
                    log_loss,
                    injury_weight
                LIMIT 1
            )
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