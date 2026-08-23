"""
NFL Analytics Platform
Current Prediction Data Science View

Purpose:
    Create and validate a dashboard-ready technical view
    of Elo, logistic, blend and exact feature impacts.

Author:
    Ferenc Kaizer

Version:
    0.1.0
"""

import duckdb


TARGET_SCHEMA = "analytics"

TARGET_VIEW = (
    "current_game_prediction_data_science_view"
)

TARGET_FULL_NAME = (
    f"{TARGET_SCHEMA}.{TARGET_VIEW}"
)

PREDICTION_TABLE = (
    "analytics.current_game_predictions"
)

CONTRIBUTION_TABLE = (
    "analytics.current_game_logistic_feature_contributions"
)


def create_current_prediction_data_science_view(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Create the technical prediction explanation view."""

    connection.execute(
        f"""
        CREATE OR REPLACE VIEW {TARGET_FULL_NAME} AS

        SELECT
            predictions.game_id,
            predictions.season,
            predictions.game_type,
            predictions.week,
            predictions.gameday,
            predictions.gametime,
            predictions.home_team,
            predictions.away_team,
            predictions.model_name,
            predictions.model_version,
            predictions.prediction_generated_at,
            predictions.prediction_mode,
            predictions.prediction_mode_reason,
            predictions.predicted_winner AS favorite,

            CASE
                WHEN predictions.predicted_winner
                        = predictions.home_team
                    THEN predictions.away_team
                ELSE predictions.home_team
            END AS underdog,

            GREATEST(
                predictions.home_win_probability,
                predictions.away_win_probability
            ) AS favorite_win_probability,

            predictions.home_win_probability,
            predictions.away_win_probability,
            predictions.published_nfelo_home_probability,
            predictions.primary_logistic_home_win_probability,
            predictions.fallback_logistic_home_win_probability,
            predictions.applied_primary_logistic_weight,
            predictions.applied_published_nfelo_weight,

            predictions.home_win_probability
                - predictions.published_nfelo_home_probability
                AS production_probability_adjustment_from_published_nfelo,

            predictions.has_complete_injury_data,
            predictions.both_listed_qb_ratings_available,
            predictions.has_complete_production_features,

            contributions.feature_name,
            contributions.raw_feature_value,
            contributions.standardized_feature_value,
            contributions.coefficient,
            contributions.log_odds_contribution,
            contributions.absolute_log_odds_contribution,
            contributions.contribution_rank,
            contributions.logistic_intercept,
            contributions.logistic_total_log_odds,
            contributions
                .logistic_reconstructed_home_win_probability,

            CASE
                WHEN contributions.feature_name IS NULL
                    THEN NULL
                WHEN contributions.log_odds_contribution > 0.0
                    THEN 'supports_home'
                ELSE 'supports_away'
            END AS feature_effect_on_home_probability,

            CASE
                WHEN contributions.feature_name IS NULL
                    THEN NULL

                WHEN (
                        contributions.log_odds_contribution > 0.0
                        AND predictions.predicted_winner
                            = predictions.home_team
                     )
                  OR (
                        contributions.log_odds_contribution <= 0.0
                        AND predictions.predicted_winner
                            = predictions.away_team
                     )
                    THEN 'supports_favorite'

                ELSE 'opposes_favorite'
            END AS feature_effect_on_favorite

        FROM {PREDICTION_TABLE}
            AS predictions

        LEFT JOIN {CONTRIBUTION_TABLE}
            AS contributions
            ON predictions.game_id
                = contributions.game_id
        """
    )


def validate_current_prediction_data_science_view(
    connection: duckdb.DuckDBPyConnection,
    expected_prediction_count: int,
    expected_feature_count: int,
) -> None:
    """Validate the technical prediction view."""

    distinct_game_count = connection.execute(
        f"""
        SELECT COUNT(DISTINCT game_id)
        FROM {TARGET_FULL_NAME}
        """
    ).fetchone()[0]

    if distinct_game_count != expected_prediction_count:
        raise RuntimeError(
            "Data Science view game count does not "
            f"match: expected {expected_prediction_count}, "
            f"found {distinct_game_count}."
        )

    invalid_group_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM (
            SELECT
                game_id,
                prediction_mode,
                COUNT(*) AS row_count,
                COUNT(feature_name)
                    AS contribution_count
            FROM {TARGET_FULL_NAME}
            GROUP BY
                game_id,
                prediction_mode
            HAVING (
                    prediction_mode = 'EXTERNAL_NFELO_BLEND'
                    AND (
                        row_count <> {expected_feature_count}
                        OR contribution_count
                            <> {expected_feature_count}
                    )
                  )
                OR (
                    prediction_mode = 'EXTERNAL_ELO_QB_FALLBACK'
                    AND (
                        row_count <> 1
                        OR contribution_count <> 0
                    )
                  )
                OR prediction_mode NOT IN (
                    'EXTERNAL_NFELO_BLEND',
                    'EXTERNAL_ELO_QB_FALLBACK'
                )
        )
        """
    ).fetchone()[0]

    if invalid_group_count > 0:
        raise RuntimeError(
            "Invalid Data Science view routing groups "
            "found."
        )

    duplicate_feature_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM (
            SELECT
                game_id,
                feature_name
            FROM {TARGET_FULL_NAME}
            WHERE feature_name IS NOT NULL
            GROUP BY
                game_id,
                feature_name
            HAVING COUNT(*) > 1
        )
        """
    ).fetchone()[0]

    if duplicate_feature_count > 0:
        raise RuntimeError(
            "Duplicate Data Science feature rows found."
        )

    invalid_probability_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE home_win_probability
                NOT BETWEEN 0.0 AND 1.0
           OR away_win_probability
                NOT BETWEEN 0.0 AND 1.0
           OR ABS(
                home_win_probability
                + away_win_probability
                - 1.0
              ) > 0.000001
           OR ABS(
                favorite_win_probability
                - GREATEST(
                    home_win_probability,
                    away_win_probability
                )
              ) > 0.000001
           OR ABS(
                production_probability_adjustment_from_published_nfelo
                -
                (
                    home_win_probability
                    - published_nfelo_home_probability
                )
              ) > 0.000001
           OR applied_primary_logistic_weight
                NOT BETWEEN 0.0 AND 1.0
           OR applied_published_nfelo_weight
                NOT BETWEEN 0.0 AND 1.0
           OR (
                prediction_mode = 'EXTERNAL_NFELO_BLEND'
                AND (
                    published_nfelo_home_probability
                        NOT BETWEEN 0.0 AND 1.0
                    OR primary_logistic_home_win_probability
                        NOT BETWEEN 0.0 AND 1.0
                    OR fallback_logistic_home_win_probability
                        IS NOT NULL
                    OR ABS(
                        applied_primary_logistic_weight
                        + applied_published_nfelo_weight
                        - 1.0
                      ) > 0.000001
                    OR ABS(
                        home_win_probability
                        -
                        (
                            applied_primary_logistic_weight
                            * primary_logistic_home_win_probability
                            + applied_published_nfelo_weight
                            * published_nfelo_home_probability
                        )
                    ) > 0.000001
                )
              )
           OR (
                prediction_mode = 'EXTERNAL_ELO_QB_FALLBACK'
                AND (
                    primary_logistic_home_win_probability IS NOT NULL
                    OR fallback_logistic_home_win_probability
                        NOT BETWEEN 0.0 AND 1.0
                    OR applied_primary_logistic_weight <> 0.0
                    OR applied_published_nfelo_weight <> 0.0
                    OR ABS(
                        home_win_probability
                        - fallback_logistic_home_win_probability
                    ) > 0.000001
                )
              )
        """
    ).fetchone()[0]

    if invalid_probability_count > 0:
        raise RuntimeError(
            "Invalid Data Science view probabilities "
            "found."
        )

    invalid_contribution_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE feature_name IS NOT NULL
          AND (
                raw_feature_value IS NULL
                OR standardized_feature_value IS NULL
                OR coefficient IS NULL
                OR log_odds_contribution IS NULL
                OR absolute_log_odds_contribution IS NULL
                OR contribution_rank IS NULL
                OR logistic_intercept IS NULL
                OR logistic_total_log_odds IS NULL
                OR logistic_reconstructed_home_win_probability
                    NOT BETWEEN 0.0 AND 1.0
                OR ABS(
                    log_odds_contribution
                    -
                    (
                        standardized_feature_value
                        * coefficient
                    )
                ) > 0.000000001
                OR ABS(
                    absolute_log_odds_contribution
                    - ABS(log_odds_contribution)
                ) > 0.000000001
                OR feature_effect_on_home_probability
                    NOT IN (
                        'supports_home',
                        'supports_away'
                    )
                OR feature_effect_on_favorite
                    NOT IN (
                        'supports_favorite',
                        'opposes_favorite'
                    )
              )
        """
    ).fetchone()[0]

    if invalid_contribution_count > 0:
        raise RuntimeError(
            "Invalid Data Science contribution rows "
            "found."
        )

    invalid_fallback_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE prediction_mode = 'EXTERNAL_ELO_QB_FALLBACK'
          AND (
                feature_name IS NOT NULL
                OR raw_feature_value IS NOT NULL
                OR standardized_feature_value IS NOT NULL
                OR coefficient IS NOT NULL
                OR log_odds_contribution IS NOT NULL
                OR contribution_rank IS NOT NULL
                OR feature_effect_on_home_probability
                    IS NOT NULL
                OR feature_effect_on_favorite IS NOT NULL
              )
        """
    ).fetchone()[0]

    if invalid_fallback_count > 0:
        raise RuntimeError(
            "External fallback has unexpected Data Science "
            "feature values."
        )

    invalid_reconstruction_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM (
            SELECT
                game_id,
                MIN(logistic_intercept)
                    AS logistic_intercept,
                MIN(logistic_total_log_odds)
                    AS logistic_total_log_odds,
                MIN(
                    logistic_reconstructed_home_win_probability
                ) AS reconstructed_probability,
                MIN(primary_logistic_home_win_probability)
                    AS logistic_probability,
                SUM(log_odds_contribution)
                    AS contribution_sum
            FROM {TARGET_FULL_NAME}
            WHERE prediction_mode = 'EXTERNAL_NFELO_BLEND'
            GROUP BY game_id
        )
        WHERE ABS(
                logistic_total_log_odds
                -
                (
                    logistic_intercept
                    + contribution_sum
                )
              ) > 0.000000001
           OR ABS(
                reconstructed_probability
                -
                (
                    1.0
                    /
                    (
                        1.0
                        + EXP(-logistic_total_log_odds)
                    )
                )
              ) > 0.000000001
           OR ABS(
                reconstructed_probability
                - logistic_probability
              ) > 0.000000001
        """
    ).fetchone()[0]

    if invalid_reconstruction_count > 0:
        raise RuntimeError(
            "Invalid Data Science logistic probability "
            "reconstruction found."
        )
