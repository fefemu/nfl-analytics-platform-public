"""
NFL Analytics Platform
Current Logistic Feature Contributions

Purpose:
    Persist and validate exact game-feature logistic
    contributions for current production predictions.

Author:
    Ferenc Kaizer

Version:
    0.1.0
"""

import duckdb
import pandas as pd

from src.modeling.current_production_predictions import (
    CURRENT_LOGISTIC_CONTRIBUTION_COLUMNS,
)


TARGET_SCHEMA = "analytics"

TARGET_TABLE = (
    "current_game_logistic_feature_contributions"
)

TARGET_FULL_NAME = (
    f"{TARGET_SCHEMA}.{TARGET_TABLE}"
)


def create_current_logistic_feature_contributions_table(
    connection: duckdb.DuckDBPyConnection,
    contributions: pd.DataFrame,
) -> None:
    """Create the normalized contribution table."""

    missing_columns = sorted(
        set(
            CURRENT_LOGISTIC_CONTRIBUTION_COLUMNS
        )
        - set(contributions.columns)
    )

    if missing_columns:
        raise ValueError(
            "Current logistic feature contributions "
            "are missing columns: "
            + ", ".join(missing_columns)
        )

    connection.execute(
        f"CREATE SCHEMA IF NOT EXISTS {TARGET_SCHEMA}"
    )

    connection.execute(
        f"""
        CREATE OR REPLACE TABLE {TARGET_FULL_NAME} (
            game_id VARCHAR,
            feature_name VARCHAR,
            raw_feature_value DOUBLE,
            standardized_feature_value DOUBLE,
            coefficient DOUBLE,
            log_odds_contribution DOUBLE,
            absolute_log_odds_contribution DOUBLE,
            contribution_rank INTEGER,
            logistic_intercept DOUBLE,
            logistic_total_log_odds DOUBLE,
            logistic_reconstructed_home_win_probability DOUBLE,
            model_name VARCHAR,
            model_version VARCHAR,
            prediction_generated_at TIMESTAMP
        )
        """
    )

    if contributions.empty:
        return

    rows = [
        tuple(
            (
                None
                if pd.isna(
                    row[column_name]
                )
                else row[column_name]
            )
            for column_name
            in CURRENT_LOGISTIC_CONTRIBUTION_COLUMNS
        )
        for row in contributions.to_dict(
            orient="records"
        )
    ]

    placeholders = ", ".join(
        "?"
        for _ in (
            CURRENT_LOGISTIC_CONTRIBUTION_COLUMNS
        )
    )

    connection.executemany(
        f"""
        INSERT INTO {TARGET_FULL_NAME}
        VALUES ({placeholders})
        """,
        rows,
    )


def validate_current_logistic_feature_contributions_table(
    connection: duckdb.DuckDBPyConnection,
    expected_row_count: int,
    expected_feature_count: int,
) -> None:
    """Validate persisted logistic contributions."""

    actual_row_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        """
    ).fetchone()[0]

    if actual_row_count != expected_row_count:
        raise RuntimeError(
            "Current logistic contribution row count "
            f"does not match: expected "
            f"{expected_row_count}, found "
            f"{actual_row_count}."
        )

    if expected_row_count == 0:
        return

    duplicate_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM (
            SELECT
                game_id,
                feature_name
            FROM {TARGET_FULL_NAME}
            GROUP BY
                game_id,
                feature_name
            HAVING COUNT(*) > 1
        )
        """
    ).fetchone()[0]

    if duplicate_count > 0:
        raise RuntimeError(
            "Duplicate current logistic feature "
            "contributions found."
        )

    invalid_value_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE game_id IS NULL
           OR feature_name IS NULL
           OR raw_feature_value IS NULL
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
                - (
                    standardized_feature_value
                    * coefficient
                )
              ) > 0.000000001
           OR ABS(
                absolute_log_odds_contribution
                - ABS(log_odds_contribution)
              ) > 0.000000001
        """
    ).fetchone()[0]

    if invalid_value_count > 0:
        raise RuntimeError(
            "Invalid current logistic contribution "
            "mathematics found."
        )

    invalid_group_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM (
            SELECT
                game_id,
                COUNT(*) AS feature_count,
                COUNT(
                    DISTINCT feature_name
                ) AS distinct_feature_count,
                COUNT(
                    DISTINCT contribution_rank
                ) AS distinct_rank_count,
                MIN(contribution_rank) AS minimum_rank,
                MAX(contribution_rank) AS maximum_rank
            FROM {TARGET_FULL_NAME}
            GROUP BY game_id
            HAVING feature_count
                    <> {expected_feature_count}
                OR distinct_feature_count
                    <> {expected_feature_count}
                OR distinct_rank_count
                    <> {expected_feature_count}
                OR minimum_rank <> 1
                OR maximum_rank
                    <> {expected_feature_count}
        )
        """
    ).fetchone()[0]

    if invalid_group_count > 0:
        raise RuntimeError(
            "Invalid current logistic contribution "
            "feature groups or ranks found."
        )
    
    invalid_reconstruction_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM (
            SELECT
                game_id,
                MIN(logistic_intercept)
                    AS minimum_intercept,
                MAX(logistic_intercept)
                    AS maximum_intercept,
                MIN(logistic_total_log_odds)
                    AS minimum_total_log_odds,
                MAX(logistic_total_log_odds)
                    AS maximum_total_log_odds,
                MIN(
                    logistic_reconstructed_home_win_probability
                ) AS minimum_probability,
                MAX(
                    logistic_reconstructed_home_win_probability
                ) AS maximum_probability,
                SUM(log_odds_contribution)
                    AS contribution_sum
            FROM {TARGET_FULL_NAME}
            GROUP BY game_id
        )
        WHERE ABS(
                minimum_intercept
                - maximum_intercept
              ) > 0.000000001
           OR ABS(
                minimum_total_log_odds
                - maximum_total_log_odds
              ) > 0.000000001
           OR ABS(
                minimum_probability
                - maximum_probability
              ) > 0.000000001
           OR ABS(
                minimum_total_log_odds
                - (
                    minimum_intercept
                    + contribution_sum
                )
              ) > 0.000000001
           OR ABS(
                minimum_probability
                - (
                    1.0
                    /
                    (
                        1.0
                        + EXP(
                            -minimum_total_log_odds
                        )
                    )
                )
              ) > 0.000000001
        """
    ).fetchone()[0]

    if invalid_reconstruction_count > 0:
        raise RuntimeError(
            "Invalid current logistic probability "
            "reconstruction found."
        )

    invalid_rank_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME} AS contribution
        WHERE EXISTS (
            SELECT 1
            FROM {TARGET_FULL_NAME} AS stronger
            WHERE stronger.game_id
                    = contribution.game_id
              AND stronger.contribution_rank
                    < contribution.contribution_rank
              AND stronger
                    .absolute_log_odds_contribution
                    < contribution
                    .absolute_log_odds_contribution
                    - 0.000000001
        )
        """
    ).fetchone()[0]

    if invalid_rank_count > 0:
        raise RuntimeError(
            "Invalid current logistic contribution "
            "ranking found."
        )

    invalid_metadata_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE model_name IS NULL
           OR model_version IS NULL
           OR prediction_generated_at IS NULL
        """
    ).fetchone()[0]

    if invalid_metadata_count > 0:
        raise RuntimeError(
            "Invalid current logistic contribution "
            "metadata found."
        )