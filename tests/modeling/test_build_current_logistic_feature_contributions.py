"""Tests for persisted logistic feature contributions."""

from datetime import datetime

import duckdb
import pandas as pd
import pytest

from src.modeling.build_current_logistic_feature_contributions import (
    TARGET_FULL_NAME,
    create_current_logistic_feature_contributions_table,
    validate_current_logistic_feature_contributions_table,
)
from src.modeling.current_production_predictions import (
    CURRENT_LOGISTIC_CONTRIBUTION_COLUMNS,
)


@pytest.fixture
def connection() -> duckdb.DuckDBPyConnection:
    """Create an in-memory DuckDB connection."""

    database = duckdb.connect(":memory:")

    yield database

    database.close()


def create_contributions() -> pd.DataFrame:
    """Create one valid five-feature contribution group."""

    generated_at = datetime(
        2026,
        8,
        6,
        12,
        0,
        0,
    )

    values = (
        (
            "elo_rating_difference",
            50.0,
            0.50,
            0.80,
        ),
        (
            "listed_qb_rating_difference",
            4.0,
            0.40,
            0.60,
        ),
        (
            "offense_injury_burden_difference",
            -0.20,
            -0.30,
            -0.50,
        ),
        (
            "defense_injury_burden_difference",
            0.10,
            0.20,
            -0.40,
        ),
        (
            "special_teams_injury_burden_difference",
            -0.05,
            -0.10,
            -0.20,
        ),
    )

    rows: list[dict[str, object]] = []

    for (
        feature_name,
        raw_value,
        standardized_value,
        coefficient,
    ) in values:
        contribution = (
            standardized_value
            * coefficient
        )

        rows.append(
            {
                "game_id": "blend_game",
                "feature_name": feature_name,
                "raw_feature_value": raw_value,
                "standardized_feature_value": (
                    standardized_value
                ),
                "coefficient": coefficient,
                "log_odds_contribution": (
                    contribution
                ),
                "absolute_log_odds_contribution": (
                    abs(contribution)
                ),
                "model_name": (
                    "elo_injury_logistic_blend"
                ),
                "model_version": "0.2.0",
                "prediction_generated_at": (
                    generated_at
                ),
            }
        )

    contributions = pd.DataFrame(rows)

    logistic_intercept = -0.10

    logistic_total_log_odds = (
        logistic_intercept
        + contributions[
            "log_odds_contribution"
        ].sum()
    )

    logistic_probability = (
        1.0
        / (
            1.0
            + pow(
                2.718281828459045,
                -logistic_total_log_odds,
            )
        )
    )

    contributions[
        "logistic_intercept"
    ] = logistic_intercept

    contributions[
        "logistic_total_log_odds"
    ] = logistic_total_log_odds

    contributions[
        "logistic_reconstructed_home_win_probability"
    ] = logistic_probability

    contributions = contributions.sort_values(
        by="absolute_log_odds_contribution",
        ascending=False,
        kind="stable",
    ).reset_index(drop=True)

    contributions[
        "contribution_rank"
    ] = range(
        1,
        len(contributions) + 1,
    )

    return contributions.loc[
        :,
        CURRENT_LOGISTIC_CONTRIBUTION_COLUMNS,
    ]


def test_create_and_validate_contribution_table(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Persist and validate exact contributions."""

    contributions = create_contributions()

    create_current_logistic_feature_contributions_table(
        connection=connection,
        contributions=contributions,
    )

    validate_current_logistic_feature_contributions_table(
        connection=connection,
        expected_row_count=5,
        expected_feature_count=5,
    )

    stored = connection.execute(
        f"""
        SELECT *
        FROM {TARGET_FULL_NAME}
        ORDER BY contribution_rank
        """
    ).fetchdf()

    assert len(stored) == 5

    assert list(
        stored["contribution_rank"]
    ) == [
        1,
        2,
        3,
        4,
        5,
    ]


def test_empty_contribution_table_is_supported(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Create and validate a stable empty table."""

    contributions = pd.DataFrame(
        columns=(
            CURRENT_LOGISTIC_CONTRIBUTION_COLUMNS
        )
    )

    create_current_logistic_feature_contributions_table(
        connection=connection,
        contributions=contributions,
    )

    validate_current_logistic_feature_contributions_table(
        connection=connection,
        expected_row_count=0,
        expected_feature_count=5,
    )

    stored_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        """
    ).fetchone()[0]

    assert stored_count == 0


def test_builder_rejects_missing_columns(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Reject an incomplete contribution schema."""

    contributions = (
        create_contributions().drop(
            columns=[
                "coefficient",
            ]
        )
    )

    with pytest.raises(
        ValueError,
        match="missing columns",
    ):
        create_current_logistic_feature_contributions_table(
            connection=connection,
            contributions=contributions,
        )


def test_validator_rejects_invalid_math(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Reject an incorrect fitted contribution."""

    create_current_logistic_feature_contributions_table(
        connection=connection,
        contributions=create_contributions(),
    )

    connection.execute(
        f"""
        UPDATE {TARGET_FULL_NAME}
        SET log_odds_contribution = 99.0
        WHERE contribution_rank = 1
        """
    )

    with pytest.raises(
        RuntimeError,
        match="mathematics",
    ):
        validate_current_logistic_feature_contributions_table(
            connection=connection,
            expected_row_count=5,
            expected_feature_count=5,
        )


def test_validator_rejects_invalid_ranks(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Reject duplicate contribution ranks."""

    create_current_logistic_feature_contributions_table(
        connection=connection,
        contributions=create_contributions(),
    )

    connection.execute(
        f"""
        UPDATE {TARGET_FULL_NAME}
        SET contribution_rank = 1
        WHERE contribution_rank = 2
        """
    )

    with pytest.raises(
        RuntimeError,
        match="groups or ranks",
    ):
        validate_current_logistic_feature_contributions_table(
            connection=connection,
            expected_row_count=5,
            expected_feature_count=5,
        )


def test_validator_rejects_invalid_reconstruction(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Reject probability inconsistent with contributions."""

    create_current_logistic_feature_contributions_table(
        connection=connection,
        contributions=create_contributions(),
    )

    connection.execute(
        f"""
        UPDATE {TARGET_FULL_NAME}
        SET logistic_total_log_odds
            = logistic_total_log_odds + 1.0
        """
    )

    with pytest.raises(
        RuntimeError,
        match="reconstruction",
    ):
        validate_current_logistic_feature_contributions_table(
            connection=connection,
            expected_row_count=5,
            expected_feature_count=5,
        )