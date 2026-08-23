"""
Tests for the current Moneyline value builder.
"""

from pathlib import Path

import duckdb
import pandas as pd
import pytest

from src.betting.build_current_moneyline_value import (
    TARGET_FULL_NAME,
    build_current_moneyline_value,
    create_current_moneyline_value_table,
    load_current_game_predictions,
    load_current_moneyline_market,
    validate_current_moneyline_value_table,
    validate_source_tables,
)
from src.betting.current_moneyline_value import (
    MONEYLINE_VALUE_COLUMNS,
    create_current_moneyline_value,
)


def create_source_database(
    database_file: Path,
) -> None:
    """Create minimal Moneyline and prediction sources."""

    with duckdb.connect(
        str(database_file)
    ) as connection:
        connection.execute(
            """
            CREATE SCHEMA analytics;

            CREATE TABLE analytics.current_market_board (
                snapshot_id VARCHAR,
                fetched_at TIMESTAMPTZ,
                game_id VARCHAR,
                season INTEGER,
                game_type VARCHAR,
                week INTEGER,
                gameday DATE,
                commence_time TIMESTAMPTZ,
                home_team VARCHAR,
                away_team VARCHAR,
                market_key VARCHAR,
                market_name VARCHAR,
                outcome_name VARCHAR,
                outcome_type VARCHAR,
                best_bookmaker_key VARCHAR,
                best_bookmaker_title VARCHAR,
                best_american_price INTEGER,
                best_decimal_odds DOUBLE,
                best_implied_probability DOUBLE,
                bookmaker_count INTEGER,
                consensus_no_vig_probability DOUBLE
            );

            CREATE TABLE analytics.current_game_predictions (
                game_id VARCHAR,
                model_name VARCHAR,
                model_version VARCHAR,
                prediction_mode VARCHAR,
                home_win_probability DOUBLE,
                away_win_probability DOUBLE,
                prediction_generated_at TIMESTAMP
            );
            """
        )

        connection.executemany(
            """
            INSERT INTO analytics.current_market_board
            VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            [
                (
                    "snapshot_001",
                    "2026-08-09T08:00:00Z",
                    "2026_01_DAL_PHI",
                    2026,
                    "REG",
                    1,
                    "2026-09-10",
                    "2026-09-11T00:20:00Z",
                    "PHI",
                    "DAL",
                    "h2h",
                    "Moneyline",
                    "PHI",
                    "home",
                    "book_a",
                    "Book A",
                    120,
                    2.20,
                    1.0 / 2.20,
                    5,
                    0.45,
                ),
                (
                    "snapshot_001",
                    "2026-08-09T08:00:00Z",
                    "2026_01_DAL_PHI",
                    2026,
                    "REG",
                    1,
                    "2026-09-10",
                    "2026-09-11T00:20:00Z",
                    "PHI",
                    "DAL",
                    "h2h",
                    "Moneyline",
                    "DAL",
                    "away",
                    "book_b",
                    "Book B",
                    -125,
                    1.80,
                    1.0 / 1.80,
                    4,
                    0.55,
                ),
                (
                    "snapshot_001",
                    "2026-08-09T08:00:00Z",
                    "2026_01_DAL_PHI",
                    2026,
                    "REG",
                    1,
                    "2026-09-10",
                    "2026-09-11T00:20:00Z",
                    "PHI",
                    "DAL",
                    "totals",
                    "Totals",
                    "Over",
                    "over",
                    "book_c",
                    "Book C",
                    -110,
                    1.91,
                    1.0 / 1.91,
                    3,
                    0.50,
                ),
            ],
        )

        connection.execute(
            """
            INSERT INTO analytics.current_game_predictions
            VALUES (
                '2026_01_DAL_PHI',
                'production_logistic',
                '0.1.0',
                'PRIMARY',
                0.50,
                0.50,
                TIMESTAMP '2026-08-09 08:30:00'
            )
            """
        )


def load_value_frame(
    connection: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    """Load sources and calculate valid Moneyline value."""

    market_board = load_current_moneyline_market(
        connection
    )

    predictions = load_current_game_predictions(
        connection
    )

    return create_current_moneyline_value(
        market_board=market_board,
        predictions=predictions,
    )


def test_validate_source_tables_rejects_missing_tables(
) -> None:
    """Fail when the required source tables are absent."""

    with duckdb.connect(":memory:") as connection:
        with pytest.raises(
            RuntimeError,
            match="Missing Moneyline value source tables",
        ):
            validate_source_tables(connection)


def test_load_current_moneyline_sources(
    tmp_path: Path,
) -> None:
    """Load only Moneyline offers and their predictions."""

    database_file = tmp_path / "sources.duckdb"

    create_source_database(database_file)

    with duckdb.connect(
        str(database_file)
    ) as connection:
        validate_source_tables(connection)

        market_board = (
            load_current_moneyline_market(
                connection
            )
        )

        predictions = (
            load_current_game_predictions(
                connection
            )
        )

    assert len(market_board) == 2
    assert market_board["market_key"].eq(
        "h2h"
    ).all()

    assert set(
        market_board["outcome_type"]
    ) == {
        "home",
        "away",
    }

    assert len(predictions) == 1

    assert predictions.iloc[0][
        "home_win_probability"
    ] == pytest.approx(0.50)


def test_create_and_validate_moneyline_value_table(
    tmp_path: Path,
) -> None:
    """Persist and validate calculated Moneyline value."""

    database_file = tmp_path / "value.duckdb"

    create_source_database(database_file)

    with duckdb.connect(
        str(database_file)
    ) as connection:
        value = load_value_frame(connection)

        create_current_moneyline_value_table(
            connection=connection,
            value=value,
        )

        validate_current_moneyline_value_table(
            connection=connection,
            expected_row_count=2,
            expected_game_count=1,
        )

        persisted = connection.execute(
            f"""
            SELECT
                outcome_type,
                model_probability,
                probability_edge,
                expected_value_per_unit,
                positive_expected_value
            FROM {TARGET_FULL_NAME}
            ORDER BY outcome_type
            """
        ).fetchdf()

    assert len(persisted) == 2

    away_row = persisted.loc[
        persisted["outcome_type"] == "away"
    ].iloc[0]

    home_row = persisted.loc[
        persisted["outcome_type"] == "home"
    ].iloc[0]

    assert away_row[
        "expected_value_per_unit"
    ] == pytest.approx(-0.10)

    assert not bool(
        away_row["positive_expected_value"]
    )

    assert home_row[
        "probability_edge"
    ] == pytest.approx(0.05)

    assert home_row[
        "expected_value_per_unit"
    ] == pytest.approx(0.10)

    assert bool(
        home_row["positive_expected_value"]
    )


def test_invalid_persisted_calculation_is_rejected(
    tmp_path: Path,
) -> None:
    """Reject a persisted EV inconsistent with its inputs."""

    database_file = tmp_path / "invalid.duckdb"

    create_source_database(database_file)

    with duckdb.connect(
        str(database_file)
    ) as connection:
        value = load_value_frame(connection)

        create_current_moneyline_value_table(
            connection=connection,
            value=value,
        )

        connection.execute(
            f"""
            UPDATE {TARGET_FULL_NAME}
            SET expected_value_per_unit = 99.0
            WHERE outcome_type = 'home'
            """
        )

        with pytest.raises(
            RuntimeError,
            match="value calculations",
        ):
            validate_current_moneyline_value_table(
                connection=connection,
                expected_row_count=2,
                expected_game_count=1,
            )


def test_missing_output_column_is_rejected(
    tmp_path: Path,
) -> None:
    """Reject an incomplete Moneyline value frame."""

    database_file = tmp_path / "missing.duckdb"

    create_source_database(database_file)

    with duckdb.connect(
        str(database_file)
    ) as connection:
        value = load_value_frame(
            connection
        ).drop(
            columns=[
                "expected_value_percent",
            ]
        )

        with pytest.raises(
            ValueError,
            match="missing columns",
        ):
            create_current_moneyline_value_table(
                connection=connection,
                value=value,
            )


def test_build_current_moneyline_value_end_to_end(
    tmp_path: Path,
) -> None:
    """Build the persisted table from source to output."""

    database_file = tmp_path / "build.duckdb"

    create_source_database(database_file)

    result = build_current_moneyline_value(
        database_file
    )

    with duckdb.connect(
        str(database_file)
    ) as connection:
        persisted = connection.execute(
            f"""
            SELECT *
            FROM {TARGET_FULL_NAME}
            ORDER BY
                expected_value_per_unit DESC
            """
        ).fetchdf()

    assert tuple(result.columns) == (
        MONEYLINE_VALUE_COLUMNS
    )

    assert len(result) == 2
    assert len(persisted) == 2

    assert persisted.iloc[0][
        "outcome_type"
    ] == "home"

    assert persisted.iloc[0][
        "expected_value_percent"
    ] == pytest.approx(10.0)

    assert persisted.iloc[0][
        "full_kelly_fraction"
    ] == pytest.approx(
        0.10 / 1.20
    )