"""End-to-end tests for current production predictions."""

from pathlib import Path

import duckdb
import pytest

from src.modeling.build_current_game_predictions import (
    EXPLANATION_FULL_NAME,
    TARGET_FULL_NAME,
    build_current_game_predictions,
)


def create_production_database(
    database_file: Path,
) -> None:
    """Create complete temporary production sources."""

    with duckdb.connect(
        str(database_file)
    ) as connection:
        connection.execute(
            """
            CREATE SCHEMA processed;
            CREATE SCHEMA analytics;

            CREATE TABLE processed.schedule (
                game_id VARCHAR,
                season INTEGER,
                game_type VARCHAR,
                week INTEGER,
                gameday DATE,
                gametime VARCHAR,
                home_team VARCHAR,
                away_team VARCHAR,
                location VARCHAR,
                home_qb_id VARCHAR,
                home_qb_name VARCHAR,
                away_qb_id VARCHAR,
                away_qb_name VARCHAR,
                is_completed BOOLEAN
            );

            CREATE TABLE analytics.current_elo_ratings (
                team VARCHAR,
                elo_rating DOUBLE,
                as_of_gameday DATE,
                last_completed_season INTEGER
            );

            CREATE TABLE analytics.current_qb_ratings (
                qb_id VARCHAR,
                qb_name VARCHAR,
                current_team VARCHAR,
                qb_rating DOUBLE,
                as_of_date DATE,
                rating_standard_error DOUBLE
            );

            CREATE TABLE analytics.game_injury_features (
                game_id VARCHAR,
                has_complete_injury_data BOOLEAN,
                offense_injury_burden_difference DOUBLE,
                defense_injury_burden_difference DOUBLE,
                special_teams_injury_burden_difference DOUBLE
            );

            CREATE TABLE analytics.game_modeling_dataset (
                game_id VARCHAR,
                season INTEGER,
                game_date DATE,
                target_home_win INTEGER,
                has_complete_injury_data BOOLEAN,
                elo_rating_difference DOUBLE,
                listed_qb_rating_difference DOUBLE,
                offense_injury_burden_difference DOUBLE,
                defense_injury_burden_difference DOUBLE,
                special_teams_injury_burden_difference DOUBLE
            );

            CREATE TABLE processed.external_nfelo_game_ratings (
                normalized_game_id VARCHAR,
                source_season INTEGER,
                source_week INTEGER,
                home_team VARCHAR,
                away_team VARCHAR,
                starting_nfelo_home DOUBLE,
                starting_nfelo_away DOUBLE,
                home_538_qb_adj DOUBLE,
                away_538_qb_adj DOUBLE,
                nfelo_home_probability_open DOUBLE
            );

            INSERT INTO processed.schedule
            VALUES
                (
                    '2026_01_NE_NYJ',
                    2026,
                    'REG',
                    1,
                    DATE '2026-09-10',
                    '20:20',
                    'NE',
                    'NYJ',
                    'Home',
                    'QB_NE',
                    'New England QB',
                    'QB_NYJ',
                    'New York QB',
                    FALSE
                ),
                (
                    '2026_01_BUF_KC',
                    2026,
                    'REG',
                    1,
                    DATE '2026-09-13',
                    '13:00',
                    'BUF',
                    'KC',
                    'Home',
                    'QB_BUF',
                    'Buffalo QB',
                    NULL,
                    NULL,
                    FALSE
                );

            INSERT INTO analytics.current_elo_ratings
            VALUES
                (
                    'NE',
                    1510.0,
                    DATE '2026-01-04',
                    2025
                ),
                (
                    'NYJ',
                    1475.0,
                    DATE '2026-01-04',
                    2025
                ),
                (
                    'BUF',
                    1570.0,
                    DATE '2026-01-04',
                    2025
                ),
                (
                    'KC',
                    1560.0,
                    DATE '2026-01-04',
                    2025
                );

            INSERT INTO analytics.current_qb_ratings
            VALUES
                (
                    'QB_NE',
                    'New England QB',
                    'NE',
                    5.5,
                    DATE '2026-01-04',
                    1.1
                ),
                (
                    'QB_NYJ',
                    'New York QB',
                    'NYJ',
                    2.0,
                    DATE '2026-01-04',
                    1.3
                ),
                (
                    'QB_BUF',
                    'Buffalo QB',
                    'BUF',
                    7.0,
                    DATE '2026-01-04',
                    0.9
                );

            INSERT INTO analytics.game_injury_features
            VALUES
                (
                    '2026_01_NE_NYJ',
                    TRUE,
                    -0.20,
                    0.10,
                    -0.05
                );
            """
        )

        historical_rows = []

        for index in range(20):
            target = index % 2

            direction = (
                1.0
                if target == 1
                else -1.0
            )

            historical_rows.append(
                (
                    f"history_{index}",
                    2020 + index // 4,
                    f"{2020 + index // 4}-09-01",
                    target,
                    True,
                    direction * 70.0,
                    direction * 4.0,
                    direction * -0.20,
                    direction * -0.15,
                    direction * -0.05,
                )
            )

        connection.executemany(
            """
            INSERT INTO analytics.game_modeling_dataset
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            historical_rows,
        )

        external_rows = [
            (
                row[0], row[1], 1, "NE", "NYJ",
                1550.0 + row[5] / 2.0,
                1550.0 - row[5] / 2.0,
                row[6] / 2.0, -row[6] / 2.0,
                0.60 if row[3] == 1 else 0.40,
            )
            for row in historical_rows
        ]

        external_rows.extend(
            [
                (
                    "2026_01_NE_NYJ", 2026, 1, "NE", "NYJ",
                    1510.0, 1475.0, 2.0, -1.0, 0.60,
                ),
                (
                    "2026_01_BUF_KC", 2026, 1, "BUF", "KC",
                    1570.0, 1560.0, 1.0, 0.0, 0.55,
                ),
            ]
        )

        connection.executemany(
            """
            INSERT INTO processed.external_nfelo_game_ratings
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            external_rows,
        )


def test_build_current_production_pipeline(
    tmp_path: Path,
) -> None:
    """Build blend and fallback predictions end to end."""

    database_file = (
        tmp_path
        / "production_predictions.duckdb"
    )

    create_production_database(
        database_file
    )

    predictions = (
        build_current_game_predictions(
            database_file=database_file
        )
    )

    assert len(predictions) == 2

    assert set(
        predictions["prediction_mode"]
    ) == {
        "EXTERNAL_NFELO_BLEND",
        "EXTERNAL_ELO_QB_FALLBACK",
    }

    assert set(
        predictions["model_name"]
    ) == {
        "external_nfelo_probability_routing",
    }

    assert set(
        predictions["model_version"]
    ) == {
        "0.3.0",
    }


def test_pipeline_persists_auditable_predictions(
    tmp_path: Path,
) -> None:
    """Persist component probabilities and weights."""

    database_file = (
        tmp_path
        / "production_audit.duckdb"
    )

    create_production_database(
        database_file
    )

    build_current_game_predictions(
        database_file=database_file
    )

    with duckdb.connect(
        str(database_file),
        read_only=True,
    ) as connection:
        rows = connection.execute(
            f"""
            SELECT
                game_id,
                prediction_mode,
                published_nfelo_home_probability,
                primary_logistic_home_win_probability,
                applied_primary_logistic_weight,
                applied_published_nfelo_weight,
                has_complete_production_features
            FROM {TARGET_FULL_NAME}
            ORDER BY game_id
            """
        ).fetchall()

    assert len(rows) == 2

    blend_row = next(
        row
        for row in rows
        if row[1] == "EXTERNAL_NFELO_BLEND"
    )

    fallback_row = next(
        row
        for row in rows
        if row[1] == "EXTERNAL_ELO_QB_FALLBACK"
    )

    assert (
        blend_row[2]
        is not None
    )
    assert (
        blend_row[3]
        is not None
    )
    assert blend_row[4] == pytest.approx(0.70)
    assert blend_row[5] == pytest.approx(0.30)
    assert blend_row[6] is True

    assert (
        fallback_row[3]
        is None
    )
    assert fallback_row[4] == pytest.approx(0.0)
    assert fallback_row[5] == pytest.approx(0.0)
    assert fallback_row[6] is False


def test_pipeline_persists_production_explanations(
    tmp_path: Path,
) -> None:
    """Persist final and component explanations."""

    database_file = (
        tmp_path
        / "production_explanations.duckdb"
    )

    create_production_database(
        database_file
    )

    build_current_game_predictions(
        database_file=database_file
    )

    with duckdb.connect(
        str(database_file),
        read_only=True,
    ) as connection:
        rows = connection.execute(
            f"""
            SELECT
                game_id,
                prediction_mode,
                home_win_probability,
                published_nfelo_home_probability,
                primary_logistic_home_win_probability,
                production_probability_adjustment_from_published_nfelo,
                matchup_label
            FROM {EXPLANATION_FULL_NAME}
            ORDER BY game_id
            """
        ).fetchall()

    assert len(rows) == 2

    for row in rows:
        assert (
            row[2]
            is not None
        )
        assert (
            row[3]
            is not None
        )
        assert row[6] in {
            "toss_up",
            "slight_edge",
            "clear_edge",
            "strong_edge",
        }

        if row[1] == "EXTERNAL_NFELO_BLEND":
            assert row[4] is not None
        else:
            assert row[4] is None
            assert row[5] is not None
