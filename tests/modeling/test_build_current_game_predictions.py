"""Tests for the current game prediction builder."""

import duckdb
import pandas as pd
import pytest

from src.modeling.build_current_game_predictions import (
    CURRENT_PRODUCTION_PREDICTION_COLUMNS,
    EXPLANATION_FULL_NAME,
    TARGET_FULL_NAME,
    create_current_explanations_table,
    create_current_predictions_table,
    create_current_production_explanations_table,
    create_current_production_predictions_table,
    load_upcoming_game_inputs,
    validate_current_explanations_table,
    validate_current_predictions_table,
    validate_current_production_explanations_table,
    validate_current_production_predictions_table,
    validate_prediction_sources,
)
from src.modeling.current_game_prediction_explanations import (
    create_prediction_explanation_frame,
)
from src.modeling.current_production_prediction_explanations import (
    PRODUCTION_EXPLANATION_COLUMNS,
    create_production_prediction_explanation_frame,
)
from src.modeling.current_game_predictions import (
    create_current_prediction_frame,
)


@pytest.fixture
def connection() -> duckdb.DuckDBPyConnection:
    """Create an in-memory prediction database."""

    database = duckdb.connect(":memory:")

    database.execute(
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
            is_completed BOOLEAN
        );

        CREATE TABLE analytics.current_elo_ratings (
            elo_rank INTEGER,
            team VARCHAR,
            elo_rating DOUBLE,
            games_played INTEGER,
            last_game_id VARCHAR,
            as_of_gameday DATE,
            last_completed_season INTEGER
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
                FALSE
            ),
            (
                '2026_01_BUF_KC',
                2026,
                'REG',
                1,
                DATE '2026-09-11',
                '19:00',
                'BUF',
                'KC',
                'Neutral',
                FALSE
            ),
            (
                '2025_18_NE_BUF',
                2025,
                'REG',
                18,
                DATE '2026-01-04',
                '13:00',
                'NE',
                'BUF',
                'Home',
                TRUE
            ),
            (
                '2026_PRE_NYJ_KC',
                2026,
                'PRE',
                1,
                DATE '2026-08-10',
                '18:00',
                'NYJ',
                'KC',
                'Home',
                FALSE
            );

        INSERT INTO analytics.current_elo_ratings
        VALUES
            (
                1,
                'NE',
                1600.0,
                17,
                'NE_LAST',
                DATE '2026-01-04',
                2025
            ),
            (
                2,
                'NYJ',
                1400.0,
                17,
                'NYJ_LAST',
                DATE '2026-01-04',
                2025
            ),
            (
                3,
                'BUF',
                1550.0,
                17,
                'BUF_LAST',
                DATE '2026-01-04',
                2025
            ),
            (
                4,
                'KC',
                1550.0,
                17,
                'KC_LAST',
                DATE '2026-01-04',
                2025
            );
        """
    )

    yield database

    database.close()


def test_validate_prediction_sources(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Accept complete schedule and Elo sources."""

    validate_prediction_sources(connection)


def test_load_upcoming_game_inputs(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Load regular and postseason upcoming games only."""

    upcoming_games = load_upcoming_game_inputs(
        connection
    )

    assert list(upcoming_games["game_id"]) == [
        "2026_01_NE_NYJ",
        "2026_01_BUF_KC",
    ]

    assert upcoming_games.loc[
        0,
        "home_elo_rating",
    ] == 1600.0

    assert upcoming_games.loc[
        0,
        "away_elo_rating",
    ] == 1400.0

    assert set(
        upcoming_games["home_rating_season"]
    ) == {
        2025
    }


def test_load_upcoming_games_rejects_missing_rating(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Reject an upcoming team without current Elo."""

    connection.execute(
        """
        DELETE FROM analytics.current_elo_ratings
        WHERE team = 'NYJ'
        """
    )

    with pytest.raises(
        RuntimeError,
        match="2026_01_NE_NYJ",
    ):
        load_upcoming_game_inputs(connection)


def test_validate_prediction_sources_rejects_table(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Reject a missing current Elo table."""

    connection.execute(
        """
        DROP TABLE analytics.current_elo_ratings
        """
    )

    with pytest.raises(
        RuntimeError,
        match="does not exist",
    ):
        validate_prediction_sources(connection)


def test_create_and_validate_predictions_table(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Persist valid current production predictions."""

    upcoming_games = load_upcoming_game_inputs(
        connection
    )

    predictions = create_current_prediction_frame(
        upcoming_games
    )

    create_current_predictions_table(
        connection=connection,
        predictions=predictions,
    )

    validate_current_predictions_table(
        connection=connection,
        expected_row_count=2,
    )

    stored_predictions = connection.execute(
        f"""
        SELECT
            game_id,
            model_name,
            model_version,
            home_win_probability,
            away_win_probability
        FROM {TARGET_FULL_NAME}
        ORDER BY game_id
        """
    ).fetchdf()

    assert len(stored_predictions) == 2
    assert set(
        stored_predictions["model_name"]
    ) == {
        "elo"
    }
    assert set(
        stored_predictions["model_version"]
    ) == {
        "1.0.0"
    }


def test_create_empty_predictions_table(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Create a stable empty table without upcoming games."""

    upcoming_games = load_upcoming_game_inputs(
        connection
    ).iloc[0:0]

    predictions = create_current_prediction_frame(
        upcoming_games
    )

    create_current_predictions_table(
        connection=connection,
        predictions=predictions,
    )

    validate_current_predictions_table(
        connection=connection,
        expected_row_count=0,
    )

    row_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        """
    ).fetchone()[0]

    assert row_count == 0


def test_prediction_validation_rejects_probability(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Reject invalid stored probabilities."""

    upcoming_games = load_upcoming_game_inputs(
        connection
    )

    predictions = create_current_prediction_frame(
        upcoming_games
    )

    create_current_predictions_table(
        connection=connection,
        predictions=predictions,
    )

    connection.execute(
        f"""
        UPDATE {TARGET_FULL_NAME}
        SET home_win_probability = 1.5
        WHERE game_id = '2026_01_NE_NYJ'
        """
    )

    with pytest.raises(
        RuntimeError,
        match="Invalid current prediction "
        "probabilities",
    ):
        validate_current_predictions_table(
            connection=connection,
            expected_row_count=2,
        )


def test_create_and_validate_explanations_table(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Persist one explanation per prediction."""

    upcoming_games = load_upcoming_game_inputs(
        connection
    )

    predictions = create_current_prediction_frame(
        upcoming_games
    )

    explanations = (
        create_prediction_explanation_frame(
            predictions
        )
    )

    create_current_explanations_table(
        connection=connection,
        explanations=explanations,
    )

    validate_current_explanations_table(
        connection=connection,
        expected_row_count=2,
    )

    stored = connection.execute(
        f"""
        SELECT
            game_id,
            favorite,
            matchup_label,
            total_home_log_odds
        FROM {EXPLANATION_FULL_NAME}
        ORDER BY game_id
        """
    ).fetchdf()

    assert len(stored) == 2
    assert stored["favorite"].notna().all()
    assert stored["matchup_label"].notna().all()


def test_explanation_validation_rejects_edge(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Reject an inconsistent adjusted Elo edge."""

    upcoming_games = load_upcoming_game_inputs(
        connection
    )

    predictions = create_current_prediction_frame(
        upcoming_games
    )

    explanations = (
        create_prediction_explanation_frame(
            predictions
        )
    )

    create_current_explanations_table(
        connection=connection,
        explanations=explanations,
    )

    connection.execute(
        f"""
        UPDATE {EXPLANATION_FULL_NAME}
        SET adjusted_home_rating_edge = 999.0
        WHERE game_id = '2026_01_NE_NYJ'
        """
    )

    with pytest.raises(
        RuntimeError,
        match="Invalid current prediction "
        "explanations",
    ):
        validate_current_explanations_table(
            connection=connection,
            expected_row_count=2,
        )



def create_external_production_predictions(
) -> pd.DataFrame:
    """Create valid primary and fallback persistence rows."""

    common_values = {
        "season": 2026,
        "game_type": "REG",
        "week": 1,
        "gameday": "2026-09-10",
        "gametime": "20:20",
        "is_neutral": False,
        "model_name": (
            "external_nfelo_probability_routing"
        ),
        "model_version": "0.3.0",
        "home_rating_current": 1550.0,
        "away_rating_current": 1500.0,
        "home_rating_pregame": 1540.0,
        "away_rating_pregame": 1500.0,
        "applied_home_advantage": 48.0,
        "home_rating_as_of": "2026-02-01",
        "away_rating_as_of": "2026-01-15",
        "prediction_generated_at": (
            "2026-08-09 12:00:00"
        ),
    }

    primary_probability = (
        0.70 * 0.70
        + 0.30 * 0.60
    )

    primary_row = {
        **common_values,
        "game_id": "primary_game",
        "home_team": "PHI",
        "away_team": "DAL",
        "home_win_probability": (
            primary_probability
        ),
        "away_win_probability": (
            1.0 - primary_probability
        ),
        "predicted_winner": "PHI",
        "prediction_mode": (
            "EXTERNAL_NFELO_BLEND"
        ),
        "prediction_mode_reason": (
            "complete_external_primary_features"
        ),
        "published_nfelo_home_probability": 0.60,
        "primary_logistic_home_win_probability": 0.70,
        "fallback_logistic_home_win_probability": None,
        "applied_primary_logistic_weight": 0.70,
        "applied_published_nfelo_weight": 0.30,
        "elo_home_win_probability": 0.60,
        "logistic_home_win_probability": 0.70,
        "applied_logistic_weight": 0.70,
        "applied_elo_weight": 0.30,
        "has_complete_injury_data": True,
        "both_listed_qb_ratings_available": True,
        "has_complete_production_features": True,
        "has_complete_fallback_features": True,
        "external_nfelo_rating_difference": 100.0,
        "listed_qb_rating_difference": 4.0,
        "external_nfelo_qb_adjustment_difference": 6.0,
        "offense_injury_burden_difference": -0.20,
        "defense_injury_burden_difference": -0.10,
        "special_teams_injury_burden_difference": -0.05,
    }

    fallback_row = {
        **common_values,
        "game_id": "fallback_game",
        "home_team": "NE",
        "away_team": "MIA",
        "home_win_probability": 0.43,
        "away_win_probability": 0.57,
        "predicted_winner": "MIA",
        "prediction_mode": (
            "EXTERNAL_ELO_QB_FALLBACK"
        ),
        "prediction_mode_reason": (
            "incomplete_external_primary_features"
        ),
        "published_nfelo_home_probability": 0.47,
        "primary_logistic_home_win_probability": None,
        "fallback_logistic_home_win_probability": 0.43,
        "applied_primary_logistic_weight": 0.0,
        "applied_published_nfelo_weight": 0.0,
        "elo_home_win_probability": 0.47,
        "logistic_home_win_probability": None,
        "applied_logistic_weight": 0.0,
        "applied_elo_weight": 0.0,
        "has_complete_injury_data": False,
        "both_listed_qb_ratings_available": False,
        "has_complete_production_features": False,
        "has_complete_fallback_features": True,
        "external_nfelo_rating_difference": -45.0,
        "listed_qb_rating_difference": None,
        "external_nfelo_qb_adjustment_difference": -3.0,
        "offense_injury_burden_difference": None,
        "defense_injury_burden_difference": None,
        "special_teams_injury_burden_difference": None,
    }

    return pd.DataFrame(
        [
            primary_row,
            fallback_row,
        ],
        columns=(
            CURRENT_PRODUCTION_PREDICTION_COLUMNS
        ),
    )


def test_create_and_validate_external_production_table(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Persist both external routing modes."""

    predictions = (
        create_external_production_predictions()
    )

    create_current_production_predictions_table(
        connection=connection,
        predictions=predictions,
    )

    validate_current_production_predictions_table(
        connection=connection,
        expected_row_count=2,
    )

    routing = connection.execute(
        f"""
        SELECT
            game_id,
            prediction_mode,
            home_win_probability
        FROM {TARGET_FULL_NAME}
        ORDER BY game_id
        """
    ).fetchdf()

    assert len(routing) == 2

    assert set(
        routing["prediction_mode"]
    ) == {
        "EXTERNAL_NFELO_BLEND",
        "EXTERNAL_ELO_QB_FALLBACK",
    }


def test_production_validation_rejects_bad_fallback(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Reject fallback probability inconsistency."""

    predictions = (
        create_external_production_predictions()
    )

    create_current_production_predictions_table(
        connection=connection,
        predictions=predictions,
    )

    connection.execute(
        f"""
        UPDATE {TARGET_FULL_NAME}
        SET home_win_probability = 0.80,
            away_win_probability = 0.20
        WHERE game_id = 'fallback_game'
        """
    )

    with pytest.raises(
        RuntimeError,
        match="fallback routing",
    ):
        validate_current_production_predictions_table(
            connection=connection,
            expected_row_count=2,
        )


def test_create_empty_external_production_table(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Create stable empty external production schema."""

    predictions = (
        create_external_production_predictions()
        .iloc[0:0]
    )

    create_current_production_predictions_table(
        connection=connection,
        predictions=predictions,
    )

    validate_current_production_predictions_table(
        connection=connection,
        expected_row_count=0,
    )

    stored_columns = {
        row[0]
        for row in connection.execute(
            f"""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'analytics'
              AND table_name = 'current_game_predictions'
            """
        ).fetchall()
    }

    assert set(
        CURRENT_PRODUCTION_PREDICTION_COLUMNS
    ) == stored_columns


from src.modeling.current_production_prediction_explanations import (
    PRODUCTION_EXPLANATION_COLUMNS,
    create_production_prediction_explanation_frame,
)


def test_create_and_validate_external_explanations(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Persist both external explanation modes."""

    predictions = (
        create_external_production_predictions()
    )

    explanations = (
        create_production_prediction_explanation_frame(
            predictions
        )
    )

    create_current_production_explanations_table(
        connection=connection,
        explanations=explanations,
    )

    validate_current_production_explanations_table(
        connection=connection,
        expected_row_count=2,
    )

    stored = connection.execute(
        f"""
        SELECT
            game_id,
            prediction_mode,
            favorite,
            home_win_probability,
            published_nfelo_home_probability
        FROM {EXPLANATION_FULL_NAME}
        ORDER BY game_id
        """
    ).fetchdf()

    assert len(stored) == 2

    assert set(
        stored["prediction_mode"]
    ) == {
        "EXTERNAL_NFELO_BLEND",
        "EXTERNAL_ELO_QB_FALLBACK",
    }


def test_external_explanation_validation_rejects_routing(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Reject an inconsistent persisted fallback."""

    predictions = (
        create_external_production_predictions()
    )

    explanations = (
        create_production_prediction_explanation_frame(
            predictions
        )
    )

    create_current_production_explanations_table(
        connection=connection,
        explanations=explanations,
    )

    connection.execute(
        f"""
        UPDATE {EXPLANATION_FULL_NAME}
        SET fallback_logistic_home_win_probability
                = 0.80,
            fallback_logistic_away_win_probability
                = 0.20
        WHERE game_id = 'fallback_game'
        """
    )

    with pytest.raises(
        RuntimeError,
        match="fallback explanation routing",
    ):
        validate_current_production_explanations_table(
            connection=connection,
            expected_row_count=2,
        )


def test_create_empty_external_explanation_table(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Create a stable empty explanation schema."""

    predictions = (
        create_external_production_predictions()
        .iloc[0:0]
    )

    explanations = (
        create_production_prediction_explanation_frame(
            predictions
        )
    )

    create_current_production_explanations_table(
        connection=connection,
        explanations=explanations,
    )

    validate_current_production_explanations_table(
        connection=connection,
        expected_row_count=0,
    )

    stored_columns = {
        row[0]
        for row in connection.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'analytics'
              AND table_name
                    = 'current_game_prediction_explanations'
            """
        ).fetchall()
    }

    assert set(
        PRODUCTION_EXPLANATION_COLUMNS
    ) == stored_columns