"""Tests for the quarterback ratings builder."""

from datetime import date

import duckdb
import pytest

from src.analytics.build_qb_ratings import (
    QuarterbackGameRecord,
    QuarterbackRatingState,
    calculate_qb_ratings,
    calculate_rating_snapshot,
    create_qb_rating_tables,
    load_qb_games,
)


def create_qb_game(
    *,
    game_id: str,
    game_date: date,
    qb_id: str,
    qb_name: str,
    team: str,
    opponent: str,
    adjusted_epa: float,
    dropbacks: int = 30,
) -> QuarterbackGameRecord:
    """Create a compact synthetic QB-game record."""

    return QuarterbackGameRecord(
        game_id=game_id,
        season=game_date.year,
        season_type="REG",
        week=1,
        game_date=game_date,
        team=team,
        opponent=opponent,
        qb_id=qb_id,
        qb_name=qb_name,
        is_primary_qb=True,
        team_dropback_share=1.0,
        dropbacks=dropbacks,
        throw_attempts=dropbacks - 3,
        epa_per_dropback=adjusted_epa,
        opponent_defensive_epa=0.0,
        adjusted_epa_per_dropback=adjusted_epa,
        cpoe=2.0,
        sacks=2,
        turnovers=1,
    )


def test_rating_state_decays_by_half_after_one_half_life(
) -> None:
    """Halve effective historical information after one half-life."""

    state = QuarterbackRatingState()

    state.update(
        game_date=date(2025, 1, 1),
        dropbacks=100,
        adjusted_epa_per_dropback=0.2,
        throw_attempts=80,
        cpoe=5.0,
        sacks=5,
        turnovers=2,
        half_life_days=365.0,
    )

    state.decay_to(
        date(2026, 1, 1),
        half_life_days=365.0,
    )

    assert state.weighted_dropbacks == pytest.approx(50.0)
    assert state.weighted_throw_attempts == pytest.approx(40.0)
    assert state.weighted_sacks == pytest.approx(2.5)
    assert state.weighted_turnovers == pytest.approx(1.0)

    assert state.mean_epa_per_dropback == pytest.approx(0.2)
    assert state.mean_cpoe == pytest.approx(5.0)


def test_rating_state_rejects_backward_time() -> None:
    """Do not allow a rating state to move backward in time."""

    state = QuarterbackRatingState(
        last_updated=date(2025, 9, 1)
    )

    with pytest.raises(
        ValueError,
        match="cannot move backward in time",
    ):
        state.decay_to(date(2025, 8, 1))


def test_small_sample_is_shrunk_more_than_large_sample(
) -> None:
    """Pull a small QB sample closer to the league average."""

    league_state = QuarterbackRatingState()
    league_state.update(
        game_date=date(2025, 1, 1),
        dropbacks=1000,
        adjusted_epa_per_dropback=0.1,
        throw_attempts=850,
        cpoe=0.0,
        sacks=60,
        turnovers=25,
    )

    small_state = QuarterbackRatingState()
    small_state.update(
        game_date=date(2025, 1, 1),
        dropbacks=20,
        adjusted_epa_per_dropback=0.3,
        throw_attempts=18,
        cpoe=5.0,
        sacks=1,
        turnovers=0,
    )

    large_state = QuarterbackRatingState()
    large_state.update(
        game_date=date(2025, 1, 1),
        dropbacks=1000,
        adjusted_epa_per_dropback=0.3,
        throw_attempts=850,
        cpoe=5.0,
        sacks=40,
        turnovers=15,
    )

    small_snapshot = calculate_rating_snapshot(
        small_state,
        league_state,
        prior_dropbacks=200,
    )

    large_snapshot = calculate_rating_snapshot(
        large_state,
        league_state,
        prior_dropbacks=200,
    )

    league_mean = 0.1

    small_distance = abs(
        small_snapshot.shrunk_adjusted_epa_per_dropback
        - league_mean
    )

    large_distance = abs(
        large_snapshot.shrunk_adjusted_epa_per_dropback
        - league_mean
    )

    assert small_distance < large_distance
    assert small_snapshot.prior_weight > large_snapshot.prior_weight


def test_first_qb_appearance_starts_at_league_average(
) -> None:
    """Assign a 100 rating before a QB has any game history."""

    games = [
        create_qb_game(
            game_id="game_1",
            game_date=date(2025, 9, 7),
            qb_id="qb_1",
            qb_name="Test QB",
            team="BUF",
            opponent="MIA",
            adjusted_epa=0.5,
        )
    ]

    history_rows, current_rows = calculate_qb_ratings(
        games,
    )

    assert len(history_rows) == 1
    assert len(current_rows) == 1

    first_rating = history_rows[0]

    assert first_rating.pregame_effective_dropbacks == 0
    assert first_rating.pregame_qb_rating == pytest.approx(100.0)
    assert first_rating.pregame_prior_weight == pytest.approx(1.0)


def test_same_day_games_do_not_leak_between_qbs() -> None:
    """Snapshot every same-day rating before updating results."""

    games = [
        create_qb_game(
            game_id="game_1",
            game_date=date(2025, 9, 7),
            qb_id="qb_1",
            qb_name="QB One",
            team="BUF",
            opponent="MIA",
            adjusted_epa=1.0,
        ),
        create_qb_game(
            game_id="game_2",
            game_date=date(2025, 9, 7),
            qb_id="qb_2",
            qb_name="QB Two",
            team="KC",
            opponent="DEN",
            adjusted_epa=-1.0,
        ),
    ]

    history_rows, _ = calculate_qb_ratings(
        games,
        prior_dropbacks=0,
    )

    assert len(history_rows) == 2

    assert history_rows[0].pregame_qb_rating == pytest.approx(100.0)
    assert history_rows[1].pregame_qb_rating == pytest.approx(100.0)


def test_second_game_uses_first_game_history_only() -> None:
    """Use completed prior games in a later pregame rating."""

    games = [
        create_qb_game(
            game_id="game_1",
            game_date=date(2025, 9, 7),
            qb_id="qb_1",
            qb_name="QB One",
            team="BUF",
            opponent="MIA",
            adjusted_epa=0.5,
        ),
        create_qb_game(
            game_id="game_2",
            game_date=date(2025, 9, 14),
            qb_id="qb_1",
            qb_name="QB One",
            team="BUF",
            opponent="NYJ",
            adjusted_epa=-0.5,
        ),
    ]

    history_rows, _ = calculate_qb_ratings(
        games,
        prior_dropbacks=0,
    )

    second_rating = history_rows[1]

    assert second_rating.pregame_effective_dropbacks > 0
    assert (
        second_rating.pregame_raw_adjusted_epa_per_dropback
        == pytest.approx(0.5)
    )


def test_load_qb_games_adjusts_for_opponent_defense(
) -> None:
    """Join the opponent's pregame defensive feature."""

    with duckdb.connect(":memory:") as connection:
        connection.execute(
            """
            CREATE SCHEMA processed;
            CREATE SCHEMA analytics;

            CREATE TABLE processed.qb_game_performance AS
            SELECT
                'game_1'::VARCHAR AS game_id,
                2025::INTEGER AS season,
                'REG'::VARCHAR AS season_type,
                1::INTEGER AS week,
                DATE '2025-09-07' AS game_date,
                'BUF'::VARCHAR AS team,
                'MIA'::VARCHAR AS opponent,
                'qb_1'::VARCHAR AS qb_id,
                'Test QB'::VARCHAR AS qb_name,
                TRUE AS is_primary_qb,
                1.0::DOUBLE AS team_dropback_share,
                30::INTEGER AS dropbacks,
                27::INTEGER AS throw_attempts,
                0.2::DOUBLE AS epa_per_dropback,
                3.0::DOUBLE AS cpoe,
                2::INTEGER AS sacks,
                1::INTEGER AS turnovers;

            CREATE TABLE analytics.rolling_team_features AS
            SELECT
                'game_1'::VARCHAR AS game_id,
                'MIA'::VARCHAR AS team,
                -0.1::DOUBLE
                    AS pregame_defensive_epa_allowed_per_play_last_8;
            """
        )

        games = load_qb_games(connection)

    assert len(games) == 1
    assert games[0].opponent_defensive_epa == pytest.approx(-0.1)
    assert games[0].adjusted_epa_per_dropback == pytest.approx(0.3)


def test_create_qb_rating_tables_matches_dataclass_schema(
) -> None:
    """Insert calculated dataclass rows into both DuckDB tables."""

    games = [
        create_qb_game(
            game_id="game_1",
            game_date=date(2025, 9, 7),
            qb_id="qb_1",
            qb_name="Test QB",
            team="BUF",
            opponent="MIA",
            adjusted_epa=0.2,
        )
    ]

    history_rows, current_rows = calculate_qb_ratings(
        games
    )

    with duckdb.connect(":memory:") as connection:
        create_qb_rating_tables(
            connection,
            history_rows,
            current_rows,
        )

        history_count = connection.execute(
            """
            SELECT COUNT(*)
            FROM analytics.qb_rating_history
            """
        ).fetchone()[0]

        current_result = connection.execute(
            """
            SELECT
                COUNT(*),
                MIN(last_game_date),
                MIN(days_since_last_game)
            FROM analytics.current_qb_ratings
            """
        ).fetchone()

    assert history_count == 1
    assert current_result == (
        1,
        date(2025, 9, 7),
        0,
    )