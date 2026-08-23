"""Tests for historical Elo processing."""

from datetime import date, time
import pytest

from src.models.elo_history import (
    HistoricalGame,
    process_elo_history,
    sort_games_chronologically,
)


def test_sort_games_chronologically() -> None:
    """Sort games by date, time and game identifier."""

    games = [
        HistoricalGame(
            game_id="2025_02_BUF_MIA",
            season=2025,
            game_type="REG",
            week=2,
            gameday=date(2025, 9, 14),
            gametime=time(20, 20),
            home_team="MIA",
            away_team="BUF",
            home_score=20,
            away_score=24,
            is_neutral=False,
        ),
        HistoricalGame(
            game_id="2025_01_GB_CHI",
            season=2025,
            game_type="REG",
            week=1,
            gameday=date(2025, 9, 7),
            gametime=time(13, 0),
            home_team="CHI",
            away_team="GB",
            home_score=17,
            away_score=21,
            is_neutral=False,
        ),
        HistoricalGame(
            game_id="2025_01_DAL_PHI",
            season=2025,
            game_type="REG",
            week=1,
            gameday=date(2025, 9, 7),
            gametime=time(20, 20),
            home_team="PHI",
            away_team="DAL",
            home_score=27,
            away_score=24,
            is_neutral=False,
        ),
    ]

    sorted_games = sort_games_chronologically(games)

    assert [
        game.game_id
        for game in sorted_games
    ] == [
        "2025_01_GB_CHI",
        "2025_01_DAL_PHI",
        "2025_02_BUF_MIA",
    ]


def test_sort_games_uses_game_id_for_equal_start_times() -> None:
    """Use game ID as a deterministic final ordering key."""

    common_values = {
        "season": 2025,
        "game_type": "REG",
        "week": 1,
        "gameday": date(2025, 9, 7),
        "gametime": time(13, 0),
        "home_score": 20,
        "away_score": 17,
        "is_neutral": False,
    }

    games = [
        HistoricalGame(
            game_id="game_b",
            home_team="BUF",
            away_team="MIA",
            **common_values,
        ),
        HistoricalGame(
            game_id="game_a",
            home_team="CHI",
            away_team="GB",
            **common_values,
        ),
    ]

    sorted_games = sort_games_chronologically(games)

    assert [
        game.game_id
        for game in sorted_games
    ] == [
        "game_a",
        "game_b",
    ]


def test_process_history_carries_rating_to_next_game() -> None:
    """Use the previous postgame rating in the next prediction."""

    games = [
        HistoricalGame(
            game_id="game_1",
            season=2025,
            game_type="REG",
            week=1,
            gameday=date(2025, 9, 7),
            gametime=time(13, 0),
            home_team="CHI",
            away_team="GB",
            home_score=24,
            away_score=17,
            is_neutral=False,
        ),
        HistoricalGame(
            game_id="game_2",
            season=2025,
            game_type="REG",
            week=2,
            gameday=date(2025, 9, 14),
            gametime=time(13, 0),
            home_team="CHI",
            away_team="DET",
            home_score=20,
            away_score=17,
            is_neutral=False,
        ),
    ]

    records, final_ratings = process_elo_history(
        games=games,
        home_advantage=0.0,
        k_factor=20.0,
    )

    assert records[0].home_rating_post == pytest.approx(
        1510.0
    )
    assert records[1].home_rating_pre == pytest.approx(
        records[0].home_rating_post
    )
    assert final_ratings["CHI"] == pytest.approx(
        records[1].home_rating_post
    )


def test_process_history_regresses_ratings_between_seasons() -> None:
    """Regress existing ratings before the first game of a new season."""

    games = [
        HistoricalGame(
            game_id="2024_game",
            season=2024,
            game_type="REG",
            week=18,
            gameday=date(2025, 1, 5),
            gametime=time(13, 0),
            home_team="CHI",
            away_team="GB",
            home_score=24,
            away_score=17,
            is_neutral=False,
        ),
        HistoricalGame(
            game_id="2025_game",
            season=2025,
            game_type="REG",
            week=1,
            gameday=date(2025, 9, 7),
            gametime=time(13, 0),
            home_team="CHI",
            away_team="DET",
            home_score=20,
            away_score=17,
            is_neutral=False,
        ),
    ]

    records, _ = process_elo_history(
        games=games,
        home_advantage=0.0,
        k_factor=20.0,
        season_retention=0.70,
    )

    assert records[0].home_rating_post == pytest.approx(
        1510.0
    )
    assert records[1].home_rating_pre == pytest.approx(
        1507.0
    )
    assert records[1].away_rating_pre == pytest.approx(
        1500.0
    )


def test_process_history_preserves_relocated_franchise_rating() -> None:
    """Carry a franchise rating across a team-code relocation."""

    games = [
        HistoricalGame(
            game_id="oakland_game",
            season=2019,
            game_type="REG",
            week=17,
            gameday=date(2019, 12, 29),
            gametime=time(16, 25),
            home_team="OAK",
            away_team="DEN",
            home_score=24,
            away_score=17,
            is_neutral=False,
        ),
        HistoricalGame(
            game_id="las_vegas_game",
            season=2020,
            game_type="REG",
            week=1,
            gameday=date(2020, 9, 13),
            gametime=time(16, 25),
            home_team="LV",
            away_team="CAR",
            home_score=20,
            away_score=17,
            is_neutral=False,
        ),
    ]

    records, final_ratings = process_elo_history(
        games=games,
        home_advantage=0.0,
        k_factor=20.0,
        season_retention=0.70,
    )

    assert records[0].home_franchise == "LV"
    assert records[1].home_franchise == "LV"
    assert records[1].home_rating_pre == pytest.approx(
        1507.0
    )
    assert "OAK" not in final_ratings
    assert "LV" in final_ratings