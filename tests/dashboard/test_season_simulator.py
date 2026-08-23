import pandas as pd

from src.dashboard.pages.season_simulator import (
    _distribution_chart,
    _leader_cards,
    _methodology_copy,
)
from src.dashboard.view_models import prepare_simulation_standings


def create_standings() -> pd.DataFrame:
    return prepare_simulation_standings(pd.DataFrame({
        "team": ["BUF", "KC"], "games": [17, 17],
        "expected_wins": [11.2, 10.4], "expected_losses": [5.8, 6.6],
        "median_wins": [11.0, 10.0], "p10_wins": [9.0, 8.0],
        "p90_wins": [14.0, 13.0], "most_likely_wins": [11, 10],
        "expected_final_elo": [1610.0, 1580.0],
    }))


def test_leader_cards_include_rank_logo_and_expected_wins() -> None:
    markup = _leader_cards(create_standings())

    assert "#1" in markup
    assert "BUF" in markup
    assert "11.2" in markup
    assert markup.count("nap-team-logo") == 2


def test_distribution_chart_contains_expected_wins_marker() -> None:
    distribution = pd.DataFrame({
        "team": ["BUF", "BUF", "BUF"],
        "wins": [10, 11, 12],
        "probability": [0.2, 0.5, 0.3],
    })

    figure = _distribution_chart("BUF", distribution, 11.1)

    assert list(figure.data[0].x) == [10, 11, 12]
    assert list(figure.data[0].y) == [0.2, 0.5, 0.3]
    assert figure.layout.shapes[0].x0 == 11.1


def test_hungarian_simulator_copy_explains_dynamic_elo() -> None:
    copy = _methodology_copy("HU", 10_000)

    assert "10 000" in copy
    assert "minden szimulált mérkőzés után frissül" in copy


def test_hungarian_chart_uses_local_axis_labels() -> None:
    distribution = pd.DataFrame({
        "team": ["BUF"], "wins": [11], "probability": [1.0],
    })

    figure = _distribution_chart("BUF", distribution, 11.0, "HU")

    assert figure.layout.xaxis.title.text == "Alapszakasz-győzelmek"
    assert figure.layout.yaxis.title.text == "Valószínűség"
