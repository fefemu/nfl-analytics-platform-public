import pandas as pd

from src.dashboard.pages.data_science_lab import (
    IMPACT_LABELS,
    _build_season_chart,
)


def test_season_chart_uses_brier_score_and_model_series() -> None:
    seasons = pd.DataFrame({
        "validation_season": [2024, 2025],
        "model_name": ["champion", "champion"],
        "brier_score": [0.21, 0.22],
    })

    figure = _build_season_chart(seasons)

    assert list(figure.data[0].x) == [2024, 2025]
    assert list(figure.data[0].y) == [0.21, 0.22]


def test_every_impact_segment_has_bilingual_label() -> None:
    for topic in IMPACT_LABELS.values():
        for label in topic.values():
            assert label["EN"]
            assert label["HU"]
