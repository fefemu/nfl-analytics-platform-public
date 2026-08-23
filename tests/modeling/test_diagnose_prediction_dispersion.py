import numpy as np
import pandas as pd
import pytest

from src.modeling.diagnose_prediction_dispersion import (
    DISPERSION_COLUMNS,
    EXTREME_COLUMNS,
    MARKET_BENCHMARK_COLUMNS,
    SEASON_WIN_COLUMNS,
    calculate_dispersion_summary,
    calculate_extreme_rate_summary,
    calculate_market_benchmark_summary,
    summarize_season_wins,
)


def test_dispersion_summary_detects_compression() -> None:
    result = calculate_dispersion_summary(
        model_layer="SPREAD",
        actual_values=np.array([-14.0, -7.0, 7.0, 14.0]),
        predicted_values=np.array([-7.0, -3.5, 3.5, 7.0]),
    )

    assert tuple(result.columns) == DISPERSION_COLUMNS
    assert result.loc[
        0,
        "prediction_to_actual_dispersion_ratio",
    ] == pytest.approx(0.5)
    assert result.loc[0, "calibration_slope"] == pytest.approx(2.0)
    assert result.loc[0, "calibration_intercept"] == pytest.approx(0.0)


def test_dispersion_summary_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="same shape"):
        calculate_dispersion_summary(
            model_layer="SPREAD",
            actual_values=np.array([1.0, 2.0]),
            predicted_values=np.array([1.0]),
        )


def test_extreme_summary_counts_absolute_values() -> None:
    result = calculate_extreme_rate_summary(
        model_layer="SPREAD_ABSOLUTE_MARGIN",
        actual_values=np.array([-14.0, -8.0, 2.0, 11.0]),
        predicted_values=np.array([-7.0, -4.0, 1.0, 8.0]),
        thresholds=(7.0, 10.0),
    )

    assert tuple(result.columns) == EXTREME_COLUMNS
    assert result["actual_game_count"].tolist() == [3, 2]
    assert result["predicted_game_count"].tolist() == [2, 0]


def test_season_win_summary_uses_all_team_seasons() -> None:
    source = pd.DataFrame(
        {
            "season": [2024, 2024, 2025, 2025],
            "team": ["A", "B", "A", "B"],
            "wins": [3.0, 14.0, 4.0, 13.0],
        }
    )

    result = summarize_season_wins(
        comparison_group="HISTORICAL",
        team_seasons=source,
    )

    assert tuple(result.columns) == SEASON_WIN_COLUMNS
    assert result.loc[0, "season_count"] == 2
    assert result.loc[0, "team_season_count"] == 4
    assert result.loc[0, "minimum_wins"] == 3.0
    assert result.loc[0, "maximum_wins"] == 14.0


def test_market_benchmark_detects_model_compression() -> None:
    result = calculate_market_benchmark_summary(
        model_layer="SPREAD",
        model_values=np.array([-3.0, -1.0, 1.0, 3.0]),
        market_values=np.array([-6.0, -2.0, 2.0, 6.0]),
    )

    assert tuple(result.columns) == MARKET_BENCHMARK_COLUMNS
    assert result.loc[
        0,
        "model_to_market_dispersion_ratio",
    ] == pytest.approx(0.5)
    assert result.loc[0, "market_on_model_slope"] == pytest.approx(2.0)


def test_season_win_summary_rejects_missing_columns() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        summarize_season_wins(
            comparison_group="HISTORICAL",
            team_seasons=pd.DataFrame({"wins": [3.0]}),
        )
