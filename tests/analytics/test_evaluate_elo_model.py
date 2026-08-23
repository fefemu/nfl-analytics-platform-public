"""Tests for Elo model evaluation metrics."""

import pytest
import duckdb

from src.analytics.evaluate_elo_model import (
    calculate_accuracy,
    calculate_brier_score,
    calculate_calibration_bins,
    calculate_log_loss,
    evaluate_probabilities,
    evaluate_probabilities_by_season,
    load_evaluation_data,
)

def test_brier_score_is_zero_for_perfect_predictions() -> None:
    """Perfect probability predictions should have zero Brier score."""

    score = calculate_brier_score(
        probabilities=[1.0, 0.0],
        outcomes=[1.0, 0.0],
    )

    assert score == pytest.approx(0.0)


def test_brier_score_is_quarter_for_equal_predictions() -> None:
    """Fifty-fifty predictions should produce a 0.25 Brier score."""

    score = calculate_brier_score(
        probabilities=[0.5, 0.5],
        outcomes=[1.0, 0.0],
    )

    assert score == pytest.approx(0.25)


def test_brier_score_rejects_empty_input() -> None:
    """Reject an evaluation without any predictions."""

    with pytest.raises(
        ValueError,
        match="At least one probability is required",
    ):
        calculate_brier_score(
            probabilities=[],
            outcomes=[],
        )


def test_brier_score_rejects_different_lengths() -> None:
    """Reject probabilities and outcomes with different lengths."""

    with pytest.raises(
        ValueError,
        match="must have equal length",
    ):
        calculate_brier_score(
            probabilities=[0.5],
            outcomes=[1.0, 0.0],
        )


def test_log_loss_is_near_zero_for_perfect_predictions() -> None:
    """Perfect predictions should have nearly zero log loss."""

    score = calculate_log_loss(
        probabilities=[1.0, 0.0],
        outcomes=[1.0, 0.0],
    )

    assert score == pytest.approx(
        0.0,
        abs=0.000000000001,
    )


def test_log_loss_matches_equal_probability_baseline() -> None:
    """Fifty-fifty predictions should have log loss near log two."""

    score = calculate_log_loss(
        probabilities=[0.5, 0.5],
        outcomes=[1.0, 0.0],
    )

    assert score == pytest.approx(
        0.693147,
        abs=0.000001,
    )


def test_confident_wrong_prediction_has_larger_log_loss() -> None:
    """A confident incorrect prediction should be penalized strongly."""

    uncertain_loss = calculate_log_loss(
        probabilities=[0.5],
        outcomes=[0.0],
    )
    confident_wrong_loss = calculate_log_loss(
        probabilities=[0.99],
        outcomes=[0.0],
    )

    assert confident_wrong_loss > uncertain_loss


def test_log_loss_rejects_invalid_probability() -> None:
    """Reject probability values outside the valid range."""

    with pytest.raises(
        ValueError,
        match="Probabilities must be between 0 and 1",
    ):
        calculate_log_loss(
            probabilities=[1.1],
            outcomes=[1.0],
        )


def test_accuracy_is_one_for_correct_winner_predictions() -> None:
    """Return perfect accuracy when every winner is predicted."""

    score = calculate_accuracy(
        probabilities=[0.7, 0.3],
        outcomes=[1.0, 0.0],
    )

    assert score == pytest.approx(1.0)


def test_accuracy_counts_incorrect_winner_predictions() -> None:
    """Include incorrect winner predictions in the accuracy."""

    score = calculate_accuracy(
        probabilities=[0.7, 0.7],
        outcomes=[1.0, 0.0],
    )

    assert score == pytest.approx(0.5)


def test_accuracy_excludes_tied_games() -> None:
    """Ignore tied NFL games when calculating winner accuracy."""

    score = calculate_accuracy(
        probabilities=[0.1, 0.7],
        outcomes=[0.5, 1.0],
    )

    assert score == pytest.approx(1.0)


def test_accuracy_rejects_only_tied_games() -> None:
    """Reject accuracy evaluation when every game is tied."""

    with pytest.raises(
        ValueError,
        match="At least one non-tied game is required",
    ):
        calculate_accuracy(
            probabilities=[0.5],
            outcomes=[0.5],
        )


def test_evaluate_probabilities_compares_model_to_baselines() -> None:
    """Return model metrics and simple baseline comparisons."""

    metrics = evaluate_probabilities(
        probabilities=[0.75, 0.25],
        outcomes=[1.0, 0.0],
    )

    assert metrics.game_count == 2
    assert metrics.tie_count == 0
    assert metrics.accuracy == pytest.approx(1.0)
    assert metrics.brier_score == pytest.approx(0.0625)
    assert metrics.log_loss == pytest.approx(
        0.287682,
        abs=0.000001,
    )
    assert (
        metrics.equal_probability_brier_score
        == pytest.approx(0.25)
    )
    assert (
        metrics.equal_probability_log_loss
        == pytest.approx(
            0.693147,
            abs=0.000001,
        )
    )
    assert metrics.home_team_accuracy == pytest.approx(0.5)


def test_load_evaluation_data_excludes_burn_in_season() -> None:
    """Exclude the first available Elo season from evaluation."""

    with duckdb.connect(":memory:") as connection:
        connection.execute("CREATE SCHEMA analytics")
        connection.execute(
            """
            CREATE TABLE analytics.elo_game_predictions (
                game_id VARCHAR,
                season INTEGER,
                gameday DATE,
                home_win_probability DOUBLE,
                actual_home_score DOUBLE
            )
            """
        )

        connection.executemany(
            """
            INSERT INTO analytics.elo_game_predictions
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (
                    "burn_in_game",
                    1999,
                    "1999-09-12",
                    0.57,
                    1.0,
                ),
                (
                    "evaluation_game",
                    2000,
                    "2000-09-03",
                    0.62,
                    1.0,
                ),
            ],
        )

        (
            seasons,
            probabilities,
            outcomes,
            evaluation_start_season,
        ) = load_evaluation_data(connection)

    assert probabilities == [pytest.approx(0.62)]
    assert outcomes == [pytest.approx(1.0)]
    assert evaluation_start_season == 2000


def test_calibration_groups_predictions_into_bins() -> None:
    """Calculate average predictions and outcomes by probability bin."""

    calibration_bins = calculate_calibration_bins(
        probabilities=[0.2, 0.4, 0.6, 0.8],
        outcomes=[0.0, 1.0, 1.0, 1.0],
        bin_count=2,
    )

    assert len(calibration_bins) == 2

    lower_bin = calibration_bins[0]

    assert lower_bin.lower_bound == pytest.approx(0.0)
    assert lower_bin.upper_bound == pytest.approx(0.5)
    assert lower_bin.game_count == 2
    assert lower_bin.average_probability == pytest.approx(0.3)
    assert (
        lower_bin.actual_home_result_rate
        == pytest.approx(0.5)
    )
    assert lower_bin.calibration_gap == pytest.approx(0.2)

    upper_bin = calibration_bins[1]

    assert upper_bin.lower_bound == pytest.approx(0.5)
    assert upper_bin.upper_bound == pytest.approx(1.0)
    assert upper_bin.game_count == 2
    assert upper_bin.average_probability == pytest.approx(0.7)
    assert (
        upper_bin.actual_home_result_rate
        == pytest.approx(1.0)
    )
    assert upper_bin.calibration_gap == pytest.approx(0.3)


def test_probability_one_uses_final_calibration_bin() -> None:
    """Place a probability of exactly one in the final bin."""

    calibration_bins = calculate_calibration_bins(
        probabilities=[1.0],
        outcomes=[1.0],
        bin_count=10,
    )

    assert len(calibration_bins) == 1
    assert calibration_bins[0].lower_bound == pytest.approx(0.9)
    assert calibration_bins[0].upper_bound == pytest.approx(1.0)


def test_evaluate_probabilities_by_season() -> None:
    """Calculate separate metrics for each NFL season."""

    season_results = evaluate_probabilities_by_season(
        seasons=[
            2024,
            2024,
            2025,
            2025,
        ],
        probabilities=[
            0.75,
            0.25,
            0.60,
            0.60,
        ],
        outcomes=[
            1.0,
            0.0,
            1.0,
            0.0,
        ],
    )

    assert len(season_results) == 2

    first_season = season_results[0]

    assert first_season.season == 2024
    assert first_season.metrics.game_count == 2
    assert first_season.metrics.accuracy == pytest.approx(1.0)
    assert (
        first_season.metrics.brier_score
        == pytest.approx(0.0625)
    )

    second_season = season_results[1]

    assert second_season.season == 2025
    assert second_season.metrics.game_count == 2
    assert second_season.metrics.accuracy == pytest.approx(0.5)
    assert (
        second_season.metrics.brier_score
        == pytest.approx(0.26)
    )


def test_season_evaluation_rejects_different_lengths() -> None:
    """Reject season evaluation inputs with different lengths."""

    with pytest.raises(
        ValueError,
        match="must have equal length",
    ):
        evaluate_probabilities_by_season(
            seasons=[2025],
            probabilities=[0.5, 0.6],
            outcomes=[1.0],
        )