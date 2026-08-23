"""Tests for consolidated external-model holdout evaluation."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.modeling.evaluate_external_model_upgrade_holdout as holdout_module
from src.modeling.evaluate_external_model_upgrade_holdout import (
    PAIRED_SUMMARY_COLUMNS,
    create_holdout_paired_summary,
    create_layer_paired_summary,
)


def create_probability_predictions() -> pd.DataFrame:
    """Create paired probability losses."""

    return pd.DataFrame(
        {
            "game_id": [
                "game_1",
                "game_2",
                "game_3",
            ],
            "current_brier_loss": [
                0.25,
                0.16,
                0.09,
            ],
            "external_brier_loss": [
                0.20,
                0.18,
                0.06,
            ],
        }
    )


def create_spread_predictions() -> pd.DataFrame:
    """Create paired Spread errors."""

    return pd.DataFrame(
        {
            "game_id": [
                "game_1",
                "game_2",
                "game_3",
            ],
            "current_absolute_error": [
                10.0,
                8.0,
                6.0,
            ],
            "external_absolute_error": [
                9.0,
                9.0,
                4.0,
            ],
        }
    )


def create_totals_predictions() -> pd.DataFrame:
    """Create paired Totals errors."""

    return pd.DataFrame(
        {
            "game_id": [
                "game_1",
                "game_2",
                "game_3",
            ],
            "current_absolute_error": [
                11.0,
                7.0,
                5.0,
            ],
            "external_absolute_error": [
                9.0,
                8.0,
                4.0,
            ],
        }
    )


def test_layer_summary_calculates_paired_delta():
    """Layer delta must equal external minus current."""

    current_losses = pd.Series(
        [
            10.0,
            8.0,
            6.0,
        ]
    )

    external_losses = pd.Series(
        [
            9.0,
            9.0,
            4.0,
        ]
    )

    result = create_layer_paired_summary(
        model_layer="SPREAD",
        loss_metric="ABSOLUTE_ERROR",
        current_losses=current_losses,
        external_losses=external_losses,
        bootstrap_iterations=500,
        random_seed=42,
    )

    expected_delta = float(
        (
            external_losses
            - current_losses
        ).mean()
    )

    assert result[
        "external_mean_loss_delta"
    ] == pytest.approx(expected_delta)

    assert result[
        "current_mean_loss"
    ] == pytest.approx(8.0)

    assert result[
        "external_mean_loss"
    ] == pytest.approx(
        22.0 / 3.0
    )

    assert result[
        "external_win_rate"
    ] == pytest.approx(
        2.0 / 3.0
    )

    assert result[
        "external_loss_rate"
    ] == pytest.approx(
        1.0 / 3.0
    )


def test_layer_summary_rejects_different_lengths():
    """Paired losses require identical lengths."""

    with pytest.raises(
        ValueError,
        match="identical lengths",
    ):
        create_layer_paired_summary(
            model_layer="TOTALS",
            loss_metric="ABSOLUTE_ERROR",
            current_losses=pd.Series(
                [
                    1.0,
                    2.0,
                ]
            ),
            external_losses=pd.Series(
                [
                    1.0,
                ]
            ),
            bootstrap_iterations=100,
            random_seed=42,
        )


def test_layer_summary_rejects_non_finite_losses():
    """Every holdout loss must be finite."""

    with pytest.raises(
        ValueError,
        match="must be finite",
    ):
        create_layer_paired_summary(
            model_layer="PROBABILITY",
            loss_metric="BRIER_SCORE",
            current_losses=pd.Series(
                [
                    0.2,
                    np.nan,
                ]
            ),
            external_losses=pd.Series(
                [
                    0.1,
                    0.2,
                ]
            ),
            bootstrap_iterations=100,
            random_seed=42,
        )


def test_combined_summary_contains_all_layers():
    """Probability, Spread and Totals are summarized."""

    summary = create_holdout_paired_summary(
        probability_predictions=(
            create_probability_predictions()
        ),
        spread_predictions=(
            create_spread_predictions()
        ),
        totals_predictions=(
            create_totals_predictions()
        ),
        bootstrap_iterations=500,
        random_seed=42,
    )

    assert tuple(summary.columns) == (
        PAIRED_SUMMARY_COLUMNS
    )

    assert set(
        summary["model_layer"]
    ) == {
        "PROBABILITY",
        "SPREAD",
        "TOTALS",
    }

    assert len(summary) == 3

    assert (
        summary["holdout_game_count"] == 3
    ).all()

    total_rate = (
        summary["external_win_rate"]
        + summary["external_loss_rate"]
        + summary["equal_loss_rate"]
    )

    np.testing.assert_allclose(
        total_rate,
        1.0,
    )


def test_combined_summary_requires_loss_columns():
    """Each layer must provide its paired losses."""

    invalid_probability = pd.DataFrame(
        {
            "current_brier_loss": [
                0.2,
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="Probability predictions "
        "are missing columns",
    ):
        create_holdout_paired_summary(
            probability_predictions=(
                invalid_probability
            ),
            spread_predictions=(
                create_spread_predictions()
            ),
            totals_predictions=(
                create_totals_predictions()
            ),
            bootstrap_iterations=100,
            random_seed=42,
        )


def test_orchestrator_loads_and_evaluates_every_layer(
    monkeypatch,
):
    """The single orchestrator wires every component."""

    calls: list[str] = []

    class DummyConnection:
        """Minimal DuckDB context manager."""

        def __enter__(self):
            calls.append("connection_enter")
            return self

        def __exit__(
            self,
            exception_type,
            exception,
            traceback,
        ):
            calls.append("connection_exit")
            return False

    probability_data = pd.DataFrame(
        {
            "source": [
                "probability",
            ]
        }
    )

    spread_data = pd.DataFrame(
        {
            "source": [
                "spread",
            ]
        }
    )

    totals_data = pd.DataFrame(
        {
            "source": [
                "totals",
            ]
        }
    )

    probability_summary = pd.DataFrame(
        {
            "candidate_name": [
                "probability",
            ]
        }
    )

    spread_summary = pd.DataFrame(
        {
            "candidate_name": [
                "spread",
            ]
        }
    )

    totals_summary = pd.DataFrame(
        {
            "candidate_name": [
                "totals",
            ]
        }
    )

    probability_predictions = (
        create_probability_predictions()
    )

    spread_predictions = (
        create_spread_predictions()
    )

    totals_predictions = (
        create_totals_predictions()
    )

    paired_summary = pd.DataFrame(
        {
            "model_layer": [
                "PROBABILITY",
                "SPREAD",
                "TOTALS",
            ]
        }
    )

    monkeypatch.setattr(
        holdout_module,
        "validate_database_file",
        lambda database_file: calls.append(
            "validate_database"
        ),
    )

    monkeypatch.setattr(
        holdout_module.duckdb,
        "connect",
        lambda *args, **kwargs: (
            DummyConnection()
        ),
    )

    monkeypatch.setattr(
        holdout_module,
        "load_probability_holdout_data",
        lambda connection: (
            calls.append(
                "load_probability"
            )
            or probability_data
        ),
    )

    monkeypatch.setattr(
        holdout_module,
        "load_spread_holdout_data",
        lambda connection: (
            calls.append("load_spread")
            or spread_data
        ),
    )

    monkeypatch.setattr(
        holdout_module,
        "load_totals_holdout_data",
        lambda connection: (
            calls.append("load_totals")
            or totals_data
        ),
    )

    monkeypatch.setattr(
        holdout_module,
        "evaluate_locked_probability_holdout",
        lambda data: (
            calls.append(
                "evaluate_probability"
            )
            or (
                probability_summary,
                probability_predictions,
            )
        ),
    )

    monkeypatch.setattr(
        holdout_module,
        "evaluate_locked_spread_holdout",
        lambda data: (
            calls.append("evaluate_spread")
            or (
                spread_summary,
                spread_predictions,
            )
        ),
    )

    monkeypatch.setattr(
        holdout_module,
        "evaluate_locked_totals_routing_holdout",
        lambda data: (
            calls.append("evaluate_totals")
            or (
                totals_summary,
                totals_predictions,
            )
        ),
    )

    monkeypatch.setattr(
        holdout_module,
        "create_holdout_paired_summary",
        lambda **kwargs: (
            calls.append("create_paired")
            or paired_summary
        ),
    )

    results = (
        holdout_module
        .run_external_model_upgrade_holdout(
            database_file=Path(
                "synthetic.duckdb"
            )
        )
    )

    assert results == (
        probability_summary,
        spread_summary,
        totals_summary,
        paired_summary,
    )

    assert calls == [
        "validate_database",
        "connection_enter",
        "load_probability",
        "load_spread",
        "load_totals",
        "connection_exit",
        "evaluate_probability",
        "evaluate_spread",
        "evaluate_totals",
        "create_paired",
    ]