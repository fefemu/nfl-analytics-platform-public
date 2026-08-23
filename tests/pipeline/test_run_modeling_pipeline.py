"""
Tests for the modeling data pipeline runner.
"""

from pathlib import Path

import pytest

from src.pipeline.run_modeling_pipeline import (
    ModelingPipelineStep,
    run_pipeline_steps,
    validate_database_file,
    validate_pipeline_steps,
    MODELING_PIPELINE_STEPS,
)


def test_run_pipeline_steps_uses_dependency_order(
    tmp_path: Path,
) -> None:
    """Run configured builders in their declared order."""

    calls: list[
        tuple[str, Path]
    ] = []

    database_file = tmp_path / "test.duckdb"
    database_file.touch()

    def first_builder(
        database_file: Path,
    ) -> None:
        calls.append(
            (
                "first",
                database_file,
            )
        )

    def second_builder(
        database_file: Path,
    ) -> None:
        calls.append(
            (
                "second",
                database_file,
            )
        )

    pipeline_steps = (
        ModelingPipelineStep(
            name="first",
            build_function=first_builder,
        ),
        ModelingPipelineStep(
            name="second",
            build_function=second_builder,
        ),
    )

    run_pipeline_steps(
        database_file=database_file,
        pipeline_steps=pipeline_steps,
    )

    assert calls == [
        (
            "first",
            database_file,
        ),
        (
            "second",
            database_file,
        ),
    ]


def test_run_pipeline_steps_stops_after_failure(
    tmp_path: Path,
) -> None:
    """Stop before downstream steps after a builder fails."""

    calls: list[str] = []

    database_file = tmp_path / "test.duckdb"
    database_file.touch()

    def failing_builder(
        database_file: Path,
    ) -> None:
        calls.append("failing")
        raise RuntimeError(
            "Expected builder failure."
        )

    def downstream_builder(
        database_file: Path,
    ) -> None:
        calls.append("downstream")

    pipeline_steps = (
        ModelingPipelineStep(
            name="failing",
            build_function=failing_builder,
        ),
        ModelingPipelineStep(
            name="downstream",
            build_function=downstream_builder,
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="Expected builder failure",
    ):
        run_pipeline_steps(
            database_file=database_file,
            pipeline_steps=pipeline_steps,
        )

    assert calls == ["failing"]


def test_validate_pipeline_steps_rejects_duplicate_names() -> None:
    """Reject ambiguous duplicate pipeline step names."""

    def builder(
        database_file: Path,
    ) -> None:
        pass

    pipeline_steps = (
        ModelingPipelineStep(
            name="duplicate",
            build_function=builder,
        ),
        ModelingPipelineStep(
            name="duplicate",
            build_function=builder,
        ),
    )

    with pytest.raises(
        ValueError,
        match="must be unique",
    ):
        validate_pipeline_steps(
            pipeline_steps
        )


def test_validate_pipeline_steps_rejects_empty_pipeline() -> None:
    """Reject an empty modeling pipeline."""

    with pytest.raises(
        ValueError,
        match="at least one step",
    ):
        validate_pipeline_steps(())


def test_validate_database_file_rejects_missing_file(
    tmp_path: Path,
) -> None:
    """Reject a missing DuckDB file."""

    database_file = tmp_path / "missing.duckdb"

    with pytest.raises(
        FileNotFoundError,
        match="does not exist",
    ):
        validate_database_file(
            database_file
        )


def test_default_pipeline_builds_dependencies_in_order(
) -> None:
    """Build all modeling dependencies in safe order."""

    step_names = [
        step.name
        for step in MODELING_PIPELINE_STEPS
    ]

    assert len(step_names) == 33

    assert step_names.index(
        "model_governance_reporting"
    ) < step_names.index(
        "historical_market_evaluation"
    ) < step_names.index(
        "current_game_predictions"
    )

    assert step_names.index(
        "external_nfelo_game_ratings"
    ) < step_names.index(
        "current_game_predictions"
    )

    assert step_names.index(
        "external_nfelo_game_ratings"
    ) < step_names.index(
        "current_spread_predictions"
    )

    assert step_names.index(
        "external_nfelo_game_ratings"
    ) < step_names.index(
        "current_season_simulation"
    )

    assert step_names.index(
        "game_schedule_features"
    ) < step_names.index(
        "game_modeling_dataset"
    )

    assert step_names.index(
        "game_schedule_features"
    ) < step_names.index(
        "game_weather_features"
    )

    assert step_names.index(
        "game_weather_features"
    ) < step_names.index(
        "game_modeling_dataset"
    )

    assert step_names.index(
        "game_scoring_environment_features"
    ) < step_names.index(
        "game_modeling_dataset"
    )

    assert step_names.index(
        "load_injury_reports"
    ) < step_names.index(
        "player_game_injury_status"
    )

    assert step_names.index(
        "load_depth_charts"
    ) < step_names.index(
        "legacy_player_game_depth_chart"
    )

    assert step_names.index(
        "load_depth_charts"
    ) < step_names.index(
        "espn_player_game_depth_chart"
    )

    assert step_names.index(
        "legacy_player_game_depth_chart"
    ) < step_names.index(
        "player_game_depth_chart"
    )

    assert step_names.index(
        "espn_player_game_depth_chart"
    ) < step_names.index(
        "player_game_depth_chart"
    )

    assert step_names.index(
        "load_snap_counts"
    ) < step_names.index(
        "player_game_snap_counts"
    )

    assert step_names.index(
        "load_player_directory"
    ) < step_names.index(
        "player_game_snap_counts"
    )

    assert step_names.index(
        "player_game_snap_counts"
    ) < step_names.index(
        "player_snap_share_history"
    )

    assert step_names.index(
        "player_game_injury_status"
    ) < step_names.index(
        "player_game_injury_context"
    )

    assert step_names.index(
        "player_game_depth_chart"
    ) < step_names.index(
        "player_game_injury_context"
    )

    assert step_names.index(
        "player_snap_share_history"
    ) < step_names.index(
        "player_game_injury_context"
    )

    assert step_names.index(
        "player_game_injury_context"
    ) < step_names.index(
        "player_injury_impact"
    )

    assert step_names.index(
        "player_injury_impact"
    ) < step_names.index(
        "team_game_injury_burden"
    )

    assert step_names.index(
        "team_game_injury_burden"
    ) < step_names.index(
        "game_injury_features"
    )

    assert step_names.index(
        "game_injury_features"
    ) < step_names.index(
        "game_modeling_dataset"
    )

    assert step_names.index(
        "game_modeling_dataset"
    ) < step_names.index(
        "modeling_game_splits"
    )

    assert step_names.index(
        "modeling_game_splits"
    ) < step_names.index(
        "model_governance_reporting"
    )

    assert step_names.index(
        "model_governance_reporting"
    ) < step_names.index(
        "current_game_predictions"
    )

    assert step_names.index(
        "elo_ratings"
    ) < step_names.index(
        "current_game_predictions"
    )

    assert step_names.index(
        "current_game_predictions"
    ) < step_names.index(
        "current_season_simulation"
    )

    assert step_names.index(
        "current_game_predictions"
    ) < step_names.index(
        "current_spread_predictions"
    )

    assert step_names.index(
        "current_spread_predictions"
    ) < step_names.index(
        "current_totals_predictions"
    )

    assert step_names.index(
        "game_modeling_dataset"
    ) < step_names.index(
        "current_totals_predictions"
    )

    assert step_names.index(
        "game_weather_features"
    ) < step_names.index(
        "current_totals_predictions"
    )

    assert step_names.index(
        "game_scoring_environment_features"
    ) < step_names.index(
        "current_totals_predictions"
    )

    assert step_names.index(
        "current_totals_predictions"
    ) < step_names.index(
        "current_game_score_predictions"
    )

    assert step_names.index(
        "current_spread_predictions"
    ) < step_names.index(
        "current_game_score_predictions"
    )

    assert step_names.index(
        "current_game_score_predictions"
    ) < step_names.index(
        "current_season_simulation"
    )

    assert step_names.index(
        "current_spread_predictions"
    ) < step_names.index(
        "current_season_simulation"
    )

    assert (
        step_names[-1]
        == "current_season_simulation"
    )
