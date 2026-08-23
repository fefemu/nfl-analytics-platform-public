"""
NFL Analytics Platform
Modeling Data Pipeline Runner

Purpose:
    Rebuild the complete DuckDB modeling-data chain
    in a validated dependency order.

Author:
    Ferenc Kaizer

Version:
    0.1.0
"""

import gc
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from src.analytics.build_elo_ratings import (
    build_elo_ratings,
)
from src.analytics.build_game_injury_features import (
    build_game_injury_features,
)
from src.analytics.build_game_qb_features import (
    build_game_qb_features,
)
from src.analytics.build_game_schedule_features import (
    build_game_schedule_features,
)
from src.analytics.build_player_game_injury_context import (
    build_player_game_injury_context,
)
from src.analytics.build_player_injury_impact import (
    build_player_injury_impact,
)
from src.analytics.build_player_snap_share_history import (
    build_player_snap_share_history,
)
from src.analytics.build_qb_ratings import (
    build_qb_ratings,
)
from src.analytics.build_rolling_team_features import (
    build_rolling_team_features,
)
from src.analytics.build_team_game_injury_burden import (
    build_team_game_injury_burden,
)
from src.analytics.build_game_scoring_environment import (
    build_game_scoring_environment,
)
from src.analytics.build_game_weather_features import (
    build_game_weather_features,
)
from src.betting.build_historical_market_evaluation import (
    build_historical_market_evaluation,
)
from src.modeling.build_current_game_predictions import (
    build_current_game_predictions,
)
from src.modeling.build_current_game_score_predictions import (
    build_current_game_score_predictions,
)
from src.modeling.build_current_spread_predictions import (
    build_current_spread_predictions,
)
from src.modeling.build_current_totals_predictions import (
    build_current_totals_predictions,
)
from src.modeling.build_game_modeling_dataset import (
    build_game_modeling_dataset,
)
from src.modeling.build_modeling_splits import (
    build_modeling_splits,
)
from src.processing.build_espn_player_game_depth_chart import (
    build_espn_player_game_depth_chart,
)
from src.processing.build_external_nfelo_game_ratings import (
    build_external_nfelo_game_ratings,
)
from src.processing.build_legacy_player_game_depth_chart import (
    build_legacy_player_game_depth_chart,
)
from src.processing.build_player_game_depth_chart import (
    build_player_game_depth_chart,
)
from src.processing.build_player_game_injury_status import (
    build_player_game_injury_status,
)
from src.processing.build_player_game_snap_counts import (
    build_player_game_snap_counts,
)
from src.processing.build_qb_game_performance import (
    build_qb_game_performance,
)
from src.processing.build_team_game_efficiency import (
    build_team_game_efficiency,
)
from src.processing.load_depth_charts_to_duckdb import (
    load_depth_charts_to_duckdb,
)
from src.processing.load_injury_reports_to_duckdb import (
    load_injury_reports_to_duckdb,
)
from src.processing.load_player_directory_to_duckdb import (
    load_player_directory_to_duckdb,
)
from src.processing.load_snap_counts_to_duckdb import (
    load_snap_counts_to_duckdb,
)
from src.simulation.build_current_season_simulation import (
    build_current_season_simulation,
)
from src.reporting.build_model_governance_reporting import (
    build_model_governance_reporting,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATABASE_FILE = PROJECT_ROOT / "data" / "nfl_analytics.duckdb"

BuildFunction = Callable[..., None]


@dataclass(frozen=True)
class ModelingPipelineStep:
    """Describe one ordered modeling pipeline step."""

    name: str
    build_function: BuildFunction


MODELING_PIPELINE_STEPS = (
    ModelingPipelineStep(
        name="external_nfelo_game_ratings",
        build_function=build_external_nfelo_game_ratings,
    ),
    ModelingPipelineStep(
        name="elo_ratings",
        build_function=build_elo_ratings,
    ),
    ModelingPipelineStep(
        name="team_game_efficiency",
        build_function=build_team_game_efficiency,
    ),
    ModelingPipelineStep(
        name="rolling_team_features",
        build_function=build_rolling_team_features,
    ),
    ModelingPipelineStep(
        name="qb_game_performance",
        build_function=build_qb_game_performance,
    ),
    ModelingPipelineStep(
        name="qb_ratings",
        build_function=build_qb_ratings,
    ),
    ModelingPipelineStep(
        name="game_qb_features",
        build_function=build_game_qb_features,
    ),
    ModelingPipelineStep(
        name="game_schedule_features",
        build_function=build_game_schedule_features,
    ),
    ModelingPipelineStep(
        name="game_weather_features",
        build_function=build_game_weather_features,
    ),
    ModelingPipelineStep(
        name="game_scoring_environment_features",
        build_function=build_game_scoring_environment,
    ),
    ModelingPipelineStep(
        name="load_injury_reports",
        build_function=load_injury_reports_to_duckdb,
    ),
    ModelingPipelineStep(
        name="player_game_injury_status",
        build_function=build_player_game_injury_status,
    ),
    ModelingPipelineStep(
        name="load_depth_charts",
        build_function=load_depth_charts_to_duckdb,
    ),
    ModelingPipelineStep(
        name="legacy_player_game_depth_chart",
        build_function=(
            build_legacy_player_game_depth_chart
        ),
    ),
    ModelingPipelineStep(
        name="espn_player_game_depth_chart",
        build_function=(
            build_espn_player_game_depth_chart
        ),
    ),
    ModelingPipelineStep(
        name="player_game_depth_chart",
        build_function=build_player_game_depth_chart,
    ),
    ModelingPipelineStep(
        name="load_snap_counts",
        build_function=load_snap_counts_to_duckdb,
    ),
    ModelingPipelineStep(
        name="load_player_directory",
        build_function=load_player_directory_to_duckdb,
    ),
    ModelingPipelineStep(
        name="player_game_snap_counts",
        build_function=build_player_game_snap_counts,
    ),
    ModelingPipelineStep(
        name="player_snap_share_history",
        build_function=build_player_snap_share_history,
    ),
    ModelingPipelineStep(
        name="player_game_injury_context",
        build_function=build_player_game_injury_context,
    ),
    ModelingPipelineStep(
        name="player_injury_impact",
        build_function=build_player_injury_impact,
    ),
    ModelingPipelineStep(
        name="team_game_injury_burden",
        build_function=build_team_game_injury_burden,
    ),
    ModelingPipelineStep(
        name="game_injury_features",
        build_function=build_game_injury_features,
    ),
    ModelingPipelineStep(
        name="game_modeling_dataset",
        build_function=build_game_modeling_dataset,
    ),
    ModelingPipelineStep(
        name="modeling_game_splits",
        build_function=build_modeling_splits,
    ),
    ModelingPipelineStep(
        name="model_governance_reporting",
        build_function=(
            build_model_governance_reporting
        ),
    ),
    ModelingPipelineStep(
        name="historical_market_evaluation",
        build_function=build_historical_market_evaluation,
    ),
    ModelingPipelineStep(
        name="current_game_predictions",
        build_function=(
            build_current_game_predictions
        ),
    ),
    ModelingPipelineStep(
        name="current_spread_predictions",
        build_function=(
            build_current_spread_predictions
        ),
    ),
    ModelingPipelineStep(
        name="current_totals_predictions",
        build_function=(
            build_current_totals_predictions
        ),
    ),
    ModelingPipelineStep(
        name="current_game_score_predictions",
        build_function=build_current_game_score_predictions,
    ),
    ModelingPipelineStep(
        name="current_season_simulation",
        build_function=(
            build_current_season_simulation
        ),
    ),
)


def validate_database_file(
    database_file: Path,
) -> None:
    """Validate the pipeline DuckDB file."""

    if not database_file.exists():
        raise FileNotFoundError(
            f"Database file does not exist: {database_file}"
        )

    if not database_file.is_file():
        raise RuntimeError(
            f"Database path is not a file: {database_file}"
        )


def validate_pipeline_steps(
    pipeline_steps: tuple[
        ModelingPipelineStep, ...
    ],
) -> None:
    """Validate pipeline step configuration."""

    if not pipeline_steps:
        raise ValueError(
            "Modeling pipeline must contain at least one step."
        )

    step_names = [
        step.name
        for step in pipeline_steps
    ]

    if any(
        not step_name.strip()
        for step_name in step_names
    ):
        raise ValueError(
            "Modeling pipeline step names must not be empty."
        )

    if len(step_names) != len(set(step_names)):
        raise ValueError(
            "Modeling pipeline step names must be unique."
        )


def run_pipeline_steps(
    database_file: Path,
    pipeline_steps: tuple[
        ModelingPipelineStep, ...
    ] = MODELING_PIPELINE_STEPS,
) -> None:
    """Run modeling builders in dependency order."""

    validate_pipeline_steps(pipeline_steps)

    pipeline_started_at = perf_counter()
    total_steps = len(pipeline_steps)

    for step_number, step in enumerate(
        pipeline_steps,
        start=1,
    ):
        step_started_at = perf_counter()

        logger.info(
            "Modeling pipeline step %s/%s started: %s",
            step_number,
            total_steps,
            step.name,
        )

        try:
            step.build_function(
                database_file=database_file,
            )

        except Exception:
            logger.exception(
                "Modeling pipeline step %s/%s failed: %s",
                step_number,
                total_steps,
                step.name,
            )
            raise

        collected_objects = gc.collect()

        logger.debug(
            "Released unused objects after step %s: %s",
            step.name,
            collected_objects,
        )

        step_duration = (
            perf_counter() - step_started_at
        )

        logger.info(
            "Modeling pipeline step %s/%s completed: "
            "%s | %.2f seconds",
            step_number,
            total_steps,
            step.name,
            step_duration,
        )

    pipeline_duration = (
        perf_counter() - pipeline_started_at
    )

    logger.info(
        "All %s modeling pipeline steps completed "
        "successfully in %.2f seconds.",
        total_steps,
        pipeline_duration,
    )


def run_modeling_pipeline(
    database_file: Path = DATABASE_FILE,
) -> None:
    """Run the complete modeling-data pipeline."""

    validate_database_file(database_file)

    logger.info(
        "Starting modeling data pipeline..."
    )

    logger.info(
        "DuckDB writers such as DBeaver must be "
        "disconnected before this pipeline runs."
    )

    run_pipeline_steps(
        database_file=database_file,
    )

    logger.info(
        "Modeling data pipeline completed successfully."
    )


def main() -> None:
    """Run the complete modeling-data pipeline."""

    try:
        run_modeling_pipeline()

    except Exception:
        logger.exception(
            "Modeling data pipeline failed."
        )
        raise


if __name__ == "__main__":
    main()
