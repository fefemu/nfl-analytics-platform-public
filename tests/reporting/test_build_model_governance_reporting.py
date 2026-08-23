"""Tests for model governance reporting tables."""

import duckdb
import pandas as pd
import pytest

from src.reporting.build_model_governance_reporting import (
    BLEND_SCORECARD_TABLE,
    BLEND_WEIGHT_TABLE,
    GOVERNANCE_SCORECARD_TABLE,
    GOVERNANCE_SEASON_TABLE,
    PRODUCTION_REGISTRY_TABLE,
    persist_reporting_frames,
    validate_reporting_tables,
)


def create_reporting_frames() -> dict[
    str,
    pd.DataFrame,
]:
    """Create deterministic reporting frames."""

    return {
        GOVERNANCE_SCORECARD_TABLE: pd.DataFrame(
            {
                "value": range(5),
            }
        ),
        GOVERNANCE_SEASON_TABLE: pd.DataFrame(
            {
                "value": range(30),
            }
        ),
        BLEND_WEIGHT_TABLE: pd.DataFrame(
            {
                "value": range(42),
            }
        ),
        BLEND_SCORECARD_TABLE: pd.DataFrame(
            {
                "value": range(9),
            }
        ),
        PRODUCTION_REGISTRY_TABLE: pd.DataFrame(
            [
                {
                    "model_name": (
                        "elo_injury_logistic_blend"
                    ),
                    "model_version": "0.2.0",
                    "deployment_status": (
                        "selected_for_2026_forward_test"
                    ),
                    "logistic_weight": 0.70,
                    "elo_weight": 0.30,
                    "forward_test_season": 2026,
                }
            ]
        ),
    }


def test_persist_reporting_frames_creates_tables(
) -> None:
    """Persist every dashboard-ready reporting table."""

    with duckdb.connect(
        ":memory:"
    ) as connection:
        persist_reporting_frames(
            connection=connection,
            reporting_frames=(
                create_reporting_frames()
            ),
        )

        counts = validate_reporting_tables(
            connection
        )

    assert counts == {
        GOVERNANCE_SCORECARD_TABLE: 5,
        GOVERNANCE_SEASON_TABLE: 30,
        BLEND_WEIGHT_TABLE: 42,
        BLEND_SCORECARD_TABLE: 9,
        PRODUCTION_REGISTRY_TABLE: 1,
    }


def test_persist_reporting_frames_rejects_empty_frame(
) -> None:
    """Reject an empty reporting output."""

    frames = create_reporting_frames()

    frames[
        GOVERNANCE_SCORECARD_TABLE
    ] = pd.DataFrame()

    with duckdb.connect(
        ":memory:"
    ) as connection:
        with pytest.raises(
            RuntimeError,
            match="Reporting frame is empty",
        ):
            persist_reporting_frames(
                connection=connection,
                reporting_frames=frames,
            )


def test_validate_reporting_tables_rejects_registry_weights(
) -> None:
    """Reject production blend weights not summing to one."""

    frames = create_reporting_frames()

    frames[
        PRODUCTION_REGISTRY_TABLE
    ].loc[
        0,
        "elo_weight",
    ] = 0.40

    with duckdb.connect(
        ":memory:"
    ) as connection:
        persist_reporting_frames(
            connection=connection,
            reporting_frames=frames,
        )

        with pytest.raises(
            RuntimeError,
            match="registry is inconsistent",
        ):
            validate_reporting_tables(
                connection
            )