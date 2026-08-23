"""Tests for the schedule pipeline runner."""

import pytest

from src.pipeline.run_schedule_pipeline import run_schedule_pipeline


def test_run_schedule_pipeline_calls_steps_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run the raw loader before the processed schedule builder."""

    calls: list[str] = []

    def fake_load_schedule_to_duckdb() -> None:
        calls.append("raw")

    def fake_build_processed_schedule() -> None:
        calls.append("processed")

    monkeypatch.setattr(
        "src.pipeline.run_schedule_pipeline."
        "load_schedule_to_duckdb",
        fake_load_schedule_to_duckdb,
    )
    monkeypatch.setattr(
        "src.pipeline.run_schedule_pipeline."
        "build_processed_schedule",
        fake_build_processed_schedule,
    )

    run_schedule_pipeline()

    assert calls == ["raw", "processed"]


def test_run_schedule_pipeline_stops_when_raw_load_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not build processed data when the raw loading step fails."""

    processed_was_called = False

    def fake_load_schedule_to_duckdb() -> None:
        raise RuntimeError("Raw schedule loading failed.")

    def fake_build_processed_schedule() -> None:
        nonlocal processed_was_called
        processed_was_called = True

    monkeypatch.setattr(
        "src.pipeline.run_schedule_pipeline."
        "load_schedule_to_duckdb",
        fake_load_schedule_to_duckdb,
    )
    monkeypatch.setattr(
        "src.pipeline.run_schedule_pipeline."
        "build_processed_schedule",
        fake_build_processed_schedule,
    )

    with pytest.raises(
        RuntimeError,
        match="Raw schedule loading failed.",
    ):
        run_schedule_pipeline()


def test_run_schedule_pipeline_propagates_processed_build_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Propagate an error raised by the processed schedule builder."""

    calls: list[str] = []

    def fake_load_schedule_to_duckdb() -> None:
        calls.append("raw")

    def fake_build_processed_schedule() -> None:
        calls.append("processed")
        raise RuntimeError("Processed schedule build failed.")

    monkeypatch.setattr(
        "src.pipeline.run_schedule_pipeline."
        "load_schedule_to_duckdb",
        fake_load_schedule_to_duckdb,
    )
    monkeypatch.setattr(
        "src.pipeline.run_schedule_pipeline."
        "build_processed_schedule",
        fake_build_processed_schedule,
    )

    with pytest.raises(
        RuntimeError,
        match="Processed schedule build failed.",
    ):
        run_schedule_pipeline()

    assert calls == ["raw", "processed"]