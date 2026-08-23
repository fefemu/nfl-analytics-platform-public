"""Tests for the current NFL odds pipeline runner."""

from pathlib import Path

import pytest

from src.pipeline.run_odds_pipeline import run_odds_pipeline


def test_run_odds_pipeline_calls_steps_in_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Run all odds pipeline steps in the correct order."""

    calls: list[str] = []
    snapshot_file = tmp_path / "odds_snapshot.json"

    def fake_download() -> Path:
        calls.append("download")
        return snapshot_file

    def fake_raw_load(
        snapshot_file: Path,
    ) -> None:
        calls.append("raw")

    def fake_processed_build() -> None:
        calls.append("processed")

    def fake_best_odds_build() -> None:
        calls.append("best")

    def fake_event_bridge_build() -> None:
        calls.append("bridge")

    def fake_market_board_build() -> None:
        calls.append("market_board")

    def fake_moneyline() -> None:
        calls.append("moneyline")

    def fake_spread() -> None:
        calls.append("spread")

    def fake_totals() -> None:
        calls.append("totals")

    def fake_board() -> None:
        calls.append("board")

    monkeypatch.setattr(
        "src.pipeline.run_odds_pipeline."
        "save_current_nfl_odds_snapshot",
        fake_download,
    )
    monkeypatch.setattr(
        "src.pipeline.run_odds_pipeline."
        "load_odds_snapshot_to_duckdb",
        fake_raw_load,
    )
    monkeypatch.setattr(
        "src.pipeline.run_odds_pipeline."
        "build_processed_odds",
        fake_processed_build,
    )
    monkeypatch.setattr(
        "src.pipeline.run_odds_pipeline."
        "build_best_odds",
        fake_best_odds_build,
    )
    monkeypatch.setattr(
        "src.pipeline.run_odds_pipeline."
        "build_odds_event_bridge",
        fake_event_bridge_build,
    )
    monkeypatch.setattr(
        "src.pipeline.run_odds_pipeline."
        "build_current_market_board",
        fake_market_board_build,
    )
    monkeypatch.setattr("src.pipeline.run_odds_pipeline.build_current_moneyline_value", fake_moneyline)
    monkeypatch.setattr("src.pipeline.run_odds_pipeline.build_current_spread_value", fake_spread)
    monkeypatch.setattr("src.pipeline.run_odds_pipeline.build_current_totals_value", fake_totals)
    monkeypatch.setattr("src.pipeline.run_odds_pipeline.build_current_betting_board", fake_board)

    run_odds_pipeline()

    assert calls == [
        "download",
        "raw",
        "processed",
        "best",
        "bridge",
        "market_board",
        "moneyline",
        "spread",
        "totals",
        "board",
    ]


@pytest.mark.parametrize(
    ("failure_step", "expected_calls"),
    [
        ("download", ["download"]),
        ("raw", ["download", "raw"]),
        (
            "processed",
            ["download", "raw", "processed"],
        ),
        (
            "best",
            ["download", "raw", "processed", "best"],
        ),
        (
            "bridge",
            [
                "download",
                "raw",
                "processed",
                "best",
                "bridge",
            ],
        ),
        (
            "market_board",
            [
                "download",
                "raw",
                "processed",
                "best",
                "bridge",
                "market_board",
            ],
        ),
        ("moneyline", ["download", "raw", "processed", "best", "bridge", "market_board", "moneyline"]),
        ("spread", ["download", "raw", "processed", "best", "bridge", "market_board", "moneyline", "spread"]),
        ("totals", ["download", "raw", "processed", "best", "bridge", "market_board", "moneyline", "spread", "totals"]),
        ("board", ["download", "raw", "processed", "best", "bridge", "market_board", "moneyline", "spread", "totals", "board"]),
    ],
)
def test_run_odds_pipeline_stops_and_propagates_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure_step: str,
    expected_calls: list[str],
) -> None:
    """Stop at the failing step and propagate its error."""

    calls: list[str] = []
    snapshot_file = tmp_path / "odds_snapshot.json"

    def record_step(step_name: str) -> None:
        calls.append(step_name)

        if failure_step == step_name:
            raise RuntimeError(
                f"{step_name} failed."
            )

    def fake_download() -> Path:
        record_step("download")
        return snapshot_file

    def fake_raw_load(
        snapshot_file: Path,
    ) -> None:
        record_step("raw")

    def fake_processed_build() -> None:
        record_step("processed")

    def fake_best_odds_build() -> None:
        record_step("best")

    def fake_event_bridge_build() -> None:
        record_step("bridge")

    def fake_market_board_build() -> None:
        record_step("market_board")

    def fake_moneyline() -> None:
        record_step("moneyline")

    def fake_spread() -> None:
        record_step("spread")

    def fake_totals() -> None:
        record_step("totals")

    def fake_board() -> None:
        record_step("board")

    monkeypatch.setattr(
        "src.pipeline.run_odds_pipeline."
        "save_current_nfl_odds_snapshot",
        fake_download,
    )
    monkeypatch.setattr(
        "src.pipeline.run_odds_pipeline."
        "load_odds_snapshot_to_duckdb",
        fake_raw_load,
    )
    monkeypatch.setattr(
        "src.pipeline.run_odds_pipeline."
        "build_processed_odds",
        fake_processed_build,
    )
    monkeypatch.setattr(
        "src.pipeline.run_odds_pipeline."
        "build_best_odds",
        fake_best_odds_build,
    )
    monkeypatch.setattr(
        "src.pipeline.run_odds_pipeline."
        "build_odds_event_bridge",
        fake_event_bridge_build,
    )
    monkeypatch.setattr(
        "src.pipeline.run_odds_pipeline."
        "build_current_market_board",
        fake_market_board_build,
    )
    monkeypatch.setattr("src.pipeline.run_odds_pipeline.build_current_moneyline_value", fake_moneyline)
    monkeypatch.setattr("src.pipeline.run_odds_pipeline.build_current_spread_value", fake_spread)
    monkeypatch.setattr("src.pipeline.run_odds_pipeline.build_current_totals_value", fake_totals)
    monkeypatch.setattr("src.pipeline.run_odds_pipeline.build_current_betting_board", fake_board)

    with pytest.raises(
        RuntimeError,
        match=f"{failure_step} failed.",
    ):
        run_odds_pipeline()

    assert calls == expected_calls
