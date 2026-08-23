from pathlib import Path

import duckdb
import pandas as pd

from src.dashboard.repository import CORE_TABLES, DashboardHealth, DashboardRepository


def test_missing_database_returns_unavailable(tmp_path: Path) -> None:
    health = DashboardRepository(tmp_path / "missing.duckdb").health()
    assert not health.database_available
    assert health.missing_core_tables == CORE_TABLES


def test_health_reports_missing_and_refresh_state(tmp_path: Path) -> None:
    database = tmp_path / "dashboard.duckdb"
    with duckdb.connect(str(database)) as connection:
        connection.execute("CREATE SCHEMA analytics")
        connection.execute("CREATE TABLE analytics.current_game_predictions AS SELECT 1 AS game_id")
        connection.execute(
            """
            CREATE TABLE analytics.refresh_run_history AS
            SELECT 'SUCCESS' AS status,
                   TIMESTAMPTZ '2026-08-14 12:00:00+00:00' AS completed_at,
                   TIMESTAMPTZ '2026-08-14 11:59:00+00:00' AS started_at
            """
        )
    health = DashboardRepository(database).health()
    assert health.database_available
    assert health.latest_refresh_status == "SUCCESS"
    assert "current_game_predictions" in health.available_tables
    assert "current_game_spread_predictions" in health.missing_core_tables


def test_health_falls_back_to_latest_production_output_time(tmp_path: Path) -> None:
    database = tmp_path / "dashboard.duckdb"
    with duckdb.connect(str(database)) as connection:
        connection.execute("CREATE SCHEMA analytics")
        connection.execute(
            """
            CREATE TABLE analytics.current_game_predictions AS
            SELECT TIMESTAMPTZ '2026-08-14 14:00:00+00:00'
                AS prediction_generated_at
            """
        )
        connection.execute(
            """
            CREATE TABLE analytics.current_season_simulation_summary AS
            SELECT TIMESTAMPTZ '2026-08-14 14:05:00+00:00'
                AS simulation_generated_at
            """
        )

    health = DashboardRepository(database).health()

    assert health.latest_refresh_status == "MODEL_BUILD"
    assert health.latest_refresh_at == pd.Timestamp("2026-08-14 14:05:00+00:00")


def test_read_table_uses_read_only_validated_identifier(tmp_path: Path) -> None:
    database = tmp_path / "dashboard.duckdb"
    with duckdb.connect(str(database)) as connection:
        connection.execute("CREATE SCHEMA analytics")
        connection.execute("CREATE TABLE analytics.example AS SELECT 7 AS value")
    repository = DashboardRepository(database)
    assert repository.read_table("example", ("value",)).iloc[0]["value"] == 7


def test_game_center_adds_optional_narrative(monkeypatch, tmp_path: Path) -> None:
    repository = DashboardRepository(tmp_path / "unused.duckdb")
    games = pd.DataFrame({"game_id": ["game"], "home_team": ["KC"]})
    narratives = pd.DataFrame({
        "game_id": ["game"],
        "headline_en": ["KC has the edge."],
        "headline_hu": ["A KC az esélyesebb."],
        "summary_en": ["Summary"],
        "summary_hu": ["Összefoglaló"],
        "model_context_en": ["Context"],
        "model_context_hu": ["Kontextus"],
        "top_factor_en": ["Elo"],
        "top_factor_hu": ["Elo"],
    })
    health = DashboardHealth(
        True,
        ("current_game_prediction_narratives",),
        (),
        "SUCCESS",
        None,
    )
    monkeypatch.setattr(repository, "load_weekly_games", lambda: games)
    monkeypatch.setattr(repository, "health", lambda: health)
    monkeypatch.setattr(repository, "read_table", lambda *args, **kwargs: narratives)

    result = repository.load_game_center_games()

    assert result.iloc[0]["headline_en"] == "KC has the edge."
    assert len(result) == 1


def test_season_simulator_loads_optional_benchmark(monkeypatch, tmp_path: Path) -> None:
    repository = DashboardRepository(tmp_path / "unused.duckdb")
    health = DashboardHealth(
        True,
        (
            "current_season_simulation_summary",
            "current_season_win_distribution",
            "current_season_elo_benchmark_team_comparison",
        ),
        (),
        "SUCCESS",
        None,
    )
    frames = {
        "current_season_simulation_summary": pd.DataFrame({"team": ["KC"]}),
        "current_season_win_distribution": pd.DataFrame({"team": ["KC"]}),
        "current_season_elo_benchmark_team_comparison": pd.DataFrame({"team": ["KC"]}),
    }
    monkeypatch.setattr(repository, "health", lambda: health)
    monkeypatch.setattr(repository, "read_table", lambda table, *args: frames[table])

    summary, distribution, benchmark = repository.load_season_simulator()

    assert summary.iloc[0]["team"] == "KC"
    assert distribution.iloc[0]["team"] == "KC"
    assert benchmark.iloc[0]["team"] == "KC"


def test_data_science_lab_loads_aggregated_impact_summary(tmp_path: Path) -> None:
    database = tmp_path / "dashboard.duckdb"
    with duckdb.connect(str(database)) as connection:
        connection.execute("CREATE SCHEMA analytics")
        connection.execute(
            """
            CREATE TABLE analytics.game_modeling_dataset AS
            SELECT 24 AS target_home_score, 20 AS target_away_score,
                   TRUE AS target_home_win, 4 AS target_point_differential,
                   44 AS target_total_points, 10 AS home_rest_days,
                   7 AS away_rest_days, TRUE AS home_post_bye,
                   FALSE AS away_post_bye, 2.0 AS home_listed_qb_rating,
                   1.0 AS away_listed_qb_rating,
                   1.0 AS home_total_injury_burden,
                   3.5 AS away_total_injury_burden,
                   TRUE AS is_indoor, FALSE AS is_freezing,
                   FALSE AS is_high_wind, FALSE AS is_weather_exposed
            """
        )

    impacts = DashboardRepository(database).load_data_science_lab()[
        "historical_impact_summary"
    ]

    assert set(impacts["topic"]) == {
        "home_field", "rest", "qb", "injury", "weather"
    }
    home = impacts.loc[
        (impacts["topic"] == "home_field")
        & (impacts["segment"] == "all_games")
    ].iloc[0]
    assert home["game_count"] == 1
    assert home["win_rate"] == 1.0
