"""Build a compact, read-only DuckDB snapshot for the public dashboard."""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import duckdb

from src.deployment.dashboard_database import DATABASE_FILE, PROJECT_ROOT


LOGGER = logging.getLogger(__name__)
DEFAULT_OUTPUT_FILE = PROJECT_ROOT / "data" / "deployment" / "dashboard.duckdb"


@dataclass(frozen=True)
class SnapshotTable:
    schema: str
    name: str
    required: bool = True

    @property
    def qualified_name(self) -> str:
        return f"{self.schema}.{self.name}"


# This manifest is the public dashboard's explicit data contract. Keeping it
# here prevents a deployment artifact from silently growing with the dev DB.
DASHBOARD_TABLES = (
    SnapshotTable("analytics", "current_game_predictions"),
    SnapshotTable("analytics", "current_game_spread_predictions"),
    SnapshotTable("analytics", "current_game_total_predictions"),
    SnapshotTable("analytics", "current_game_score_predictions"),
    SnapshotTable("analytics", "current_game_prediction_narratives"),
    SnapshotTable("analytics", "current_betting_board"),
    SnapshotTable("analytics", "current_season_simulation_summary"),
    SnapshotTable("analytics", "current_season_win_distribution"),
    SnapshotTable(
        "analytics", "current_season_elo_benchmark_team_comparison"
    ),
    SnapshotTable("analytics", "production_model_registry"),
    SnapshotTable("analytics", "model_governance_scorecard"),
    SnapshotTable("analytics", "model_governance_season_results"),
    SnapshotTable("analytics", "model_blend_scorecard"),
    SnapshotTable("analytics", "game_modeling_dataset"),
    SnapshotTable("analytics", "refresh_run_history", required=False),
    SnapshotTable("analytics", "current_elo_ratings", required=False),
    SnapshotTable("raw", "depth_charts_espn"),
    SnapshotTable("raw", "player_directory"),
    SnapshotTable("raw", "injury_reports"),
    SnapshotTable("processed", "external_nfelo_game_ratings"),
    SnapshotTable("processed", "schedule"),
    SnapshotTable(
        "processed", "external_nfelounits_units", required=False
    ),
)


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _quote_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _table_exists(
    connection: duckdb.DuckDBPyConnection,
    catalog: str,
    table: SnapshotTable,
) -> bool:
    return bool(
        connection.execute(
            """
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_catalog = ?
              AND table_schema = ?
              AND table_name = ?
            """,
            [catalog, table.schema, table.name],
        ).fetchone()[0]
    )


def validate_source_tables(
    connection: duckdb.DuckDBPyConnection,
) -> tuple[SnapshotTable, ...]:
    """Return available manifest tables and reject missing required inputs."""

    available = tuple(
        table
        for table in DASHBOARD_TABLES
        if _table_exists(connection, "source_db", table)
    )
    available_names = {table.qualified_name for table in available}
    missing = [
        table.qualified_name
        for table in DASHBOARD_TABLES
        if table.required and table.qualified_name not in available_names
    ]
    if missing:
        raise RuntimeError(
            "Dashboard snapshot source is missing required tables: "
            + ", ".join(missing)
        )
    return available


def copy_dashboard_tables(
    connection: duckdb.DuckDBPyConnection,
    tables: tuple[SnapshotTable, ...],
) -> dict[str, int]:
    """Copy only the explicitly approved dashboard tables."""

    row_counts: dict[str, int] = {}
    for schema in sorted({table.schema for table in tables}):
        connection.execute(f"CREATE SCHEMA {_quote_identifier(schema)}")

    for table in tables:
        schema = _quote_identifier(table.schema)
        name = _quote_identifier(table.name)
        connection.execute(
            f"CREATE TABLE {schema}.{name} AS "
            f"SELECT * FROM source_db.{schema}.{name}"
        )
        source_count = connection.execute(
            f"SELECT COUNT(*) FROM source_db.{schema}.{name}"
        ).fetchone()[0]
        target_count = connection.execute(
            f"SELECT COUNT(*) FROM {schema}.{name}"
        ).fetchone()[0]
        if source_count != target_count:
            raise RuntimeError(
                f"Dashboard snapshot row count mismatch for "
                f"{table.qualified_name}: source={source_count}, "
                f"target={target_count}."
            )
        row_counts[table.qualified_name] = target_count
    return row_counts


def validate_snapshot_contents(
    connection: duckdb.DuckDBPyConnection,
    expected_tables: tuple[SnapshotTable, ...],
) -> None:
    """Reject missing or unexpected user tables in the deployment artifact."""

    actual = {
        f"{schema}.{name}"
        for schema, name in connection.execute(
            """
            SELECT table_schema, table_name
            FROM information_schema.tables
            WHERE table_catalog = current_database()
              AND table_schema NOT IN ('information_schema', 'pg_catalog')
            """
        ).fetchall()
    }
    expected = {table.qualified_name for table in expected_tables}
    if actual != expected:
        raise RuntimeError(
            "Dashboard snapshot table contract mismatch: "
            f"missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}."
        )


def build_dashboard_snapshot(
    source_file: Path = DATABASE_FILE,
    output_file: Path = DEFAULT_OUTPUT_FILE,
) -> dict[str, int]:
    """Build and atomically publish the compact dashboard database."""

    source = Path(source_file).resolve()
    output = Path(output_file).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Dashboard snapshot source not found: {source}")
    if source == output:
        raise ValueError("Dashboard snapshot output must differ from its source.")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.building")
    if temporary.exists():
        temporary.unlink()

    connection = duckdb.connect(str(temporary))
    try:
        connection.execute(
            f"ATTACH {_quote_string(str(source))} AS source_db (READ_ONLY)"
        )
        tables = validate_source_tables(connection)
        row_counts = copy_dashboard_tables(connection, tables)
        connection.execute("DETACH source_db")
        validate_snapshot_contents(connection, tables)
        connection.execute("CHECKPOINT")
    except Exception:
        connection.close()
        if temporary.exists():
            temporary.unlink()
        raise
    else:
        connection.close()

    os.replace(temporary, output)
    LOGGER.info(
        "Dashboard deployment snapshot built: %s tables, %.2f MB at %s",
        len(row_counts),
        output.stat().st_size / 1024 / 1024,
        output,
    )
    return row_counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the compact public dashboard DuckDB snapshot."
    )
    parser.add_argument("--source", type=Path, default=DATABASE_FILE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_FILE)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    args = parse_args()
    build_dashboard_snapshot(args.source, args.output)


if __name__ == "__main__":
    main()
