"""Tests for the compact public dashboard DuckDB builder."""

from pathlib import Path

import duckdb
import pytest

from src.deployment.build_dashboard_snapshot import (
    DASHBOARD_TABLES,
    build_dashboard_snapshot,
)


def create_source_database(
    database_file: Path,
    *,
    include_optional: bool = True,
) -> None:
    with duckdb.connect(str(database_file)) as connection:
        for schema in {table.schema for table in DASHBOARD_TABLES}:
            connection.execute(f"CREATE SCHEMA {schema}")
        for index, table in enumerate(DASHBOARD_TABLES):
            if not table.required and not include_optional:
                continue
            connection.execute(
                f"CREATE TABLE {table.qualified_name} "
                "(row_id INTEGER, value VARCHAR)"
            )
            connection.execute(
                f"INSERT INTO {table.qualified_name} VALUES (?, ?)",
                [index, table.qualified_name],
            )


def user_tables(database_file: Path) -> set[str]:
    with duckdb.connect(str(database_file), read_only=True) as connection:
        return {
            f"{schema}.{table}"
            for schema, table in connection.execute(
                """
                SELECT table_schema, table_name
                FROM information_schema.tables
                WHERE table_schema NOT IN ('information_schema', 'pg_catalog')
                """
            ).fetchall()
        }


def test_build_snapshot_copies_only_manifest_tables(tmp_path: Path) -> None:
    source = tmp_path / "source.duckdb"
    output = tmp_path / "dashboard.duckdb"
    create_source_database(source)
    with duckdb.connect(str(source)) as connection:
        connection.execute("CREATE TABLE analytics.private_debug (id INTEGER)")

    counts = build_dashboard_snapshot(source, output)

    expected = {table.qualified_name for table in DASHBOARD_TABLES}
    assert set(counts) == expected
    assert set(counts.values()) == {1}
    assert user_tables(output) == expected
    assert "analytics.private_debug" not in user_tables(output)


def test_optional_table_may_be_absent(tmp_path: Path) -> None:
    source = tmp_path / "source.duckdb"
    output = tmp_path / "dashboard.duckdb"
    create_source_database(source, include_optional=False)

    counts = build_dashboard_snapshot(source, output)

    assert "analytics.refresh_run_history" not in counts
    assert "processed.external_nfelounits_units" not in counts
    assert output.is_file()


def test_missing_required_table_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "source.duckdb"
    output = tmp_path / "dashboard.duckdb"
    create_source_database(source)
    with duckdb.connect(str(source)) as connection:
        connection.execute("DROP TABLE analytics.current_betting_board")

    with pytest.raises(
        RuntimeError,
        match="analytics.current_betting_board",
    ):
        build_dashboard_snapshot(source, output)

    assert not output.exists()


def test_failed_build_preserves_existing_output(tmp_path: Path) -> None:
    source = tmp_path / "source.duckdb"
    output = tmp_path / "dashboard.duckdb"
    create_source_database(source)
    output.write_bytes(b"previous-good-artifact")
    with duckdb.connect(str(source)) as connection:
        connection.execute("DROP TABLE raw.depth_charts_espn")

    with pytest.raises(RuntimeError, match="raw.depth_charts_espn"):
        build_dashboard_snapshot(source, output)

    assert output.read_bytes() == b"previous-good-artifact"


def test_source_and_output_must_differ(tmp_path: Path) -> None:
    source = tmp_path / "source.duckdb"
    create_source_database(source)

    with pytest.raises(ValueError, match="must differ"):
        build_dashboard_snapshot(source, source)
