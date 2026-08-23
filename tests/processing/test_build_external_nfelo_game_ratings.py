"""
Tests for external nfelo game rating builder.
"""

from pathlib import Path

import duckdb
import pandas as pd
import pytest

from src.processing.build_external_nfelo_game_ratings import (
    TARGET_FULL_NAME,
    build_external_nfelo_game_ratings,
    create_external_nfelo_game_ratings_table,
    download_external_nfelo_games,
    validate_external_nfelo_game_ratings_table,
    validate_modeling_game_coverage,
)
from src.processing.normalize_nfelo_game_ratings import (
    normalize_nfelo_game_ratings,
)


def create_source_data() -> pd.DataFrame:
    """Create valid external nfelo source data."""

    return pd.DataFrame(
        [
            {
                "game_id": "2019_01_OAK_DEN",
                "starting_nfelo_home": 1520.0,
                "starting_nfelo_away": 1480.0,
                "nfelo_dif_base": 40.0,
                "nfelo_home_probability_open": 0.58,
                "nfelo_home_probability_close": 0.60,
            },
            {
                "game_id": "2026_01_DEN_KC",
                "starting_nfelo_home": 1575.0,
                "starting_nfelo_away": 1570.0,
                "nfelo_dif_base": 5.0,
                "nfelo_home_probability_open": 0.53,
                "nfelo_home_probability_close": 0.54,
            },
        ]
    )


def create_database(
    database_file: Path,
    modeling_game_ids: list[str] | None = None,
) -> None:
    """Create a test DuckDB and optional modeling data."""

    with duckdb.connect(
        str(database_file)
    ) as connection:
        if modeling_game_ids is None:
            return

        connection.execute(
            "CREATE SCHEMA analytics"
        )

        modeling_data = pd.DataFrame(
            {
                "game_id": modeling_game_ids,
            }
        )

        connection.register(
            "_modeling_data",
            modeling_data,
        )

        connection.execute(
            """
            CREATE TABLE
                analytics.game_modeling_dataset
            AS
            SELECT *
            FROM _modeling_data
            """
        )

        connection.unregister(
            "_modeling_data"
        )


def test_download_external_nfelo_games_from_local_csv(
    tmp_path: Path,
) -> None:
    """The downloader accepts a local test CSV."""

    source_file = tmp_path / "nfelo.csv"

    create_source_data().to_csv(
        source_file,
        index=False,
    )

    downloaded = download_external_nfelo_games(
        source_url=str(source_file)
    )

    assert len(downloaded) == 2

    assert set(
        downloaded["game_id"]
    ) == {
        "2019_01_OAK_DEN",
        "2026_01_DEN_KC",
    }


def test_create_and_validate_external_table(
) -> None:
    """Persist and validate normalized external data."""

    normalized = normalize_nfelo_game_ratings(
        create_source_data()
    )

    with duckdb.connect(":memory:") as connection:
        create_external_nfelo_game_ratings_table(
            connection=connection,
            normalized_data=normalized,
        )

        validate_external_nfelo_game_ratings_table(
            connection=connection,
            expected_row_count=2,
        )

        persisted = connection.execute(
            f"""
            SELECT
                source_game_id,
                normalized_game_id,
                source_name,
                source_url,
                source_fetched_at
            FROM {TARGET_FULL_NAME}
            ORDER BY normalized_game_id
            """
        ).fetchdf()

    assert len(persisted) == 2

    assert persisted[
        "source_name"
    ].eq("nfelo_games").all()

    assert persisted[
        "source_url"
    ].notna().all()

    assert persisted[
        "source_fetched_at"
    ].notna().all()


def test_modeling_coverage_validation() -> None:
    """Every modeling game must match external ratings."""

    normalized = normalize_nfelo_game_ratings(
        create_source_data()
    )

    with duckdb.connect(":memory:") as connection:
        connection.execute(
            "CREATE SCHEMA analytics"
        )

        modeling_data = pd.DataFrame(
            {
                "game_id": [
                    "2019_01_OAK_DEN",
                    "2026_01_DEN_KC",
                ],
            }
        )

        connection.register(
            "_modeling_data",
            modeling_data,
        )

        connection.execute(
            """
            CREATE TABLE
                analytics.game_modeling_dataset
            AS
            SELECT *
            FROM _modeling_data
            """
        )

        connection.unregister(
            "_modeling_data"
        )

        create_external_nfelo_game_ratings_table(
            connection=connection,
            normalized_data=normalized,
        )

        validate_modeling_game_coverage(
            connection
        )


def test_incomplete_modeling_coverage_is_rejected(
) -> None:
    """Reject a modeling game without external match."""

    normalized = normalize_nfelo_game_ratings(
        create_source_data()
    )

    with duckdb.connect(":memory:") as connection:
        connection.execute(
            "CREATE SCHEMA analytics"
        )

        modeling_data = pd.DataFrame(
            {
                "game_id": [
                    "2019_01_OAK_DEN",
                    "missing_game",
                ],
            }
        )

        connection.register(
            "_modeling_data",
            modeling_data,
        )

        connection.execute(
            """
            CREATE TABLE
                analytics.game_modeling_dataset
            AS
            SELECT *
            FROM _modeling_data
            """
        )

        connection.unregister(
            "_modeling_data"
        )

        create_external_nfelo_game_ratings_table(
            connection=connection,
            normalized_data=normalized,
        )

        with pytest.raises(
            RuntimeError,
            match="1 of 2 games matched",
        ):
            validate_modeling_game_coverage(
                connection
            )


def test_invalid_persisted_rating_is_rejected(
) -> None:
    """Reject corrupted external rating values."""

    normalized = normalize_nfelo_game_ratings(
        create_source_data()
    )

    with duckdb.connect(":memory:") as connection:
        create_external_nfelo_game_ratings_table(
            connection=connection,
            normalized_data=normalized,
        )

        connection.execute(
            f"""
            UPDATE {TARGET_FULL_NAME}
            SET starting_nfelo_home = NULL
            WHERE source_game_id = '2026_01_DEN_KC'
            """
        )

        with pytest.raises(
            RuntimeError,
            match="Invalid external nfelo ratings",
        ):
            validate_external_nfelo_game_ratings_table(
                connection=connection,
                expected_row_count=2,
            )


def test_build_external_nfelo_ratings_end_to_end(
    tmp_path: Path,
) -> None:
    """Build from a local CSV into temporary DuckDB."""

    source_file = tmp_path / "nfelo.csv"
    database_file = tmp_path / "test.duckdb"

    create_source_data().to_csv(
        source_file,
        index=False,
    )

    create_database(
        database_file=database_file,
        modeling_game_ids=[
            "2019_01_OAK_DEN",
            "2026_01_DEN_KC",
        ],
    )

    result = build_external_nfelo_game_ratings(
        database_file=database_file,
        source_url=str(source_file),
    )

    with duckdb.connect(
        str(database_file)
    ) as connection:
        persisted = connection.execute(
            f"""
            SELECT
                normalized_game_id,
                starting_nfelo_home,
                starting_nfelo_away
            FROM {TARGET_FULL_NAME}
            ORDER BY normalized_game_id
            """
        ).fetchdf()

    assert len(result) == 2
    assert len(persisted) == 2

    assert set(
        persisted["normalized_game_id"]
    ) == {
        "2019_01_OAK_DEN",
        "2026_01_DEN_KC",
    }