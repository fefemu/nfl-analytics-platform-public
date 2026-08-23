"""
NFL Analytics Platform
Game Weather Feature Builder

Purpose:
    Normalize schedule-level venue and game-time weather
    fields into stable model-ready features.

Historical temperature and wind values describe game-time
conditions. Future production predictions will require a
separately timestamped pregame forecast source.

Author:
    Ferenc Kaizer

Version:
    0.1.0
"""

import logging
from pathlib import Path

import duckdb


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATABASE_FILE = (
    PROJECT_ROOT
    / "data"
    / "nfl_analytics.duckdb"
)

SOURCE_SCHEMA = "processed"
SOURCE_TABLE = "schedule"
SOURCE_FULL_NAME = (
    f"{SOURCE_SCHEMA}.{SOURCE_TABLE}"
)

TARGET_SCHEMA = "analytics"
TARGET_TABLE = "game_weather_features"
TARGET_FULL_NAME = (
    f"{TARGET_SCHEMA}.{TARGET_TABLE}"
)

INDOOR_ROOF_VALUES = (
    "dome",
    "closed",
)

EXPOSED_ROOF_VALUES = (
    "outdoors",
    "open",
)

ALLOWED_ROOF_VALUES = (
    *INDOOR_ROOF_VALUES,
    *EXPOSED_ROOF_VALUES,
    "unknown",
)

NEUTRAL_TEMPERATURE_F = 65.0
NEUTRAL_WIND_MPH = 0.0

FREEZING_TEMPERATURE_F = 32.0
HIGH_WIND_MPH = 15.0
EXTREME_HEAT_TEMPERATURE_F = 85.0

COLD_REFERENCE_TEMPERATURE_F = 50.0
HEAT_REFERENCE_TEMPERATURE_F = 80.0
WIND_REFERENCE_MPH = 10.0

REQUIRED_SOURCE_COLUMNS = {
    "game_id",
    "season",
    "gameday",
    "home_team",
    "away_team",
    "roof",
    "surface",
    "stadium_id",
    "stadium",
    "temp",
    "wind",
}


def validate_database_file(
    database_file: Path = DATABASE_FILE,
) -> None:
    """Validate the DuckDB database path."""

    if not database_file.exists():
        raise FileNotFoundError(
            f"Database file does not exist: "
            f"{database_file}"
        )

    if not database_file.is_file():
        raise RuntimeError(
            f"Database path is not a file: "
            f"{database_file}"
        )


def get_table_columns(
    connection: duckdb.DuckDBPyConnection,
    schema_name: str,
    table_name: str,
) -> set[str]:
    """Return available table columns."""

    return {
        row[0]
        for row in connection.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = ?
              AND table_name = ?
            """,
            [
                schema_name,
                table_name,
            ],
        ).fetchall()
    }


def validate_weather_source(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Validate the processed schedule source."""

    available_columns = get_table_columns(
        connection=connection,
        schema_name=SOURCE_SCHEMA,
        table_name=SOURCE_TABLE,
    )

    if not available_columns:
        raise RuntimeError(
            "Weather source table does not exist: "
            f"{SOURCE_FULL_NAME}"
        )

    missing_columns = sorted(
        REQUIRED_SOURCE_COLUMNS
        - available_columns
    )

    if missing_columns:
        raise RuntimeError(
            "Weather source is missing columns: "
            + ", ".join(missing_columns)
        )

    source_row_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {SOURCE_FULL_NAME}
        """
    ).fetchone()[0]

    if source_row_count == 0:
        raise RuntimeError(
            "Weather source table is empty."
        )

    logger.info(
        "Game weather source validated: %s rows "
        "in %s.",
        source_row_count,
        SOURCE_FULL_NAME,
    )


def create_game_weather_features_table(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Create normalized model-ready weather features."""

    allowed_roof_sql = ", ".join(
        f"'{roof_value}'"
        for roof_value in ALLOWED_ROOF_VALUES
    )

    indoor_roof_sql = ", ".join(
        f"'{roof_value}'"
        for roof_value in INDOOR_ROOF_VALUES
    )

    exposed_roof_sql = ", ".join(
        f"'{roof_value}'"
        for roof_value in EXPOSED_ROOF_VALUES
    )

    connection.execute(
        f"""
        CREATE SCHEMA IF NOT EXISTS {TARGET_SCHEMA};

        CREATE OR REPLACE TABLE {TARGET_FULL_NAME} AS

        WITH normalized AS (

            SELECT
                game_id,
                season,
                CAST(gameday AS DATE) AS game_date,
                home_team,
                away_team,
                COALESCE(
                    NULLIF(
                        LOWER(TRIM(roof)),
                        ''
                    ),
                    'unknown'
                ) AS roof_type,
                LOWER(TRIM(surface)) AS surface_type,
                stadium_id,
                stadium,
                CAST(temp AS DOUBLE)
                    AS raw_temperature_f,
                CAST(wind AS DOUBLE)
                    AS raw_wind_mph

            FROM {SOURCE_FULL_NAME}
        ),

        classified AS (

            SELECT
                *,
                roof_type IN (
                    {indoor_roof_sql}
                ) AS is_indoor,

                roof_type IN (
                    {exposed_roof_sql}
                ) AS is_weather_exposed,

                (
                    roof_type IN (
                        {exposed_roof_sql}
                    )
                    AND raw_temperature_f
                        IS NOT NULL
                    AND raw_wind_mph
                        IS NOT NULL
                ) AS has_game_weather

            FROM normalized

        )

        SELECT
            game_id,
            season,
            game_date,
            home_team,
            away_team,
            roof_type,
            surface_type,
            stadium_id,
            stadium,
            is_indoor,
            is_weather_exposed,
            has_game_weather,
            raw_temperature_f,
            raw_wind_mph,

            CASE
                WHEN has_game_weather
                    THEN raw_temperature_f
                ELSE {NEUTRAL_TEMPERATURE_F}
            END AS modeled_temperature_f,

            CASE
                WHEN has_game_weather
                    THEN raw_wind_mph
                ELSE {NEUTRAL_WIND_MPH}
            END AS modeled_wind_mph,

            (
                has_game_weather
                AND raw_temperature_f
                    <= {FREEZING_TEMPERATURE_F}
            ) AS is_freezing,

            (
                has_game_weather
                AND raw_wind_mph
                    >= {HIGH_WIND_MPH}
            ) AS is_high_wind,

            (
                has_game_weather
                AND raw_temperature_f
                    >= {
                        EXTREME_HEAT_TEMPERATURE_F
                    }
            ) AS is_extreme_heat,

            CASE
                WHEN has_game_weather
                    THEN GREATEST(
                        {
                            COLD_REFERENCE_TEMPERATURE_F
                        }
                        - raw_temperature_f,
                        0.0
                    )
                ELSE 0.0
            END AS cold_degrees_below_50,

            CASE
                WHEN has_game_weather
                    THEN GREATEST(
                        raw_temperature_f
                        - {
                            HEAT_REFERENCE_TEMPERATURE_F
                        },
                        0.0
                    )
                ELSE 0.0
            END AS heat_degrees_above_80,

            CASE
                WHEN has_game_weather
                    THEN GREATEST(
                        raw_wind_mph
                        - {WIND_REFERENCE_MPH},
                        0.0
                    )
                ELSE 0.0
            END AS wind_mph_above_10

        FROM classified
        """
    )

    logger.info(
        "Game weather feature table created: %s.",
        TARGET_FULL_NAME,
    )


def validate_game_weather_features(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Validate normalized weather features."""

    source_row_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {SOURCE_FULL_NAME}
        """
    ).fetchone()[0]

    target_row_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        """
    ).fetchone()[0]

    if target_row_count != source_row_count:
        raise RuntimeError(
            "Game weather row count does not match "
            f"schedule: source={source_row_count}, "
            f"target={target_row_count}."
        )

    duplicate_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM (
            SELECT game_id
            FROM {TARGET_FULL_NAME}
            GROUP BY game_id
            HAVING COUNT(*) > 1
        )
        """
    ).fetchone()[0]

    if duplicate_count > 0:
        raise RuntimeError(
            "Duplicate game weather identifiers found."
        )

    invalid_roof_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE roof_type NOT IN (
            'dome',
            'closed',
            'outdoors',
            'open',
            'unknown'
        )
           OR (
                roof_type <> 'unknown'
                AND is_indoor
                    = is_weather_exposed
              )
           OR (
                roof_type = 'unknown'
                AND (
                    is_indoor
                    OR is_weather_exposed
                )
              )
        """
    ).fetchone()[0]

    if invalid_roof_count > 0:
        raise RuntimeError(
            "Invalid game weather roof "
            "classification found."
        )

    invalid_raw_weather_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE (
                raw_temperature_f IS NOT NULL
                AND raw_temperature_f
                    NOT BETWEEN -30.0 AND 130.0
              )
           OR (
                raw_wind_mph IS NOT NULL
                AND raw_wind_mph
                    NOT BETWEEN 0.0 AND 100.0
              )
        """
    ).fetchone()[0]

    if invalid_raw_weather_count > 0:
        raise RuntimeError(
            "Invalid raw game weather values found."
        )

    invalid_weather_flag_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE has_game_weather
            IS DISTINCT FROM (
                is_weather_exposed
                AND raw_temperature_f IS NOT NULL
                AND raw_wind_mph IS NOT NULL
            )
        """
    ).fetchone()[0]

    if invalid_weather_flag_count > 0:
        raise RuntimeError(
            "Invalid game weather coverage flags found."
        )

    invalid_modeled_weather_count = (
        connection.execute(
            f"""
            SELECT COUNT(*)
            FROM {TARGET_FULL_NAME}
            WHERE modeled_temperature_f
                    IS NULL
               OR modeled_wind_mph IS NULL
               OR (
                    has_game_weather
                    AND (
                        modeled_temperature_f
                            <> raw_temperature_f
                        OR modeled_wind_mph
                            <> raw_wind_mph
                    )
                  )
               OR (
                    NOT has_game_weather
                    AND (
                        modeled_temperature_f
                            <> {
                                NEUTRAL_TEMPERATURE_F
                            }
                        OR modeled_wind_mph
                            <> {NEUTRAL_WIND_MPH}
                    )
                  )
            """
        ).fetchone()[0]
    )

    if invalid_modeled_weather_count > 0:
        raise RuntimeError(
            "Invalid modeled game weather values found."
        )

    invalid_extreme_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE is_freezing
            IS DISTINCT FROM (
                has_game_weather
                AND raw_temperature_f
                    <= {FREEZING_TEMPERATURE_F}
            )
           OR is_high_wind
            IS DISTINCT FROM (
                has_game_weather
                AND raw_wind_mph
                    >= {HIGH_WIND_MPH}
            )
           OR is_extreme_heat
            IS DISTINCT FROM (
                has_game_weather
                AND raw_temperature_f
                    >= {
                        EXTREME_HEAT_TEMPERATURE_F
                    }
            )
           OR cold_degrees_below_50 < 0.0
           OR heat_degrees_above_80 < 0.0
           OR wind_mph_above_10 < 0.0
        """
    ).fetchone()[0]

    if invalid_extreme_count > 0:
        raise RuntimeError(
            "Invalid extreme weather features found."
        )

    logger.info(
        "Game weather features validated: %s rows.",
        target_row_count,
    )


def build_game_weather_features(
    database_file: Path = DATABASE_FILE,
) -> None:
    """Build the game weather feature table."""

    validate_database_file(database_file)

    logger.info(
        "Starting game weather feature build..."
    )

    with duckdb.connect(
        str(database_file)
    ) as connection:
        validate_weather_source(connection)

        connection.execute(
            "BEGIN TRANSACTION"
        )

        try:
            create_game_weather_features_table(
                connection
            )

            validate_game_weather_features(
                connection
            )

            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise

    logger.info(
        "Game weather feature build completed: %s.",
        TARGET_FULL_NAME,
    )


def main() -> None:
    """Run the game weather feature builder."""

    try:
        build_game_weather_features()
    except Exception:
        logger.exception(
            "Game weather feature build failed."
        )
        raise


if __name__ == "__main__":
    main()