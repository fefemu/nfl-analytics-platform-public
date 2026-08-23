"""
NFL Analytics Platform
Quarterback Ratings Builder

Purpose:
    Build leakage-safe historical and current quarterback ratings
    from quarterback game performance data.

Author:
    Ferenc Kaizer

Version:
    0.1.0
"""

import logging
import math
from dataclasses import astuple, dataclass
from datetime import date
from itertools import groupby
from pathlib import Path

import duckdb


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATABASE_FILE = PROJECT_ROOT / "data" / "nfl_analytics.duckdb"

QB_SOURCE_SCHEMA = "processed"
QB_SOURCE_TABLE = "qb_game_performance"
QB_SOURCE_FULL_NAME = (
    f"{QB_SOURCE_SCHEMA}.{QB_SOURCE_TABLE}"
)

ROLLING_SOURCE_SCHEMA = "analytics"
ROLLING_SOURCE_TABLE = "rolling_team_features"
ROLLING_SOURCE_FULL_NAME = (
    f"{ROLLING_SOURCE_SCHEMA}.{ROLLING_SOURCE_TABLE}"
)

TARGET_SCHEMA = "analytics"

HISTORY_TABLE = "qb_rating_history"
HISTORY_FULL_NAME = f"{TARGET_SCHEMA}.{HISTORY_TABLE}"

CURRENT_TABLE = "current_qb_ratings"
CURRENT_FULL_NAME = f"{TARGET_SCHEMA}.{CURRENT_TABLE}"

RECENCY_HALF_LIFE_DAYS = 365.0
PRIOR_DROPBACKS = 200.0

RATING_CENTER = 100.0
RATING_SCALE = 15.0
MINIMUM_STANDARD_DEVIATION = 0.05

REQUIRED_QB_COLUMNS = {
    "game_id",
    "season",
    "season_type",
    "week",
    "game_date",
    "team",
    "opponent",
    "qb_id",
    "qb_name",
    "is_primary_qb",
    "team_dropback_share",
    "dropbacks",
    "throw_attempts",
    "epa_per_dropback",
    "competitive_epa_per_dropback",
    "success_rate",
    "completion_rate",
    "cpoe",
    "sacks",
    "sack_rate",
    "interceptions",
    "interception_rate",
    "fumbles_lost",
    "turnovers",
    "turnover_rate",
}

REQUIRED_ROLLING_COLUMNS = {
    "game_id",
    "team",
    "pregame_defensive_epa_allowed_per_play_last_8",
}



@dataclass
class QuarterbackRatingState:
    """Store recency-weighted quarterback performance history."""

    weighted_dropbacks: float = 0.0
    weighted_epa_total: float = 0.0
    weighted_epa_squared_total: float = 0.0

    weighted_throw_attempts: float = 0.0
    weighted_cpoe_total: float = 0.0

    weighted_sacks: float = 0.0
    weighted_turnovers: float = 0.0

    last_updated: date | None = None

    def decay_to(
        self,
        as_of_date: date,
        half_life_days: float = RECENCY_HALF_LIFE_DAYS,
    ) -> None:
        """Decay historical information to a new date."""

        if half_life_days <= 0:
            raise ValueError(
                "Recency half-life must be positive."
            )

        if self.last_updated is None:
            self.last_updated = as_of_date
            return

        elapsed_days = (
            as_of_date - self.last_updated
        ).days

        if elapsed_days < 0:
            raise ValueError(
                "Rating state cannot move backward in time."
            )

        if elapsed_days == 0:
            return

        decay_factor = math.pow(
            0.5,
            elapsed_days / half_life_days,
        )

        self.weighted_dropbacks *= decay_factor
        self.weighted_epa_total *= decay_factor
        self.weighted_epa_squared_total *= decay_factor

        self.weighted_throw_attempts *= decay_factor
        self.weighted_cpoe_total *= decay_factor

        self.weighted_sacks *= decay_factor
        self.weighted_turnovers *= decay_factor

        self.last_updated = as_of_date

    def update(
        self,
        game_date: date,
        dropbacks: float,
        adjusted_epa_per_dropback: float,
        throw_attempts: float,
        cpoe: float | None,
        sacks: float,
        turnovers: float,
        half_life_days: float = RECENCY_HALF_LIFE_DAYS,
    ) -> None:
        """Add one completed QB-game to the rating state."""

        if dropbacks <= 0:
            raise ValueError(
                "QB game dropbacks must be positive."
            )

        if throw_attempts < 0:
            raise ValueError(
                "QB throw attempts cannot be negative."
            )

        self.decay_to(
            game_date,
            half_life_days,
        )

        self.weighted_dropbacks += dropbacks

        self.weighted_epa_total += (
            dropbacks * adjusted_epa_per_dropback
        )

        self.weighted_epa_squared_total += (
            dropbacks
            * adjusted_epa_per_dropback
            * adjusted_epa_per_dropback
        )

        if cpoe is not None and throw_attempts > 0:
            self.weighted_throw_attempts += throw_attempts
            self.weighted_cpoe_total += (
                throw_attempts * cpoe
            )

        self.weighted_sacks += sacks
        self.weighted_turnovers += turnovers

    @property
    def mean_epa_per_dropback(
        self,
    ) -> float | None:
        """Return recency-weighted EPA per dropback."""

        if self.weighted_dropbacks <= 0:
            return None

        return (
            self.weighted_epa_total
            / self.weighted_dropbacks
        )

    @property
    def epa_standard_deviation(
        self,
    ) -> float | None:
        """Return weighted between-game EPA standard deviation."""

        mean_epa = self.mean_epa_per_dropback

        if mean_epa is None:
            return None

        mean_squared = (
            self.weighted_epa_squared_total
            / self.weighted_dropbacks
        )

        variance = max(
            mean_squared - mean_epa * mean_epa,
            0.0,
        )

        return math.sqrt(variance)

    @property
    def mean_cpoe(
        self,
    ) -> float | None:
        """Return recency-weighted completion percentage over expected."""

        if self.weighted_throw_attempts <= 0:
            return None

        return (
            self.weighted_cpoe_total
            / self.weighted_throw_attempts
        )

    @property
    def sack_rate(
        self,
    ) -> float | None:
        """Return recency-weighted sack rate."""

        if self.weighted_dropbacks <= 0:
            return None

        return (
            self.weighted_sacks
            / self.weighted_dropbacks
        )

    @property
    def turnover_rate(
        self,
    ) -> float | None:
        """Return recency-weighted turnover rate."""

        if self.weighted_dropbacks <= 0:
            return None

        return (
            self.weighted_turnovers
            / self.weighted_dropbacks
        )



@dataclass(frozen=True)
class QuarterbackRatingSnapshot:
    """Represent one leakage-safe pregame QB rating snapshot."""

    effective_dropbacks: float
    raw_adjusted_epa_per_dropback: float
    shrunk_adjusted_epa_per_dropback: float

    league_mean_epa_per_dropback: float
    league_epa_standard_deviation: float

    rating_index: float
    prior_weight: float
    rating_standard_error: float

    cpoe: float | None
    sack_rate: float | None
    turnover_rate: float | None


def calculate_rating_snapshot(
    qb_state: QuarterbackRatingState,
    league_state: QuarterbackRatingState,
    prior_dropbacks: float = PRIOR_DROPBACKS,
) -> QuarterbackRatingSnapshot:
    """Calculate an empirical-Bayes-style QB rating snapshot."""

    if prior_dropbacks < 0:
        raise ValueError(
            "Prior dropbacks cannot be negative."
        )

    league_mean = (
        league_state.mean_epa_per_dropback
    )

    if league_mean is None:
        league_mean = 0.0

    league_standard_deviation = (
        league_state.epa_standard_deviation
    )

    if league_standard_deviation is None:
        league_standard_deviation = (
            MINIMUM_STANDARD_DEVIATION
        )

    league_standard_deviation = max(
        league_standard_deviation,
        MINIMUM_STANDARD_DEVIATION,
    )

    effective_dropbacks = (
        qb_state.weighted_dropbacks
    )

    raw_qb_epa = qb_state.mean_epa_per_dropback

    if raw_qb_epa is None:
        raw_qb_epa = league_mean

    total_information = (
        effective_dropbacks + prior_dropbacks
    )

    if total_information <= 0:
        shrunk_qb_epa = league_mean
        prior_weight = 1.0
    else:
        shrunk_qb_epa = (
            effective_dropbacks * raw_qb_epa
            + prior_dropbacks * league_mean
        ) / total_information

        prior_weight = (
            prior_dropbacks / total_information
        )

    standardized_rating = (
        shrunk_qb_epa - league_mean
    ) / league_standard_deviation

    rating_index = (
        RATING_CENTER
        + RATING_SCALE * standardized_rating
    )

    rating_standard_error = (
        RATING_SCALE
        / math.sqrt(max(total_information, 1.0))
    )

    return QuarterbackRatingSnapshot(
        effective_dropbacks=effective_dropbacks,
        raw_adjusted_epa_per_dropback=raw_qb_epa,
        shrunk_adjusted_epa_per_dropback=shrunk_qb_epa,
        league_mean_epa_per_dropback=league_mean,
        league_epa_standard_deviation=(
            league_standard_deviation
        ),
        rating_index=rating_index,
        prior_weight=prior_weight,
        rating_standard_error=rating_standard_error,
        cpoe=qb_state.mean_cpoe,
        sack_rate=qb_state.sack_rate,
        turnover_rate=qb_state.turnover_rate,
    )


@dataclass(frozen=True)
class QuarterbackGameRecord:
    """Represent one completed QB-game used for rating updates."""

    game_id: str
    season: int
    season_type: str
    week: int
    game_date: date

    team: str
    opponent: str

    qb_id: str
    qb_name: str | None
    is_primary_qb: bool
    team_dropback_share: float

    dropbacks: int
    throw_attempts: int

    epa_per_dropback: float
    opponent_defensive_epa: float
    adjusted_epa_per_dropback: float

    cpoe: float | None
    sacks: int
    turnovers: int


def validate_database_file(
    database_file: Path = DATABASE_FILE,
) -> None:
    """Validate that the DuckDB database file exists."""

    if not database_file.exists():
        raise FileNotFoundError(
            f"Database file does not exist: {database_file}"
        )

    if not database_file.is_file():
        raise RuntimeError(
            f"Database path is not a file: {database_file}"
        )

    logger.info(
        "Database file validated: %s",
        database_file,
    )


def validate_source_tables(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Validate QB performance and rolling feature sources."""

    required_tables = {
        (
            QB_SOURCE_SCHEMA,
            QB_SOURCE_TABLE,
        ): REQUIRED_QB_COLUMNS,
        (
            ROLLING_SOURCE_SCHEMA,
            ROLLING_SOURCE_TABLE,
        ): REQUIRED_ROLLING_COLUMNS,
    }

    for (
        schema_name,
        table_name,
    ), required_columns in required_tables.items():
        table_exists = connection.execute(
            """
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_schema = ?
              AND table_name = ?
            """,
            [schema_name, table_name],
        ).fetchone()[0]

        full_name = f"{schema_name}.{table_name}"

        if table_exists == 0:
            raise RuntimeError(
                f"Source table does not exist: {full_name}"
            )

        available_columns = {
            row[0]
            for row in connection.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = ?
                  AND table_name = ?
                """,
                [schema_name, table_name],
            ).fetchall()
        }

        missing_columns = sorted(
            required_columns - available_columns
        )

        if missing_columns:
            missing_names = ", ".join(missing_columns)
            raise RuntimeError(
                f"Missing columns in {full_name}: "
                f"{missing_names}"
            )

    logger.info(
        "QB rating source tables validated: %s and %s.",
        QB_SOURCE_FULL_NAME,
        ROLLING_SOURCE_FULL_NAME,
    )


def load_qb_games(
    connection: duckdb.DuckDBPyConnection,
) -> list[QuarterbackGameRecord]:
    """Load QB games with pregame opponent-defense adjustment."""

    rows = connection.execute(
        f"""
        SELECT
            qb.game_id,
            qb.season,
            qb.season_type,
            qb.week,
            qb.game_date,
            qb.team,
            qb.opponent,
            qb.qb_id,
            qb.qb_name,
            qb.is_primary_qb,
            qb.team_dropback_share,
            qb.dropbacks,
            qb.throw_attempts,
            qb.epa_per_dropback,

            COALESCE(
                opponent_features
                    .pregame_defensive_epa_allowed_per_play_last_8,
                0.0
            ) AS opponent_defensive_epa,

            qb.epa_per_dropback
            - COALESCE(
                opponent_features
                    .pregame_defensive_epa_allowed_per_play_last_8,
                0.0
              ) AS adjusted_epa_per_dropback,

            qb.cpoe,
            qb.sacks,
            qb.turnovers

        FROM {QB_SOURCE_FULL_NAME} AS qb

        LEFT JOIN {ROLLING_SOURCE_FULL_NAME}
            AS opponent_features
            ON qb.game_id = opponent_features.game_id
           AND qb.opponent = opponent_features.team

        ORDER BY
            qb.game_date,
            qb.game_id,
            qb.team,
            qb.qb_id
        """
    ).fetchall()

    games = [
        QuarterbackGameRecord(
            game_id=row[0],
            season=row[1],
            season_type=row[2],
            week=row[3],
            game_date=row[4],
            team=row[5],
            opponent=row[6],
            qb_id=row[7],
            qb_name=row[8],
            is_primary_qb=row[9],
            team_dropback_share=row[10],
            dropbacks=row[11],
            throw_attempts=row[12],
            epa_per_dropback=row[13],
            opponent_defensive_epa=row[14],
            adjusted_epa_per_dropback=row[15],
            cpoe=row[16],
            sacks=row[17],
            turnovers=row[18],
        )
        for row in rows
    ]

    if not games:
        raise RuntimeError(
            "No QB games were loaded for rating calculation."
        )

    logger.info(
        "QB rating games loaded: %s rows from seasons %s-%s.",
        len(games),
        min(game.season for game in games),
        max(game.season for game in games),
    )

    return games


@dataclass(frozen=True)
class QuarterbackRatingHistoryRow:
    """Store a QB rating as it existed before a game."""

    game_id: str
    season: int
    season_type: str
    week: int
    game_date: date

    team: str
    opponent: str

    qb_id: str
    qb_name: str | None
    is_primary_qb: bool
    team_dropback_share: float

    pregame_effective_dropbacks: float
    pregame_raw_adjusted_epa_per_dropback: float
    pregame_shrunk_adjusted_epa_per_dropback: float

    pregame_league_mean_epa_per_dropback: float
    pregame_league_epa_standard_deviation: float

    pregame_qb_rating: float
    pregame_prior_weight: float
    pregame_rating_standard_error: float

    pregame_cpoe: float | None
    pregame_sack_rate: float | None
    pregame_turnover_rate: float | None


@dataclass(frozen=True)
class CurrentQuarterbackRatingRow:
    """Store the latest available rating for one quarterback."""

    qb_rank: int
    qb_id: str
    qb_name: str | None
    current_team: str
    as_of_date: date
    last_game_date: date
    days_since_last_game: int

    games_played: int
    career_dropbacks: int
    effective_dropbacks: float

    raw_adjusted_epa_per_dropback: float
    shrunk_adjusted_epa_per_dropback: float

    league_mean_epa_per_dropback: float
    league_epa_standard_deviation: float

    qb_rating: float
    prior_weight: float
    rating_standard_error: float

    cpoe: float | None
    sack_rate: float | None
    turnover_rate: float | None



def calculate_qb_ratings(
    games: list[QuarterbackGameRecord],
    half_life_days: float = RECENCY_HALF_LIFE_DAYS,
    prior_dropbacks: float = PRIOR_DROPBACKS,
) -> tuple[
    list[QuarterbackRatingHistoryRow],
    list[CurrentQuarterbackRatingRow],
]:
    """Calculate historical pregame and current QB ratings."""

    if not games:
        raise ValueError(
            "QB rating calculation requires at least one game."
        )

    if half_life_days <= 0:
        raise ValueError(
            "Recency half-life must be positive."
        )

    if prior_dropbacks < 0:
        raise ValueError(
            "Prior dropbacks cannot be negative."
        )

    ordered_games = sorted(
        games,
        key=lambda game: (
            game.game_date,
            game.game_id,
            game.team,
            game.qb_id,
        ),
    )

    qb_states: dict[
        str,
        QuarterbackRatingState,
    ] = {}

    league_state = QuarterbackRatingState()

    games_played: dict[str, int] = {}
    career_dropbacks: dict[str, int] = {}

    latest_name: dict[str, str | None] = {}
    latest_team: dict[str, str] = {}
    latest_game_date: dict[str, date] = {}

    history_rows: list[
        QuarterbackRatingHistoryRow
    ] = []

    for game_date, date_group_iterator in groupby(
        ordered_games,
        key=lambda game: game.game_date,
    ):
        date_games = list(date_group_iterator)

        league_state.decay_to(
            game_date,
            half_life_days,
        )

        participating_qb_ids = {
            game.qb_id
            for game in date_games
        }

        for qb_id in participating_qb_ids:
            qb_state = qb_states.setdefault(
                qb_id,
                QuarterbackRatingState(),
            )

            qb_state.decay_to(
                game_date,
                half_life_days,
            )

        # ---------------------------------------------
        # Snapshot every rating before updating the day
        # ---------------------------------------------

        for game in date_games:
            qb_state = qb_states[game.qb_id]

            snapshot = calculate_rating_snapshot(
                qb_state,
                league_state,
                prior_dropbacks,
            )

            history_rows.append(
                QuarterbackRatingHistoryRow(
                    game_id=game.game_id,
                    season=game.season,
                    season_type=game.season_type,
                    week=game.week,
                    game_date=game.game_date,
                    team=game.team,
                    opponent=game.opponent,
                    qb_id=game.qb_id,
                    qb_name=game.qb_name,
                    is_primary_qb=game.is_primary_qb,
                    team_dropback_share=(
                        game.team_dropback_share
                    ),
                    pregame_effective_dropbacks=(
                        snapshot.effective_dropbacks
                    ),
                    pregame_raw_adjusted_epa_per_dropback=(
                        snapshot
                        .raw_adjusted_epa_per_dropback
                    ),
                    pregame_shrunk_adjusted_epa_per_dropback=(
                        snapshot
                        .shrunk_adjusted_epa_per_dropback
                    ),
                    pregame_league_mean_epa_per_dropback=(
                        snapshot
                        .league_mean_epa_per_dropback
                    ),
                    pregame_league_epa_standard_deviation=(
                        snapshot
                        .league_epa_standard_deviation
                    ),
                    pregame_qb_rating=(
                        snapshot.rating_index
                    ),
                    pregame_prior_weight=(
                        snapshot.prior_weight
                    ),
                    pregame_rating_standard_error=(
                        snapshot.rating_standard_error
                    ),
                    pregame_cpoe=snapshot.cpoe,
                    pregame_sack_rate=(
                        snapshot.sack_rate
                    ),
                    pregame_turnover_rate=(
                        snapshot.turnover_rate
                    ),
                )
            )

        # ---------------------------------------------
        # Only completed games may update the states
        # ---------------------------------------------

        for game in date_games:
            qb_state = qb_states[game.qb_id]

            qb_state.update(
                game_date=game.game_date,
                dropbacks=game.dropbacks,
                adjusted_epa_per_dropback=(
                    game.adjusted_epa_per_dropback
                ),
                throw_attempts=game.throw_attempts,
                cpoe=game.cpoe,
                sacks=game.sacks,
                turnovers=game.turnovers,
                half_life_days=half_life_days,
            )

            league_state.update(
                game_date=game.game_date,
                dropbacks=game.dropbacks,
                adjusted_epa_per_dropback=(
                    game.adjusted_epa_per_dropback
                ),
                throw_attempts=game.throw_attempts,
                cpoe=game.cpoe,
                sacks=game.sacks,
                turnovers=game.turnovers,
                half_life_days=half_life_days,
            )

            games_played[game.qb_id] = (
                games_played.get(game.qb_id, 0) + 1
            )

            career_dropbacks[game.qb_id] = (
                career_dropbacks.get(game.qb_id, 0)
                + game.dropbacks
            )

            latest_name[game.qb_id] = game.qb_name
            latest_team[game.qb_id] = game.team
            latest_game_date[game.qb_id] = (
                game.game_date
            )

    as_of_date = ordered_games[-1].game_date

    current_without_rank = []

    for qb_id, qb_state in qb_states.items():
        qb_state.decay_to(
            as_of_date,
            half_life_days,
        )

        snapshot = calculate_rating_snapshot(
            qb_state,
            league_state,
            prior_dropbacks,
        )

        current_without_rank.append(
            (
                qb_id,
                snapshot,
            )
        )

    current_without_rank.sort(
        key=lambda item: (
            -item[1].rating_index,
            item[0],
        )
    )

    current_rows = [
        CurrentQuarterbackRatingRow(
            qb_rank=rank,
            qb_id=qb_id,
            qb_name=latest_name[qb_id],
            current_team=latest_team[qb_id],
            as_of_date=as_of_date,
            last_game_date=latest_game_date[qb_id],
            days_since_last_game=(
                as_of_date
                - latest_game_date[qb_id]
            ).days,
            games_played=games_played[qb_id],
            career_dropbacks=career_dropbacks[qb_id],
            effective_dropbacks=(
                snapshot.effective_dropbacks
            ),
            raw_adjusted_epa_per_dropback=(
                snapshot.raw_adjusted_epa_per_dropback
            ),
            shrunk_adjusted_epa_per_dropback=(
                snapshot
                .shrunk_adjusted_epa_per_dropback
            ),
            league_mean_epa_per_dropback=(
                snapshot.league_mean_epa_per_dropback
            ),
            league_epa_standard_deviation=(
                snapshot.league_epa_standard_deviation
            ),
            qb_rating=snapshot.rating_index,
            prior_weight=snapshot.prior_weight,
            rating_standard_error=(
                snapshot.rating_standard_error
            ),
            cpoe=snapshot.cpoe,
            sack_rate=snapshot.sack_rate,
            turnover_rate=snapshot.turnover_rate,
        )
        for rank, (
            qb_id,
            snapshot,
        ) in enumerate(
            current_without_rank,
            start=1,
        )
    ]

    logger.info(
        "QB ratings calculated: %s history rows and "
        "%s current QB ratings.",
        len(history_rows),
        len(current_rows),
    )

    return history_rows, current_rows


def create_qb_rating_tables(
    connection: duckdb.DuckDBPyConnection,
    history_rows: list[QuarterbackRatingHistoryRow],
    current_rows: list[CurrentQuarterbackRatingRow],
) -> None:
    """Create historical and current QB rating tables."""

    if not history_rows:
        raise RuntimeError(
            "QB rating history rows cannot be empty."
        )

    if not current_rows:
        raise RuntimeError(
            "Current QB rating rows cannot be empty."
        )

    connection.execute(
        f"""
        CREATE SCHEMA IF NOT EXISTS {TARGET_SCHEMA};

        CREATE OR REPLACE TABLE {HISTORY_FULL_NAME} (
            game_id VARCHAR,
            season INTEGER,
            season_type VARCHAR,
            week INTEGER,
            game_date DATE,
            team VARCHAR,
            opponent VARCHAR,
            qb_id VARCHAR,
            qb_name VARCHAR,
            is_primary_qb BOOLEAN,
            team_dropback_share DOUBLE,

            pregame_effective_dropbacks DOUBLE,
            pregame_raw_adjusted_epa_per_dropback DOUBLE,
            pregame_shrunk_adjusted_epa_per_dropback DOUBLE,

            pregame_league_mean_epa_per_dropback DOUBLE,
            pregame_league_epa_standard_deviation DOUBLE,

            pregame_qb_rating DOUBLE,
            pregame_prior_weight DOUBLE,
            pregame_rating_standard_error DOUBLE,

            pregame_cpoe DOUBLE,
            pregame_sack_rate DOUBLE,
            pregame_turnover_rate DOUBLE
        );

        CREATE OR REPLACE TABLE {CURRENT_FULL_NAME} (
            qb_rank INTEGER,
            qb_id VARCHAR,
            qb_name VARCHAR,
            current_team VARCHAR,
            as_of_date DATE,
            last_game_date DATE,
            days_since_last_game INTEGER,

            games_played INTEGER,
            career_dropbacks INTEGER,
            effective_dropbacks DOUBLE,

            raw_adjusted_epa_per_dropback DOUBLE,
            shrunk_adjusted_epa_per_dropback DOUBLE,

            league_mean_epa_per_dropback DOUBLE,
            league_epa_standard_deviation DOUBLE,

            qb_rating DOUBLE,
            prior_weight DOUBLE,
            rating_standard_error DOUBLE,

            cpoe DOUBLE,
            sack_rate DOUBLE,
            turnover_rate DOUBLE
        )
        """
    )

    history_placeholders = ", ".join(
        ["?"] * 22
    )

    connection.executemany(
        f"""
        INSERT INTO {HISTORY_FULL_NAME}
        VALUES ({history_placeholders})
        """,
        [
            astuple(row)
            for row in history_rows
        ],
    )

    current_placeholders = ", ".join(
        ["?"] * 20
    )

    connection.executemany(
        f"""
        INSERT INTO {CURRENT_FULL_NAME}
        VALUES ({current_placeholders})
        """,
        [
            astuple(row)
            for row in current_rows
        ],
    )

    logger.info(
        "QB rating tables created: %s history rows and "
        "%s current rows.",
        len(history_rows),
        len(current_rows),
    )


def validate_qb_rating_tables(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Validate historical and current QB rating tables."""

    history_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {HISTORY_FULL_NAME}
        """
    ).fetchone()[0]

    source_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {QB_SOURCE_FULL_NAME}
        """
    ).fetchone()[0]

    current_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {CURRENT_FULL_NAME}
        """
    ).fetchone()[0]

    if history_count == 0:
        raise RuntimeError(
            f"QB rating history is empty: {HISTORY_FULL_NAME}"
        )

    if current_count == 0:
        raise RuntimeError(
            f"Current QB ratings are empty: {CURRENT_FULL_NAME}"
        )

    if history_count != source_count:
        raise RuntimeError(
            "QB rating history row count does not match source: "
            f"history={history_count}, source={source_count}"
        )

    duplicate_history_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM (
            SELECT
                game_id,
                team,
                qb_id
            FROM {HISTORY_FULL_NAME}
            GROUP BY
                game_id,
                team,
                qb_id
            HAVING COUNT(*) > 1
        )
        """
    ).fetchone()[0]

    if duplicate_history_count > 0:
        raise RuntimeError(
            "Duplicate QB rating history keys found: "
            f"{duplicate_history_count}"
        )

    duplicate_current_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM (
            SELECT qb_id
            FROM {CURRENT_FULL_NAME}
            GROUP BY qb_id
            HAVING COUNT(*) > 1
        )
        """
    ).fetchone()[0]

    if duplicate_current_count > 0:
        raise RuntimeError(
            "Duplicate current QB ratings found: "
            f"{duplicate_current_count}"
        )

    invalid_history_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {HISTORY_FULL_NAME}
        WHERE pregame_effective_dropbacks < 0
           OR pregame_prior_weight NOT BETWEEN 0 AND 1
           OR pregame_rating_standard_error < 0
           OR NOT isfinite(pregame_qb_rating)
           OR NOT isfinite(
                pregame_shrunk_adjusted_epa_per_dropback
              )
        """
    ).fetchone()[0]

    if invalid_history_count > 0:
        raise RuntimeError(
            "Invalid historical QB ratings found: "
            f"{invalid_history_count}"
        )

    invalid_first_rating_count = connection.execute(
        f"""
        WITH ordered_ratings AS (
            SELECT
                qb_id,
                game_date,
                game_id,
                pregame_effective_dropbacks,
                pregame_qb_rating,
                ROW_NUMBER() OVER (
                    PARTITION BY qb_id
                    ORDER BY
                        game_date,
                        game_id
                ) AS qb_game_number
            FROM {HISTORY_FULL_NAME}
        )
        SELECT COUNT(*)
        FROM ordered_ratings
        WHERE qb_game_number = 1
          AND (
                ABS(pregame_effective_dropbacks)
                    > 0.000000001
                OR ABS(
                    pregame_qb_rating
                    - {RATING_CENTER}
                ) > 0.000000001
          )
        """
    ).fetchone()[0]

    if invalid_first_rating_count > 0:
        raise RuntimeError(
            "First QB appearances contain historical leakage: "
            f"{invalid_first_rating_count}"
        )

    invalid_current_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {CURRENT_FULL_NAME}
        WHERE qb_rank <= 0
           OR games_played <= 0
           OR career_dropbacks <= 0
           OR effective_dropbacks <= 0
           OR days_since_last_game < 0
           OR prior_weight NOT BETWEEN 0 AND 1
           OR rating_standard_error < 0
           OR NOT isfinite(qb_rating)
           OR NOT isfinite(
                shrunk_adjusted_epa_per_dropback
              )
        """
    ).fetchone()[0]

    if invalid_current_count > 0:
        raise RuntimeError(
            "Invalid current QB ratings found: "
            f"{invalid_current_count}"
        )

    invalid_rank_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM (
            SELECT
                qb_rank,
                ROW_NUMBER() OVER (
                    ORDER BY
                        qb_rating DESC,
                        qb_id
                ) AS expected_rank
            FROM {CURRENT_FULL_NAME}
        )
        WHERE qb_rank <> expected_rank
        """
    ).fetchone()[0]

    if invalid_rank_count > 0:
        raise RuntimeError(
            "Current QB ranking order is invalid: "
            f"{invalid_rank_count}"
        )

    logger.info(
        "QB rating tables validated successfully: "
        "%s history rows and %s current ratings.",
        history_count,
        current_count,
    )


def build_qb_ratings(
    database_file: Path = DATABASE_FILE,
    half_life_days: float = RECENCY_HALF_LIFE_DAYS,
    prior_dropbacks: float = PRIOR_DROPBACKS,
) -> None:
    """Build historical and current quarterback ratings."""

    validate_database_file(database_file)

    logger.info(
        "Starting QB ratings build with half-life %.1f days "
        "and %.1f prior dropbacks.",
        half_life_days,
        prior_dropbacks,
    )

    with duckdb.connect(str(database_file)) as connection:
        validate_source_tables(connection)

        games = load_qb_games(connection)

        history_rows, current_rows = (
            calculate_qb_ratings(
                games=games,
                half_life_days=half_life_days,
                prior_dropbacks=prior_dropbacks,
            )
        )

        connection.execute("BEGIN TRANSACTION")

        try:
            create_qb_rating_tables(
                connection,
                history_rows,
                current_rows,
            )

            validate_qb_rating_tables(connection)

            connection.execute("COMMIT")

            logger.info(
                "QB ratings transaction committed."
            )

        except Exception:
            connection.execute("ROLLBACK")

            logger.exception(
                "QB ratings build failed; "
                "transaction rolled back."
            )
            raise

    logger.info(
        "QB ratings build completed: %s and %s.",
        HISTORY_FULL_NAME,
        CURRENT_FULL_NAME,
    )


def main() -> None:
    """Run the quarterback ratings builder."""

    try:
        build_qb_ratings()
    except Exception:
        logger.exception(
            "QB ratings builder failed."
        )
        raise


if __name__ == "__main__":
    main()