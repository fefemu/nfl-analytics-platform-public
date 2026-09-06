"""Build the derived live-forward settlement and performance layer."""

from pathlib import Path

import duckdb

from src.betting.forward_performance import (
    create_forward_performance_summary,
    create_forward_settlement,
    persist_forward_performance,
    validate_forward_performance,
)
from src.modeling.train_logistic_baseline import DATABASE_FILE, validate_database_file


def build_forward_performance(database_file: Path = DATABASE_FILE) -> tuple[int, int]:
    """Settle eligible archived entries against finalized schedule results."""
    validate_database_file(database_file)
    with duckdb.connect(str(database_file)) as connection:
        entries = connection.execute("SELECT * FROM analytics.forward_tip_market_movement").fetchdf()
        schedule = connection.execute("SELECT game_id, home_score, away_score, is_completed FROM processed.schedule").fetchdf()
        ledger = create_forward_settlement(entries, schedule)
        summary = create_forward_performance_summary(ledger)
        connection.execute("BEGIN TRANSACTION")
        try:
            persist_forward_performance(connection, ledger, summary)
            validate_forward_performance(connection)
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise
    return len(ledger), len(summary)


if __name__ == "__main__":
    build_forward_performance()
