"""Integration tests for the combined betting board."""

from pathlib import Path

import duckdb

from src.betting.build_current_betting_board import BOARD_COLUMNS, build_current_betting_board


def create_database(path: Path) -> None:
    with duckdb.connect(str(path)) as connection:
        connection.execute("CREATE SCHEMA analytics")
        common = """
            snapshot_id VARCHAR, fetched_at TIMESTAMP, game_id VARCHAR, season INTEGER,
            game_type VARCHAR, week INTEGER, gameday DATE, commence_time TIMESTAMP,
            home_team VARCHAR, away_team VARCHAR, market_key VARCHAR, market_name VARCHAR,
            outcome_name VARCHAR, outcome_type VARCHAR, best_bookmaker_key VARCHAR,
            best_bookmaker_title VARCHAR, best_american_price INTEGER, best_decimal_odds DOUBLE,
            bookmaker_count INTEGER, model_name VARCHAR, model_version VARCHAR,
            prediction_mode VARCHAR, probability_edge DOUBLE,
            probability_edge_percentage_points DOUBLE, fair_decimal_odds DOUBLE,
            expected_value_per_unit DOUBLE, expected_value_percent DOUBLE,
            full_kelly_fraction DOUBLE, positive_expected_value BOOLEAN,
            prediction_generated_at TIMESTAMP
        """
        connection.execute(f"""CREATE TABLE analytics.current_moneyline_value (
            {common}, model_probability DOUBLE
        )""")
        connection.execute(f"""CREATE TABLE analytics.current_spread_value (
            {common}, market_line DOUBLE, point DOUBLE, no_push_cover_probability DOUBLE,
            cover_probability DOUBLE,
            push_probability DOUBLE, loss_probability DOUBLE
        )""")
        connection.execute(f"""CREATE TABLE analytics.current_totals_value (
            {common}, market_line DOUBLE, point DOUBLE, no_push_win_probability DOUBLE,
            win_probability DOUBLE,
            push_probability DOUBLE, loss_probability DOUBLE
        )""")
        base = """'s', TIMESTAMP '2026-08-09', 'g', 2026, 'REG', 1,
            DATE '2026-09-10', TIMESTAMP '2026-09-10', 'BUF', 'NYJ', {}, {}, {}, {},
            'book', 'Book', -110, 2.0, 5, 'model', '1', 'mode', 0.05, 5.0,
            1.8, 0.1, 10.0, 0.1, TRUE, TIMESTAMP '2026-08-09'"""
        connection.execute("INSERT INTO analytics.current_moneyline_value VALUES (" +
                           base.format("'h2h'", "'Moneyline'", "'BUF'", "'home'") + ", 0.6)")
        connection.execute("INSERT INTO analytics.current_spread_value VALUES (" +
                           base.format("'spreads'", "'Spread'", "'BUF'", "'home'") + ", -3.0, -3.0, 0.6, 0.6, 0.0, 0.4)")
        connection.execute("INSERT INTO analytics.current_totals_value VALUES (" +
                           base.format("'totals'", "'Totals'", "'Over'", "'over'") + ", 45.5, 45.5, 0.6, 0.6, 0.0, 0.4)")


def test_build_combined_board(tmp_path: Path) -> None:
    database = tmp_path / "board.duckdb"
    create_database(database)
    board = build_current_betting_board(database)
    assert tuple(board.columns) == BOARD_COLUMNS
    assert len(board) == 3
    assert set(board["market_key"]) == {"h2h", "spreads", "totals"}
    with duckdb.connect(str(database), read_only=True) as connection:
        assert connection.execute("SELECT COUNT(*) FROM analytics.current_betting_board").fetchone()[0] == 3
