"""Bilingual live-forward performance page backed by prepared analytics tables."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.dashboard.components import empty_state, metric_tile
from src.dashboard.i18n import DEFAULT_LANGUAGE, Language
from src.dashboard.view_models import format_decimal_odds


MARKET_LABELS = {"ALL": "All", "h2h": "Moneyline", "spreads": "Spread", "totals": "Total"}


def has_settled_results(settlement: pd.DataFrame) -> bool:
    """Keep the public page hidden until a genuine result exists."""
    return bool(
        not settlement.empty
        and "settlement_status" in settlement.columns
        and settlement["settlement_status"].eq("SETTLED").any()
    )


def _number(value: object, language: Language, digits: int = 1, suffix: str = "") -> str:
    if pd.isna(value):
        return "—"
    text = f"{float(value):.{digits}f}"
    if language == "HU":
        text = text.replace(".", ",")
    return text + suffix


def _select_summary(
    summary: pd.DataFrame,
    season: int,
    week: int | None,
    market_key: str,
    sample_scope: str = "ALL_TRACKED",
) -> pd.Series | None:
    """Select one prepared aggregate without calculating metrics in the UI."""
    if summary.empty:
        return None
    if "sample_scope" not in summary.columns:
        if sample_scope != "ALL_TRACKED":
            return None
        sample_filter = pd.Series(True, index=summary.index)
    else:
        sample_filter = summary["sample_scope"].eq(sample_scope)
    rows = summary.loc[
        summary["season"].eq(season)
        & summary["market_key"].eq(market_key)
        & sample_filter
    ]
    rows = rows.loc[rows["week"].isna()] if week is None else rows.loc[rows["week"].eq(week)]
    if rows.empty:
        return None
    return rows.iloc[0]


def _filter_settlement(
    settlement: pd.DataFrame,
    season: int,
    week: int | None,
    market_key: str,
    settled_only: bool,
    clv_only: bool = False,
) -> pd.DataFrame:
    rows = settlement.loc[settlement["season"].eq(season)].copy()
    if week is not None:
        rows = rows.loc[rows["week"].eq(week)]
    if market_key != "ALL":
        rows = rows.loc[rows["market_key"].eq(market_key)]
    if settled_only:
        rows = rows.loc[rows["settlement_status"].eq("SETTLED")]
    if clv_only:
        rows = rows.loc[rows["is_clv"]]
    return rows.sort_values(["commence_time", "game_id", "market_key"])


def _result_label(value: object, language: Language) -> str:
    labels = {
        "WIN": ("Win", "Nyert"), "LOSS": ("Loss", "Vesztett"),
        "PUSH": ("Push", "Push"), "PENDING": ("Upcoming", "Közelgő"),
    }
    key = "PENDING" if pd.isna(value) else str(value)
    pair = labels.get(key, (key, key))
    return pair[1] if language == "HU" else pair[0]


def _selection_label(row: pd.Series) -> str:
    if row["market_key"] == "h2h":
        return str(row["home_team"] if str(row["outcome_type"]).lower() == "home" else row["away_team"])
    return str(row["outcome_name"])


def render_forward_performance(
    settlement: pd.DataFrame,
    summary: pd.DataFrame,
    language: Language = DEFAULT_LANGUAGE,
) -> None:
    """Render prepared forward-test results with explicit sample warnings."""
    if settlement.empty or summary.empty:
        empty_state(
            "Live tracking is active" if language == "EN" else "Az élő követés aktív",
            (
                "Results will appear after the first locked selections and completed games."
                if language == "EN" else
                "Az eredmények az első rögzített kiválasztások és lezárt meccsek után jelennek meg."
            ),
        )
        return

    st.warning(
        "A kis elemszámú forward minta nem bizonyít tartós nyereségességet. A backtest és az élő eredmények elkülönülnek."
        if language == "HU" else
        "A small forward sample is not evidence of sustainable profitability. Backtests and live results remain separate."
    )
    seasons = sorted(settlement["season"].dropna().astype(int).unique(), reverse=True)
    season = st.selectbox("Szezon" if language == "HU" else "Season", seasons)
    available_weeks = sorted(settlement.loc[settlement["season"].eq(season), "week"].dropna().astype(int).unique())
    all_weeks = "Összes hét" if language == "HU" else "All weeks"
    week_choice = st.selectbox("Hét" if language == "HU" else "Week", (all_weeks, *available_weeks))
    week = None if week_choice == all_weeks else int(week_choice)
    market_options = {
        ("Összes piac" if language == "HU" else "All markets"): "ALL",
        "Moneyline": "h2h", "Spread": "spreads", "Total": "totals",
    }
    market_label = st.selectbox("Piac" if language == "HU" else "Market", tuple(market_options))
    market_key = market_options[market_label]
    scope_labels = (
        {"Minden követett": "ALL_TRACKED", "Csak lezárt": "SETTLED_ONLY", "CLV-re jogosult": "CLV_ELIGIBLE"}
        if language == "HU" else
        {"All tracked": "ALL_TRACKED", "Settled only": "SETTLED_ONLY", "CLV eligible": "CLV_ELIGIBLE"}
    )
    scope_label = st.selectbox("Minta" if language == "HU" else "Scope", tuple(scope_labels))

    sample_scope = scope_labels[scope_label]
    aggregate = _select_summary(summary, season, week, market_key, sample_scope)
    rows = _filter_settlement(
        settlement, season, week, market_key,
        sample_scope == "SETTLED_ONLY", sample_scope == "CLV_ELIGIBLE",
    )
    if aggregate is None:
        empty_state(
            "Nincs összesített eredmény" if language == "HU" else "No aggregate available",
            "Válassz másik szűrést." if language == "HU" else "Choose another filter combination.",
        )
        return

    metrics = st.columns(4)
    with metrics[0]:
        metric_tile("Lezárt / követett" if language == "HU" else "Settled / tracked", f"{int(aggregate['settled_count'])} / {int(aggregate['tracked_count'])}")
    with metrics[1]:
        metric_tile("W–L–P", f"{int(aggregate['win_count'])}–{int(aggregate['loss_count'])}–{int(aggregate['push_count'])}")
    with metrics[2]:
        metric_tile("ROI", _number(aggregate["roi_percent"], language, 1, "%"), "green" if pd.notna(aggregate["roi_percent"]) and aggregate["roi_percent"] > 0 else "")
    with metrics[3]:
        metric_tile("Profit", _number(aggregate["total_profit_units"], language, 2, " u"))

    second = st.columns(4)
    with second[0]:
        metric_tile("Win rate", _number(100.0 * aggregate["win_rate"] if pd.notna(aggregate["win_rate"]) else None, language, 1, "%"))
    with second[1]:
        metric_tile("Átlagodds" if language == "HU" else "Average odds", _number(aggregate["average_decimal_odds"], language, 2))
    with second[2]:
        metric_tile("Max. drawdown", _number(aggregate["maximum_drawdown_units"], language, 2, " u"))
    with second[3]:
        metric_tile("Brier score", _number(aggregate["brier_score"], language, 4))

    st.caption(
        (f"Log Loss: {_number(aggregate['log_loss'], language, 4)} · CLV: {int(aggregate['clv_count'])} rögzített záró snapshot")
        if language == "HU" else
        (f"Log Loss: {_number(aggregate['log_loss'], language, 4)} · CLV: {int(aggregate['clv_count'])} qualifying close snapshots")
    )
    st.markdown("### " + ("Rögzített kiválasztások" if language == "HU" else "Locked selections"))
    if rows.empty:
        st.info("Ehhez a szűréshez még nincs lezárt eredmény." if language == "HU" else "No settled results match this filter yet.")
        return
    detail = rows.copy()
    detail["matchup"] = detail["away_team"].astype(str) + " @ " + detail["home_team"].astype(str)
    detail["selection"] = detail.apply(_selection_label, axis=1)
    detail["market"] = detail["market_key"].map(MARKET_LABELS)
    detail["odds"] = detail["entry_decimal_odds"].map(lambda value: format_decimal_odds(value, language))
    detail["result_display"] = detail.apply(
        lambda row: _result_label(row["result"] if row["settlement_status"] == "SETTLED" else None, language), axis=1
    )
    detail["profit_display"] = detail["profit_units"].map(lambda value: _number(value, language, 2, " u"))
    st.dataframe(
        detail[["week", "matchup", "market", "selection", "odds", "result_display", "profit_display"]],
        width="stretch", hide_index=True,
        column_config={
            "week": "Hét" if language == "HU" else "Week",
            "matchup": "Mérkőzés" if language == "HU" else "Matchup",
            "market": "Piac" if language == "HU" else "Market",
            "selection": "Kiválasztás" if language == "HU" else "Selection",
            "odds": "Entry odds",
            "result_display": "Eredmény" if language == "HU" else "Result",
            "profit_display": "Profit",
        },
    )
