"""Bilingual results with separate model and published-betting scopes."""

from __future__ import annotations

import math
import pandas as pd
import streamlit as st

from src.dashboard.components import empty_state, metric_tile
from src.dashboard.i18n import DEFAULT_LANGUAGE, Language
from src.dashboard.view_models import format_decimal_odds

MARKET_LABELS = {"ALL": "All", "h2h": "Moneyline", "spreads": "Spread", "totals": "Total"}


def has_live_results(model_results: pd.DataFrame, betting_results: pd.DataFrame) -> bool:
    """Expose the page once either verified result stream has real output."""
    return bool(not model_results.empty or (
        not betting_results.empty
        and "settlement_status" in betting_results
        and betting_results["settlement_status"].eq("SETTLED").any()
    ))


def has_settled_results(settlement: pd.DataFrame) -> bool:
    """Compatibility helper retained for rolling deployments and older tests."""
    return bool(not settlement.empty and "settlement_status" in settlement
                and settlement["settlement_status"].eq("SETTLED").any())


def _number(value: object, language: Language, digits: int = 1, suffix: str = "") -> str:
    if pd.isna(value):
        return "—"
    text = f"{float(value):.{digits}f}"
    return (text.replace(".", ",") if language == "HU" else text) + suffix


def prepare_model_result_rows(results: pd.DataFrame) -> pd.DataFrame:
    """Normalize immutable game results, including older published schemas."""
    if results.empty:
        return results.copy()
    rows = results.copy()
    home_won = rows["home_score"].gt(rows["away_score"])
    away_won = rows["away_score"].gt(rows["home_score"])
    rows["actual_winner"] = None
    rows.loc[home_won, "actual_winner"] = rows.loc[home_won, "home_team"]
    rows.loc[away_won, "actual_winner"] = rows.loc[away_won, "away_team"]
    rows["moneyline_winner_correct"] = rows["predicted_winner"].eq(rows["actual_winner"])
    outcome = home_won.astype(float)
    probability = pd.to_numeric(rows["home_win_probability"], errors="coerce").clip(1e-15, 1 - 1e-15)
    rows["moneyline_brier_loss"] = (probability - outcome) ** 2
    rows["moneyline_log_loss"] = -(outcome * probability.map(math.log) + (1 - outcome) * (1 - probability).map(math.log))
    rows["predicted_win_probability"] = rows["home_win_probability"].where(
        rows["predicted_winner"].eq(rows["home_team"]), rows["away_win_probability"]
    )
    rows["matchup"] = rows["away_team"].astype(str) + " @ " + rows["home_team"].astype(str)
    rows["final_score"] = (
        rows["away_team"].astype(str) + " " + rows["away_score"].astype("Int64").astype(str)
        + "–" + rows["home_team"].astype(str) + " " + rows["home_score"].astype("Int64").astype(str)
    )
    rows["actual_home_margin"] = rows["home_score"] - rows["away_score"]
    rows["actual_total_points"] = rows["home_score"] + rows["away_score"]
    rows["spread_error"] = rows["predicted_home_margin"] - rows["actual_home_margin"]
    rows["spread_absolute_error"] = rows["spread_error"].abs()
    rows["total_error"] = rows["predicted_total_points"] - rows["actual_total_points"]
    rows["total_absolute_error"] = rows["total_error"].abs()
    return rows.sort_values(["season", "week", "commence_time", "game_id"])


def results_updated_at(model_results: pd.DataFrame) -> pd.Timestamp | None:
    """Return the freshness of the FINAL evaluation dataset, not model refresh."""
    if model_results.empty or "result_evaluated_at" not in model_results:
        return None
    values = pd.to_datetime(model_results["result_evaluated_at"], utc=True, errors="coerce")
    return None if values.isna().all() else values.max()


def _select_summary(summary: pd.DataFrame, season: int, week: int | None, market_key: str) -> pd.Series | None:
    """Select an aggregate from the published-selection summary only."""
    if summary.empty:
        return None
    rows = summary.loc[summary["season"].eq(season) & summary["market_key"].eq(market_key)
                       & summary["sample_scope"].eq("ALL_TRACKED")]
    rows = rows.loc[rows["week"].isna()] if week is None else rows.loc[rows["week"].eq(week)]
    return None if rows.empty else rows.iloc[0]


def _filter_settlement(settlement: pd.DataFrame, season: int, week: int | None,
                       market_key: str, settled_only: bool = False,
                       clv_only: bool = False) -> pd.DataFrame:
    """Filter published-selection settlement; legacy flags remain compatibility-only."""
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
    labels = {"WIN": ("Win", "Nyert"), "LOSS": ("Loss", "Vesztett"),
              "PUSH": ("Push", "Push"), "PENDING": ("Upcoming", "Közelgő")}
    pair = labels.get("PENDING" if pd.isna(value) else str(value), (str(value), str(value)))
    return pair[1] if language == "HU" else pair[0]


def _selection_label(row: pd.Series) -> str:
    if row["market_key"] == "h2h":
        return str(row["home_team"] if str(row["outcome_type"]).lower() == "home" else row["away_team"])
    return str(row["outcome_name"])


def _entry_line(row: pd.Series, language: Language) -> str:
    if row["market_key"] == "h2h" or pd.isna(row["entry_point"]):
        return "—"
    point = float(row["entry_point"])
    value = f"{point:+g}" if row["market_key"] == "spreads" else f"{point:g}"
    return value.replace(".", ",") if language == "HU" else value


def betting_empty_message(language: Language) -> str:
    """Explain verified forward betting tracking without internal terminology."""
    return (
        "Még nincs lezárt Javasolt tipp. A fogadási teljesítmény csak a Betting Boardon publikált és kickoff előtt rögzített Javasolt tippek alapján kerül kiértékelésre."
        if language == "HU" else
        "No Recommended Picks have been settled yet. Betting performance is evaluated only from Recommended Picks published on the Betting Board and locked before kickoff."
    )


def _filters(rows: pd.DataFrame, language: Language, prefix: str, market: bool = False) -> tuple[int, int | None, str]:
    seasons = sorted(rows["season"].dropna().astype(int).unique(), reverse=True)
    season = st.selectbox("Szezon" if language == "HU" else "Season", seasons, key=f"{prefix}_season")
    weeks = sorted(rows.loc[rows["season"].eq(season), "week"].dropna().astype(int).unique())
    all_weeks = "Összes hét" if language == "HU" else "All weeks"
    chosen = st.selectbox("Hét" if language == "HU" else "Week", (all_weeks, *weeks), key=f"{prefix}_week")
    market_key = "ALL"
    if market:
        options = {("Összes piac" if language == "HU" else "All markets"): "ALL",
                   "Moneyline": "h2h", "Spread": "spreads", "Total": "totals"}
        label = st.selectbox("Piac" if language == "HU" else "Market", tuple(options), key=f"{prefix}_market")
        market_key = options[label]
    return season, None if chosen == all_weeks else int(chosen), market_key


def _regression_metrics(rows: pd.DataFrame, error_column: str) -> tuple[float, float, float]:
    errors = pd.to_numeric(rows[error_column], errors="coerce").dropna()
    if errors.empty:
        return float("nan"), float("nan"), float("nan")
    return float(errors.abs().mean()), float((errors.pow(2).mean()) ** 0.5), float(errors.mean())


def _margin_label(value: object, home_team: object, away_team: object,
                  language: Language) -> str:
    if pd.isna(value):
        return "—"
    margin = float(value)
    if abs(margin) < 0.05:
        return "Pick'em"
    team = str(home_team if margin > 0 else away_team)
    amount = _number(abs(margin), language, 1)
    return f"{team} {amount} ponttal" if language == "HU" else f"{team} by {amount}"


def _render_regression_results(rows: pd.DataFrame, language: Language,
                               *, market: str) -> None:
    is_spread = market == "spread"
    predicted = "predicted_home_margin" if is_spread else "predicted_total_points"
    actual = "actual_home_margin" if is_spread else "actual_total_points"
    error = "spread_error" if is_spread else "total_error"
    absolute = "spread_absolute_error" if is_spread else "total_absolute_error"
    mae, rmse, bias = _regression_metrics(rows, error)
    metrics = st.columns(4)
    with metrics[0]: metric_tile("Kiértékelt meccsek" if language == "HU" else "Games evaluated", str(len(rows)))
    with metrics[1]: metric_tile("MAE", _number(mae, language, 2))
    with metrics[2]: metric_tile("RMSE", _number(rmse, language, 2))
    with metrics[3]: metric_tile("Bias", _number(bias, language, 2))
    detail = rows.copy()
    if is_spread:
        detail["predicted_display"] = detail.apply(
            lambda row: _margin_label(row[predicted], row["home_team"], row["away_team"], language), axis=1
        )
        detail["actual_display"] = detail.apply(
            lambda row: _margin_label(row[actual], row["home_team"], row["away_team"], language), axis=1
        )
    else:
        detail["predicted_display"] = detail[predicted].map(lambda value: _number(value, language, 1))
        detail["actual_display"] = detail[actual].map(lambda value: _number(value, language, 0))
    detail["absolute_display"] = detail[absolute].map(lambda value: _number(value, language, 1))
    st.markdown("### " + ("Meccsenkénti modellértékelés" if language == "HU" else "Game-level model evaluation"))
    st.dataframe(
        detail[["week", "matchup", "predicted_display", "actual_display", "absolute_display"]],
        width="stretch", hide_index=True,
        column_config={
            "week": "Hét" if language == "HU" else "Week",
            "matchup": "Mérkőzés" if language == "HU" else "Matchup",
            "predicted_display": ("Várt margin" if language == "HU" else "Predicted margin") if is_spread else ("Várt total" if language == "HU" else "Predicted total"),
            "actual_display": ("Tényleges margin" if language == "HU" else "Actual margin") if is_spread else ("Tényleges total" if language == "HU" else "Actual total"),
            "absolute_display": "Abszolút hiba" if language == "HU" else "Absolute error",
        },
    )


def _render_model_results(model_results: pd.DataFrame, language: Language) -> None:
    st.caption("A modell előrejelzési minősége, a fogadási tippektől függetlenül."
               if language == "HU" else "Prediction quality of the model, independent of betting selections.")
    if model_results.empty:
        empty_state("Még nincs lezárt modellértékelés" if language == "HU" else "No completed model evaluation yet",
                    "Az első FINAL mérkőzés után jelenik meg." if language == "HU" else "Results appear after the first FINAL game.")
        return
    rows = prepare_model_result_rows(model_results)
    season, week, _ = _filters(rows, language, "model_results")
    rows = rows.loc[rows["season"].eq(season)]
    if week is not None:
        rows = rows.loc[rows["week"].eq(week)]
    moneyline_tab, spread_tab, total_tab = st.tabs(("Moneyline", "Spread", "Total"))
    with moneyline_tab:
        correct, total = int(rows["moneyline_winner_correct"].sum()), len(rows)
        metrics = st.columns(4)
        with metrics[0]: metric_tile("Kiértékelt meccsek" if language == "HU" else "Games evaluated", str(total))
        with metrics[1]: metric_tile("Helyes predikciók / Pontosság" if language == "HU" else "Correct predictions / Accuracy", f"{correct}/{total} · {_number(100 * correct / total if total else None, language, 1, '%')}")
        with metrics[2]: metric_tile("Brier Score", _number(rows["moneyline_brier_loss"].mean(), language, 4))
        with metrics[3]: metric_tile("Log Loss", _number(rows["moneyline_log_loss"].mean(), language, 4))
        detail = rows.copy()
        detail["probability"] = detail["predicted_win_probability"].map(lambda x: _number(100 * x, language, 1, "%"))
        detail["correct"] = detail["moneyline_winner_correct"].map({True: "Helyes" if language == "HU" else "Correct", False: "Helytelen" if language == "HU" else "Incorrect"})
        st.markdown("### " + ("Meccsenkénti modellértékelés" if language == "HU" else "Game-level model evaluation"))
        st.dataframe(detail[["week", "matchup", "predicted_winner", "probability", "actual_winner", "final_score", "correct"]], width="stretch", hide_index=True,
                     column_config={"week": "Hét" if language == "HU" else "Week", "matchup": "Mérkőzés" if language == "HU" else "Matchup",
                                    "predicted_winner": "Várt győztes" if language == "HU" else "Predicted winner",
                                    "probability": "Meccs előtti valószínűség" if language == "HU" else "Pre-game probability",
                                    "actual_winner": "Tényleges győztes" if language == "HU" else "Actual winner",
                                    "final_score": "Végeredmény" if language == "HU" else "Final score",
                                    "correct": "Értékelés" if language == "HU" else "Evaluation"})
    with spread_tab:
        _render_regression_results(rows, language, market="spread")
    with total_tab:
        _render_regression_results(rows, language, market="total")


def _render_betting_results(settlement: pd.DataFrame, summary: pd.DataFrame, language: Language) -> None:
    st.caption("A Betting Boardon Javasolt tippként publikált és a meccs kezdete előtt rögzített tippek forward teljesítménye."
               if language == "HU" else "Forward performance of Recommended Picks published on the Betting Board and locked before kickoff.")
    if settlement.empty or summary.empty:
        st.info(betting_empty_message(language))
        return
    season, week, market_key = _filters(settlement, language, "betting_results", market=True)
    aggregate = _select_summary(summary, season, week, market_key)
    rows = _filter_settlement(settlement, season, week, market_key)
    if aggregate is None:
        st.info("Ehhez a szűréshez még nincs publikált jelzés." if language == "HU" else "No published selections match this filter yet.")
        return
    first = st.columns(4)
    with first[0]: metric_tile("Lezárt / Javasolt tippek" if language == "HU" else "Settled / Recommended Picks", f"{int(aggregate['settled_count'])} / {int(aggregate['tracked_count'])}")
    with first[1]: metric_tile("W–L–P", f"{int(aggregate['win_count'])}–{int(aggregate['loss_count'])}–{int(aggregate['push_count'])}")
    with first[2]: metric_tile("Nettó unit" if language == "HU" else "Net units", _number(aggregate["total_profit_units"], language, 2, " u"))
    with first[3]: metric_tile("ROI", _number(aggregate["roi_percent"], language, 1, "%"))
    second = st.columns(2)
    with second[0]: metric_tile("Átlagodds" if language == "HU" else "Average odds", _number(aggregate["average_decimal_odds"], language, 2))
    with second[1]: metric_tile("CLV", "—", help_text="Későbbi closing snapshot adatokkal válik elérhetővé." if language == "HU" else "Available later with qualifying closing snapshots.")
    st.warning("A kis elemszámú forward minta nem bizonyít tartós nyereségességet." if language == "HU" else "A small forward sample is not evidence of sustainable profitability.")
    st.markdown("### " + ("Publikált Javasolt tippek" if language == "HU" else "Published Recommended Picks"))
    detail = rows.copy()
    detail["matchup"] = detail["away_team"].astype(str) + " @ " + detail["home_team"].astype(str)
    detail["selection"] = detail.apply(_selection_label, axis=1)
    detail["market"] = detail["market_key"].map(MARKET_LABELS)
    detail["line"] = detail.apply(lambda row: _entry_line(row, language), axis=1)
    detail["odds"] = detail["entry_decimal_odds"].map(lambda x: format_decimal_odds(x, language))
    detail["result_display"] = detail.apply(lambda row: _result_label(row["result"] if row["settlement_status"] == "SETTLED" else None, language), axis=1)
    detail["profit_display"] = detail["profit_units"].map(lambda x: _number(x, language, 2, " u"))
    st.dataframe(detail[["week", "matchup", "market", "selection", "line", "odds", "result_display", "profit_display"]], width="stretch", hide_index=True,
                 column_config={"week": "Hét" if language == "HU" else "Week", "matchup": "Mérkőzés" if language == "HU" else "Matchup", "market": "Piac" if language == "HU" else "Market",
                                "selection": "Jelzés" if language == "HU" else "Selection", "line": "Entry line", "odds": "Entry odds",
                                "result_display": "Eredmény" if language == "HU" else "Result", "profit_display": "Profit"})


def render_forward_performance(model_results: pd.DataFrame, betting_settlement: pd.DataFrame,
                               betting_summary: pd.DataFrame,
                               language: Language = DEFAULT_LANGUAGE) -> None:
    """Render model quality and published betting performance separately."""
    st.info("A modelleredmények a predikciók teljesítményét, a fogadási eredmények pedig kizárólag a Betting Boardon publikált Javasolt tippek teljesítményét mutatják."
            if language == "HU" else "Model Results measure prediction performance, while Betting Results track only Recommended Picks published on the Betting Board.")
    tab_labels = (("Modelleredmények", "Fogadási eredmények")
                  if language == "HU" else ("Model Results", "Betting Results"))
    model_tab, betting_tab = st.tabs(tab_labels)
    with model_tab:
        _render_model_results(model_results, language)
    with betting_tab:
        _render_betting_results(betting_settlement, betting_summary, language)
