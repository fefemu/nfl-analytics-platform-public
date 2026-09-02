"""Central EN/HU UI copy with stable analytics terminology."""

from typing import Literal


Language = Literal["EN", "HU"]
DEFAULT_LANGUAGE: Language = "EN"

# Technical product terms intentionally remain unchanged in Hungarian: pipeline,
# Brier score, log loss, calibration, Spread, Totals, Moneyline, edge, expected
# value, primary, fallback, holdout, backtest, feature, routing, Monte Carlo and Elo.
COPY: dict[str, dict[Language, str]] = {
    "nav_overview": {"EN": "Weekly Overview", "HU": "Heti áttekintés"},
    "nav_games": {"EN": "Game Center", "HU": "Game Center"},
    "nav_betting": {"EN": "Betting Board", "HU": "Betting Board"},
    "nav_teams": {"EN": "Teams", "HU": "Csapatok"},
    "nav_simulator": {"EN": "Season Simulator", "HU": "Szezonszimulátor"},
    "nav_lab": {"EN": "Data Science Lab", "HU": "Data Science Lab"},
    "nav_about": {"EN": "About the Platform", "HU": "A platformról"},
    "language": {"EN": "Language", "HU": "Nyelv"},
    "eyebrow": {"EN": "2026 FORWARD ANALYTICS", "HU": "2026 FORWARD ANALYTICS"},
    "subtitle": {"EN": "Production models, market intelligence and season simulations in one workspace.", "HU": "Éles modellek, piaci elemzések és szezon-szimulációk egy helyen."},
    "subtitle_overview": {"EN": "Matchups, expected scores and win probabilities for the selected week.", "HU": "A kiválasztott hét mérkőzései, várható eredményei és győzelmi valószínűségei."},
    "subtitle_games": {"EN": "A detailed model and market view of a selected matchup.", "HU": "Egy kiválasztott mérkőzés részletes modell- és piaci elemzése."},
    "subtitle_betting": {"EN": "Markets where model estimates differ most from current bookmaker pricing.", "HU": "Azok a piacok, ahol a modell becslése leginkább eltér a fogadóirodák aktuális árazásától."},
    "subtitle_teams": {"EN": "Current NFL rosters and starting formations by offensive and defensive unit.", "HU": "Az NFL-csapatok aktuális kerete és kezdő felállása támadó- és védőegység szerinti bontásban."},
    "subtitle_simulator": {"EN": "Probabilistic season outcomes from 10,000 Monte Carlo simulations.", "HU": "A teljes NFL-szezon 10 000 Monte Carlo-szimulációból becsült lehetséges kimenetelei."},
    "subtitle_lab": {"EN": "How are NFL forecasts built and tested?", "HU": "Hogyan készülnek és hogyan teszteljük az NFL-előrejelzéseket?"},
    "subtitle_about": {"EN": "What the platform does, how it is built and how its outputs should be used.", "HU": "Mit tud a platform, hogyan épül fel, és hogyan érdemes értelmezni az eredményeit."},
    "system_state": {"EN": "System state", "HU": "Rendszerállapot"},
    "ready": {"EN": "Ready", "HU": "Kész"},
    "setup": {"EN": "Setup", "HU": "Beállítás"},
    "model_suite": {"EN": "Model suite", "HU": "Modellcsomag"},
    "markets_3": {"EN": "3 markets", "HU": "3 piactípus"},
    "simulation": {"EN": "Simulation", "HU": "Monte Carlo"},
    "runs_10000": {"EN": "10,000 runs", "HU": "10 000 futás"},
    "last_refresh": {"EN": "Last successful refresh", "HU": "Utolsó sikeres frissítés"},
    "not_available": {"EN": "Not available", "HU": "Nem elérhető"},
    "data_ready": {"EN": "DATA READY", "HU": "ADATOK ELÉRHETŐK"},
    "refresh_required": {"EN": "REFRESH REQUIRED", "HU": "FRISSÍTÉS SZÜKSÉGES"},
    "responsible": {"EN": "Analytical estimates, not guarantees. Bet responsibly.", "HU": "Analitikai becslések, nem garanciák. Fogadj felelősen."},
    "week": {"EN": "Week", "HU": "Hét"},
    "games": {"EN": "Games", "HU": "Meccsek"},
    "most_even": {"EN": "Most even matchup", "HU": "Legkiegyenlítettebb meccs"},
    "highest_total": {"EN": "Highest model total", "HU": "Legmagasabb várható összpontszám"},
    "matchups": {"EN": "Matchups", "HU": "Meccsek"},
    "select_matchup": {"EN": "Select matchup", "HU": "Válassz meccset"},
    "why_model": {"EN": "What drives the model's prediction?", "HU": "Mi alapján várja ezt az eredményt a modell?"},
    "market_comparison": {"EN": "Market comparison", "HU": "Piaci összehasonlítás"},
    "technical_routing": {"EN": "Technical routing and model identifiers", "HU": "Technikai útválasztás és modellazonosítók"},
    "market": {"EN": "Market", "HU": "Piac"},
    "minimum_ev": {"EN": "Minimum EV %", "HU": "Minimum EV %"},
    "minimum_model_probability": {"EN": "Minimum model probability %", "HU": "Minimális modellvalószínűség %"},
    "minimum_books": {"EN": "Minimum books", "HU": "Fogadóirodák minimális száma"},
    "positive_ev_only": {"EN": "Positive EV only", "HU": "Csak pozitív EV"},
    "top_candidates": {"EN": "Top picks", "HU": "Top tippek"},
    "market_detail": {"EN": "Market detail", "HU": "Részletes piaci adatok"},
    "teams": {"EN": "Teams", "HU": "Csapatok"},
    "simulations": {"EN": "Simulations", "HU": "Szimulációk"},
    "top_expected": {"EN": "Top expected wins", "HU": "Legmagasabb várható győzelemszám"},
    "generated": {"EN": "Generated", "HU": "Generálva"},
    "leaders": {"EN": "Expected-wins leaders", "HU": "Várható győzelmek – élmezőny"},
    "all_team_outlook": {"EN": "All-team outlook", "HU": "Teljes liga előrejelzése"},
    "team_distribution": {"EN": "Team distribution", "HU": "Csapateloszlás"},
    "team": {"EN": "Team", "HU": "Csapat"},
    "dynamic_frozen": {"EN": "Dynamic vs frozen Elo", "HU": "Dinamikus és rögzített Elo összehasonlítása"},
    "lab_intro": {"EN": "Curated evidence behind the production models.", "HU": "Az éles modelleket alátámasztó, válogatott bizonyítékok."},
    "production_model": {"EN": "Production model", "HU": "Éles modell"},
    "model_comparison": {"EN": "Model comparison", "HU": "Modellek összehasonlítása"},
    "season_stability": {"EN": "Season stability", "HU": "Stabilitás szezononként"},
    "methodology": {"EN": "Methodology", "HU": "Módszertan"},
    "data_sources": {"EN": "Data sources", "HU": "Adatforrások"},
    "limitations": {"EN": "Limitations", "HU": "Korlátok"},
}


def tr(language: Language, key: str) -> str:
    """Return UI copy in the selected language."""

    try:
        return COPY[key][language]
    except KeyError as error:
        raise KeyError(f"Missing dashboard translation: {key} / {language}") from error
