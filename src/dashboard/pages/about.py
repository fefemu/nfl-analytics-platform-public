"""Bilingual platform story, architecture and author profile."""

import base64
from pathlib import Path

import streamlit as st

from src.dashboard.i18n import DEFAULT_LANGUAGE, Language


ASSET_DIRECTORY = Path(__file__).resolve().parents[1] / "assets"
ARCHITECTURE_DIAGRAM = ASSET_DIRECTORY / "platform_architecture.svg"
DATA_MODEL_DIAGRAM = ASSET_DIRECTORY / "platform_data_model.svg"
DATA_FLOW_DIAGRAM = ASSET_DIRECTORY / "platform_data_flow.svg"
GITHUB_URL = "https://github.com/fefemu/nfl-analytics-platform-public"
LINKEDIN_URL = "https://www.linkedin.com/in/ferenc-kaizer-038625b0/"
EMAIL_ADDRESS = "kaizer.ferenc88@gmail.com"


DIAGRAM_TRANSLATIONS = {
    "platform_architecture.svg": {
        "Technológiai architektúra": "Technology architecture",
        "Milyen technológiai komponensekből épül fel a platform?": "Which technology components power the platform?",
        "Adatforrások": "Data sources",
        "Külső futball- és piaci adatok": "External football and market data",
        "Adatfeldolgozás": "Data processing",
        "Begyűjtés · tisztítás · feature engineering": "Ingestion · cleaning · feature engineering",
        "Adattárolás": "Data storage",
        "Lokális analitikai adattár": "Local analytics store",
        "és lekérdezések": "and queries",
        "Modellezés és szimuláció": "Modeling and simulation",
        "Predikciók és szezonkimenetek": "Predictions and season outcomes",
        "Megjelenítés": "Presentation",
        "Interaktív webes felület": "Interactive web interface",
        "Fejlesztés és minőségbiztosítás": "Development and quality assurance",
    },
    "platform_data_model.svg": {
        "DuckDB adatmodell": "DuckDB data model",
        "mind a 63 aktuális fizikai tábla": "all 63 current physical tables",
        "1. Forrásadatok — RAW": "1. Source data — RAW",
        "Mérkőzés és játékos": "Games and players",
        "Depth chart és sérülés": "Depth charts and injuries",
        "Odds snapshotok": "Odds snapshots",
        "2. Tisztított és egységesített adatok — PROCESSED": "2. Cleaned and standardized data — PROCESSED",
        "Mérkőzés és teljesítmény": "Games and performance",
        "Játékos és depth chart": "Players and depth charts",
        "Külső rating és odds": "External ratings and odds",
        "3. Elemzési és modellezési réteg — ANALYTICS": "3. Analytics and modeling layer — ANALYTICS",
        "Feature-ök": "Features",
        "Modellezési adatok és governance": "Modeling data and governance",
        "Kapcsolás: game_id": "Join key: game_id",
        "Játékoskapcsolás: gsis_id": "Player join: gsis_id",
        "Csapatkapcsolás: team": "Team join: team",
        "Historikus betting audit": "Historical betting audit",
        "Meccs-, market- és időpontkulcsok": "Game, market and timestamp keys",
        "a historikus kiértékeléshez.": "for historical evaluation.",
        "4. Alkalmazási kimenetek — ANALYTICS OUTPUT": "4. Application outputs — ANALYTICS OUTPUT",
        "Aktuális előrejelzések": "Current predictions",
        "Betting és value": "Betting and value",
        "Szezon-szimuláció": "Season simulation",
        "Jelmagyarázat:": "Legend:",
        "adattranszformáció": "data transformation",
        "logikai kapcsolat (nem deklarált FK)": "logical relationship (not a declared FK)",
    },
    "platform_data_flow.svg": {
        "Adatfrissítési folyamat": "Data refresh workflow",
        "Három fázisban a frissítés indításától az új eredmények megjelenéséig": "Three phases from refresh start to published results",
        "ADAT ÉS MODELLEZÉS": "DATA AND MODELING",
        "1. Frissítés indítása": "1. Start refresh",
        "Online API vagy lokális": "Online API or local",
        "odds snapshot mód": "odds snapshot mode",
        "2. Auditbejegyzés": "2. Audit record",
        "státusz: RUNNING": "status: RUNNING",
        "3. Modellezési pipeline": "3. Modeling pipeline",
        "→ modeling dataset/splits/governance → predictionök": "→ modeling dataset/splits/governance → predictions",
        "→ várt pontok → szezon-szimuláció": "→ implied scores → season simulation",
        "A builderek validáció után commitolnak.": "Builders commit after validation.",
        "Modellezés": "Modeling",
        "sikeres?": "successful?",
        "PIAC ÉS PUBLIKÁLÁS": "MARKET AND PUBLISHING",
        "4. Odds pipeline": "4. Odds pipeline",
        "Piaci adatok és": "Market data and",
        "számítások rendben?": "calculations valid?",
        "Kickoff előtti előrejelzések rögzítése": "Archive pre-kickoff predictions",
        "későbbi kiértékeléshez": "for later evaluation",
        "6. Sikeres lezárás": "6. Successful completion",
        "státusz: SUCCESS": "status: SUCCESS",
        "KISZOLGÁLÁS": "SERVING",
        "7. Validált DuckDB-outputok": "7. Validated DuckDB outputs",
        "prediction · betting · simulation táblák": "prediction · betting · simulation tables",
        "8. Streamlit megjelenítés": "8. Streamlit presentation",
        "Read-only hozzáférés az új outputokhoz": "Read-only access to new outputs",
        "9. Adatok frissítve": "9. Data refreshed",
        "Legutóbbi sikeres output időpontja": "Latest successful output timestamp",
        "Hibaág: futás megszakítása": "Failure path: stop the run",
        "státusz: FAILED · hibaüzenet mentése · nincs Forward archive": "status: FAILED · save error message · no Forward archive",
        "nem": "no",
        "igen": "yes",
    },
}


def _localized_diagram(path: Path, language: Language) -> str:
    """Return an isolated SVG data URI translated for the active language."""

    svg = path.read_text(encoding="utf-8")
    if language == "EN":
        for hungarian, english in DIAGRAM_TRANSLATIONS[path.name].items():
            svg = svg.replace(hungarian, english)
    encoded = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


def _render_diagram(path: Path, language: Language) -> None:
    uri = _localized_diagram(path, language)
    st.markdown(
        f'<img src="{uri}" style="width:100%;height:auto" '
        'alt="Platform technical diagram">',
        unsafe_allow_html=True,
    )


def _feature_cards(language: Language) -> None:
    items = (
        (
            ("Heti előrejelzések", "Győzelmi esélyek, várható pontkülönbség, összpontszám és csapatpontszámok."),
            ("Game Center", "Mérkőzésenként részletes modellelemzés, előrejelzések és a legfontosabb háttéradatok."),
            ("Piaci elemzés", "Moneyline, Spread és Total oddsok összevetése a modell becsléseivel."),
            ("Csapatok", "Aktuális keretek, kezdő felállások, csapaterősség és egységszintű mutatók."),
            ("Szezonszimuláció", "10 000 Monte Carlo-futásból várható győzelmek és lehetséges szezonkimenetek."),
            ("Data Science Lab", "A modellek, mérőszámok, validáció és történelmi elemzések közérthető bemutatása."),
        ) if language == "HU" else (
            ("Weekly forecasts", "Win probabilities, expected margin, total points and implied team scores."),
            ("Game Center", "Detailed matchup-level model analysis, forecasts and important context."),
            ("Market analysis", "Moneyline, Spread and Total prices compared with model estimates."),
            ("Teams", "Current rosters, starting formations, team strength and unit-level indicators."),
            ("Season simulation", "Expected wins and possible season outcomes from 10,000 Monte Carlo runs."),
            ("Data Science Lab", "Accessible explanations of models, metrics, validation and historical analysis."),
        )
    )
    columns = st.columns(3)
    for index, (title, body) in enumerate(items):
        with columns[index % 3]:
            st.markdown(
                '<div class="nap-card">'
                f'<div class="nap-panel-title">{title}</div>'
                f'<div class="nap-muted">{body}</div></div>',
                unsafe_allow_html=True,
            )


def _data_flow(language: Language) -> None:
    steps = (
        (
            ("1", "Adatgyűjtés", "Külső adatforrások letöltése és a nyers állapot változatlan megőrzése."),
            ("2", "Adatelőkészítés", "Azonosítók, időpontok, csapatok, játékosok, sérülések és piaci adatok egységesítése."),
            ("3", "Változók előállítása", "Csapaterősség, QB-helyzet, sérülések, forma, pihenés, időjárás és további, kickoff előtt ismert információk előállítása."),
            ("4", "Modellezés", "Külön modellek készítik a győzelmi valószínűség, a várható pontkülönbség és az összpontszám becslését."),
            ("5", "Validáció", "Automatizált tesztek, adatminőségi ellenőrzések és modellvalidáció futtatása."),
            ("6", "Publikálás", "A sikeresen validált eredmények bekerülnek az analitikai rétegbe, majd megjelennek a webes felületen."),
        ) if language == "HU" else (
            ("1", "Ingest", "Download source data and retain replayable, unchanged snapshots."),
            ("2", "Normalize", "Standardize identifiers, times, teams, injuries and bookmaker prices."),
            ("3", "Features", "Pre-kickoff team, QB, injury, weather and rest indicators."),
            ("4", "Models", "Separate win, Spread and Total predictions plus season simulation."),
            ("5", "Validate", "Automated tests and SQL quality checks on important outputs."),
            ("6", "Present", "Load validated analytics tables into the read-only Streamlit interface."),
        )
    )
    columns = st.columns(3)
    for index, (number, title, body) in enumerate(steps):
        with columns[index % 3]:
            st.markdown(
                '<div class="nap-card">'
                f'<div class="nap-eyebrow">{number}</div>'
                f'<div class="nap-panel-title">{title}</div>'
                f'<div class="nap-muted">{body}</div></div>',
                unsafe_allow_html=True,
            )


def _sources(language: Language) -> None:
    rows = (
        (
            ("nflverse", "Schedule, play-by-play, játékos-, roster-, depth chart-, sérülés- és csapatadatok."),
            ("nfelo / nfelounits", "Külső csapaterősség, QB adjustment, publikált valószínűségek és egységszintű teljesítménymutatók."),
            ("The Odds API", "Aktuális Moneyline, Spread és Total piacok, oddsok és fogadóirodák."),
            ("Saját feldolgozási réteg", "A forrásokból időrendhelyes feature-ök, modeling datasetek és ellenőrzött üzleti kimenetek készülnek."),
        ) if language == "HU" else (
            ("nflverse", "Schedule, play-by-play, player, roster, depth-chart, injury and team data."),
            ("nfelo / nfelounits", "External team strength, QB adjustments, published probabilities and unit performance."),
            ("The Odds API", "Current Moneyline, Spread and Total markets, prices and bookmakers."),
            ("Platform processing layer", "Chronological features, modeling datasets and validated business outputs built from source data."),
        )
    )
    columns = st.columns(2)
    for index, (source, purpose) in enumerate(rows):
        with columns[index % 2]:
            st.markdown(
                '<div class="nap-card">'
                f'<div class="nap-panel-title">{source}</div>'
                f'<div class="nap-muted">{purpose}</div></div>',
                unsafe_allow_html=True,
            )


def _model_inputs(language: Language) -> None:
    cards = (
        (
            (
                "Győzelmi valószínűség",
                "- külső nfelo csapaterősség;\n- QB-helyzet és QB adjustment;\n"
                "- támadó-, védő- és special teams sérülési terhelés;\n"
                "- publikált nfelo valószínűség;\n- teljes vagy hiányos aktuális adatokhoz validált külön modell.",
            ),
            (
                "Várható pontkülönbség",
                "- külső nfelo csapaterősség-különbség;\n"
                "- külső nfelo QB adjustment-különbség.\n\n"
                "A pozitív becslés a hazai, a negatív a vendégcsapat felé mutat.",
            ),
            (
                "Várható összpontszám",
                "- támadó EPA és a védelem által engedett EPA;\n- irányítóhelyzet;\n"
                "- fedett vagy szabadtéri stadion;\n- hőmérséklet és szél;\n"
                "- a liga közelmúltbeli pontszerzési környezete.",
            ),
        ) if language == "HU" else (
            (
                "Win probability",
                "- external nfelo team strength;\n- QB context and QB adjustment;\n"
                "- offense, defense and special-teams injury burden;\n"
                "- published nfelo probability;\n- separately validated models for complete and incomplete inputs.",
            ),
            (
                "Expected point differential",
                "- external nfelo team-strength difference;\n"
                "- external nfelo QB-adjustment difference.\n\n"
                "A positive estimate favors the home team; a negative one favors the away team.",
            ),
            (
                "Expected total points",
                "- offensive EPA and defensive EPA allowed;\n- quarterback context;\n"
                "- indoor or outdoor venue;\n- temperature and wind;\n"
                "- the league's recent scoring environment.",
            ),
        )
    )
    columns = st.columns(3)
    for column, (title, body) in zip(columns, cards, strict=True):
        with column:
            st.markdown(f"#### {title}")
            st.markdown(body)
    st.caption(
        "Nem minden modell használ minden felsorolt változót: a győzelmi, Spread és Total modellek eltérő célokra készülnek."
        if language == "HU" else
        "Not every model uses every listed input: win, Spread and Total models solve different prediction tasks."
    )


def _validation(language: Language) -> None:
    st.write(
        "A modelleket mindig időrendben teszteljük: egy adott mérkőzés előrejelzéséhez "
        "csak olyan információ használható, amely a kezdőrúgás előtt már ismert volt. "
        "A fejlesztés során korábbi szezonokon végzett backtest, elkülönített holdout minta "
        "és a 2026-os szezon valódi előrejelzéseiből álló forward test segít ellenőrizni, "
        "hogy a modellek új adatokon is megfelelően működnek-e."
        if language == "HU" else
        "Models are compared with chronological backtests that train on earlier seasons "
        "and predict later games. The final candidate was checked once on the 2025 holdout; "
        "archived pre-kickoff 2026 predictions form the forward test. Shifted rolling "
        "features protect against data leakage."
    )
    st.info(
        "A részletes módszertan, a modell-összehasonlítás, a Brier score, a kalibráció és a hatásvizsgálatok a Data Science Lab oldalon találhatók."
        if language == "HU" else
        "Detailed methodology, Brier score, model comparison and impact analysis are available in the Data Science Lab."
    )


def _technology(language: Language) -> None:
    technologies = (
        (
            ("Python + pandas", "adatbegyűjtés, tisztítás, feature engineering és pipeline orchestration"),
            ("DuckDB + SQL", "lokális analitikai adattár, transzformációk és quality checkek"),
            ("scikit-learn", "Logistic Regression és Ridge modellek"),
            ("pytest", "automatizált unit, integration és regression tesztek"),
            ("Plotly + Streamlit", "interaktív vizualizációk és webes felület"),
            ("Git + GitHub", "verziókezelés és a forráskód publikálása"),
        ) if language == "HU" else (
            ("Python + pandas", "ingestion, cleaning, feature engineering and pipeline orchestration"),
            ("DuckDB + SQL", "local analytics storage, transformations and quality checks"),
            ("scikit-learn", "Logistic Regression and Ridge models"),
            ("pytest", "automated unit, integration and regression tests"),
            ("Plotly + Streamlit", "interactive visualizations and the web interface"),
            ("Git + GitHub", "version control and source-code publication"),
        )
    )
    for technology, purpose in technologies:
        st.markdown(f"**{technology}** — {purpose}")


def _diagrams(language: Language) -> None:
    architecture, data_model, data_flow = st.tabs(
        ("Architektúra", "Adatmodell", "Adatfrissítési folyamat")
        if language == "HU" else
        ("Architecture", "Data model", "Data flow")
    )
    with architecture:
        _render_diagram(ARCHITECTURE_DIAGRAM, language)
        st.caption(
            "A diagram a platform jelenlegi technológiai felépítését mutatja az adatforrásoktól a felhasználói felületig."
            if language == "HU" else
            "The current operational path from source systems to Streamlit; no planned components are shown."
        )
    with data_model:
        _render_diagram(DATA_MODEL_DIAGRAM, language)
        st.caption(
            "A DuckDB aktuális fizikai táblái: 9 RAW, 12 processed és 42 analytics tábla. A szaggatott kapcsolatok kódban használt logikai joinokat jelölnek, nem deklarált idegen kulcsokat."
            if language == "HU" else
            "The current physical DuckDB tables: 9 RAW, 12 processed and 42 analytics tables. Dashed relationships are logical joins used in code, not declared foreign keys."
        )
    with data_flow:
        _render_diagram(DATA_FLOW_DIAGRAM, language)
        st.caption(
            "Az auditált in-season refresh tényleges sorrendje. Sikertelen validáció esetén a futás FAILED státusszal leáll, és nem készül forward archive."
            if language == "HU" else
            "The actual audited in-season refresh order. Failed validation stops the run with FAILED status and no forward archive is created."
        )


def _author(language: Language) -> None:
    if language == "HU":
        st.write(
            "Az **NFL Analytics Platform** saját Data Science projektem, amelyben az NFL, "
            "a sportfogadás és az adatelemzés iránti érdeklődésemet kapcsoltam össze. "
            "A célom nem csupán egy előrejelző modell elkészítése volt, hanem egy teljes "
            "folyamat felépítése az adatok begyűjtésétől és feldolgozásától a modellezésen "
            "és tesztelésen át egészen a webes megjelenítésig."
        )
        st.write(
            "A fejlesztés során AI-eszközöket is használok, elsősorban a kódolás, a "
            "refaktorálás és a hibakeresés támogatására. A modellezési döntéseket, az "
            "architektúra kialakítását, a tesztelést és az eredmények értékelését én végzem."
        )
        labels = ("GitHub – Forráskód", "LinkedIn – Kapcsolat", "E-mail")
    else:
        st.write(
            "The **NFL Analytics Platform** is my personal Data Science project combining "
            "my interest in the NFL, sports betting and analytics. My goal was not merely "
            "to train a prediction model, but to build the complete path from ingestion and "
            "processing through modeling and testing to a public web interface."
        )
        st.write(
            "I also use AI tools to support coding, refactoring and debugging. I make the "
            "modeling decisions, design the architecture, define the tests and evaluate the results."
        )
        labels = ("GitHub – Source code", "LinkedIn – Contact", "Email")
    st.markdown(
        f"[{labels[0]}]({GITHUB_URL}) &nbsp; | &nbsp; "
        f"[{labels[1]}]({LINKEDIN_URL}) &nbsp; | &nbsp; "
        f"[{labels[2]}](mailto:{EMAIL_ADDRESS})",
        unsafe_allow_html=True,
    )


def render_about(language: Language = DEFAULT_LANGUAGE) -> None:
    """Render a non-repetitive, end-user-oriented platform overview."""

    if language == "HU":
        st.markdown("### Mi az NFL Analytics Platform?")
        st.write(
            "Az NFL Analytics Platform egy független NFL elemzési projekt, amely adatok "
            "és statisztikai modellek segítségével készít előrejelzéseket. Minden aktuális "
            "mérkőzéshez megbecsüli a győzelmi esélyeket, a várható pontkülönbséget, az "
            "összpontszámot és a csapatok várható pontszámát. A becsléseket a fogadóirodák "
            "aktuális áraival is összeveti, így láthatóvá válik, mely mérkőzéseket értékeli "
            "eltérően a modell és a piac."
        )
        headings = (
            "Mit kap a felhasználó?", "Hogyan lesz a nyers adatból előrejelzés?",
            "Adatforrások", "Mit vesznek figyelembe a modellek?",
            "Hogyan ellenőrizzük a modelleket?", "Technikai háttér",
            "A projektről és rólam",
        )
    else:
        st.markdown("### What is the NFL Analytics Platform?")
        st.write(
            "The NFL Analytics Platform is an **independent, data- and model-driven NFL "
            "analytics product**. It estimates win probability, expected margin, expected "
            "total points and implied team scores, then compares those estimates with "
            "current bookmaker prices to identify where the model and market disagree."
        )
        headings = (
            "What does the user get?", "How does raw data become a prediction?",
            "Data sources", "What do the models consider?", "How are models checked?",
            "Technical background", "About the project and me",
        )
    st.markdown(f"### {headings[0]}")
    _feature_cards(language)
    st.markdown(f"### {headings[1]}")
    _data_flow(language)
    st.markdown(f"### {headings[2]}")
    _sources(language)
    st.markdown(f"### {headings[3]}")
    _model_inputs(language)
    st.markdown(f"### {headings[4]}")
    _validation(language)
    st.markdown(f"### {headings[5]}")
    st.write(
        "A platform három nézetben mutatja be a technikai megvalósítást: az Architektúra "
        "a használt technológiákat, az Adatmodell a DuckDB adatrétegeit és tábláit, az "
        "Adatfrissítési folyamat pedig a teljes frissítés menetét mutatja be."
        if language == "HU" else
        "Three views explain the implementation: Architecture shows the technologies, "
        "Data model shows DuckDB layers and tables, and Data refresh flow shows the complete refresh process."
    )
    _diagrams(language)
    st.markdown(f"### {headings[6]}")
    _author(language)
    st.warning(
        "Az előrejelzések valószínűségi becslések, nem garantált kimenetek. A pozitív "
        "várható érték (EV) nem jelenti azt, hogy egy adott fogadás nyereséges lesz. "
        "Az eredmények tájékoztató és elemzési célt szolgálnak."
        if language == "HU" else
        "Predictions are probability estimates, not guarantees. Positive EV does not mean "
        "an individual bet will win. The NFL Analytics Platform is independent and is not "
        "affiliated with the NFL, its clubs or any bookmaker. Bet responsibly."
    )
