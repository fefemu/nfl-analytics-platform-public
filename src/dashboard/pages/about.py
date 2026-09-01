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
        "Fejlesztés és automatizált minőségbiztosítás": "Development and automated quality assurance",
        "pytest · Git · GitHub Actions": "pytest · Git · GitHub Actions",
    },
    "platform_data_model.svg": {
        "DuckDB adatmodell": "DuckDB data model",
        "mind a 63 aktuális fizikai tábla": "all 63 current physical tables",
        "1. RAW — Forráshű adatok": "1. RAW — Source-faithful data",
        "Mérkőzés és játékos": "Games and players",
        "Depth chart és sérülés": "Depth charts and injuries",
        "Odds snapshotok": "Odds snapshots",
        "2. PROCESSED — Tisztított és egységesített adatok": "2. PROCESSED — Cleaned and standardized data",
        "Mérkőzés és teljesítmény": "Games and performance",
        "Játékos és depth chart": "Players and depth charts",
        "Külső rating és odds": "External ratings and odds",
        "3. ANALYTICS — Feature-ök és modellezési adatok": "3. ANALYTICS — Features and modeling data",
        "Feature-ök": "Features",
        "Modellezési adatok és governance": "Modeling data and governance",
        "Kapcsolás: game_id": "Join key: game_id",
        "Játékoskapcsolás: gsis_id": "Player join: gsis_id",
        "Csapatkapcsolás: team": "Team join: team",
        "Historikus betting audit": "Historical betting audit",
        "Meccs-, market- és időpontkulcsok": "Game, market and timestamp keys",
        "a historikus kiértékeléshez.": "for historical evaluation.",
        "4. OUTPUT — Publikálható alkalmazási eredmények": "4. OUTPUT — Publishable application results",
        "Aktuális előrejelzések": "Current predictions",
        "Betting és value": "Betting and value",
        "Szezon-szimuláció": "Season simulation",
        "Jelmagyarázat:": "Legend:",
        "adattranszformáció": "data transformation",
        "logikai kapcsolat (nem deklarált FK)": "logical relationship (not a declared FK)",
    },
    "platform_data_flow.svg": {
        "Adatfrissítési folyamat": "Data refresh workflow",
        "Az automatikus vagy manuális indítástól a validált publikálásig": "From scheduled or manual start to validated publication",
        "ADAT ÉS MODELLEZÉS": "DATA AND MODELING",
        "1. Futás indítása": "1. Start run",
        "GitHub Actions": "GitHub Actions",
        "Ütemezetten vagy manuálisan": "Scheduled or manual",
        "2. Környezet és audit": "2. Environment and audit",
        "Run ID": "Run ID",
        "státusz: RUNNING": "status: RUNNING",
        "3. Adatforrások frissítése": "3. Refresh data sources",
        "nflverse · nfelo · depth chart": "nflverse · nfelo · depth charts",
        "injuries · snap count · időjárás": "injuries · snap counts · weather",
        "4. Előkészítés és modellezés": "4. Preparation and modeling",
        "tisztítás → feature-ök → predictionök": "cleaning → features → predictions",
        "→ szezon-szimuláció": "→ season simulation",
        "Modellezés": "Modeling",
        "sikeres?": "successful?",
        "PIAC ÉS PUBLIKÁLÁS": "MARKET AND PUBLISHING",
        "5. Oddsok frissítése": "5. Refresh odds",
        "The Odds API → snapshot": "The Odds API → snapshot",
        "→ processed odds": "→ processed odds",
        "6. Piaci számítások": "6. Market calculations",
        "no-vig probability → Edge → EV": "no-vig probability → Edge → EV",
        "→ Betting Board": "→ Betting Board",
        "Ellenőrzések": "Checks",
        "Kickoff előtti előrejelzések rögzítése": "Archive pre-kickoff predictions",
        "későbbi kiértékeléshez": "for later evaluation",
        "7. Forward archive": "7. Forward archive",
        "8. Publikálás": "8. Publication",
        "Validált output táblák": "Validated output tables",
        "Az előző release atomikus cseréje": "Atomically replace the previous release",
        "KISZOLGÁLÁS": "SERVING",
        "9. Streamlit": "9. Streamlit",
        "Az alkalmazás az új adatállapotot olvassa": "The app reads the new data state",
        "10. Adatok frissítve": "10. Data refreshed",
        "Az utolsó sikeres publikálás időpontja": "Timestamp of the latest successful publication",
        "Futás sikertelen": "Run failed",
        "FAILED státusz · hibalóg": "FAILED status · error log",
        "Az előző publikált adatállapot változatlan marad": "The previously published data state remains unchanged",
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


def _quality_assurance(language: Language) -> None:
    st.markdown(
        "#### Automatizált tesztelés és minőségbiztosítás"
        if language == "HU" else
        "#### Automated testing and quality assurance"
    )
    st.write(
        (
            "A platform működését több mint 1200 automatizált teszt ellenőrzi. "
            "A tesztcsomag lefedi az adatfeldolgozást, a feature engineeringet, a modell- "
            "és előrejelzési logikát, a piaci számításokat, valamint a kritikus "
            "pipeline-folyamatokat."
        ) if language == "HU" else (
            "More than 1,200 automated tests cover data processing, feature engineering, "
            "model and prediction logic, market calculations, and critical pipeline flows."
        )
    )
    cards = (
        (
            ("Kódminőség", "1200+ pytest teszt ellenőrzi, hogy a komponensek és folyamatok a várt módon működnek."),
            ("Adatminőség", "Séma-, teljességi és konzisztencia-ellenőrzések védik az aktuális production adatállapotot."),
            ("Modellminőség", "Időrendi backtest, elkülönített holdout és kickoff előtti forward test méri a modelleket."),
        ) if language == "HU" else (
            ("Code quality", "1,200+ pytest tests verify that components and workflows behave as expected."),
            ("Data quality", "Schema, completeness and consistency checks protect the current production data state."),
            ("Model quality", "Chronological backtests, a separate holdout and pre-kickoff forward tests evaluate the models."),
        )
    )
    columns = st.columns(3)
    for column, (title, body) in zip(columns, cards, strict=True):
        with column:
            st.markdown(
                '<div class="nap-card">'
                f'<div class="nap-panel-title">{title}</div>'
                f'<div class="nap-muted">{body}</div></div>',
                unsafe_allow_html=True,
            )
    st.caption(
        "A teljes pytest csomag minden ütemezett vagy manuális production refresh során, a publikálás előtt automatikusan lefut."
        if language == "HU" else
        "The full pytest suite runs automatically before publication in every scheduled or manually started production refresh."
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


def _automatic_refresh(language: Language) -> None:
    if language == "HU":
        st.write(
            "A platform adatai és előrejelzései automatikusan frissülnek a GitHub "
            "Actions segítségével. A teljes folyamat hetente három alkalommal fut. "
            "Frissíti a rendelkezésre álló NFL-adatokat, újraszámolja a változókat "
            "és előrejelzéseket, lekéri az aktuális oddsokat, majd újragenerálja a "
            "piaci összehasonlításokat és a szezon-szimulációt."
        )
        st.write(
            "A frissítési folyamat során automatizált adatminőségi és "
            "konzisztencia-ellenőrzések futnak; kritikus hiba esetén a publikálás "
            "megszakad, és a platform továbbra is az utolsó sikeresen validált "
            "adatállapotot használja."
        )
        schedule = (("Kedd", "08:00"), ("Csütörtök", "15:00"), ("Vasárnap", "09:00"))
        footer = (
            "Az időzített futások mellett a frissítés manuálisan is elindítható "
            "GitHub Actionsből. A jobb felső sarokban látható „Adatok frissítve” "
            "időpont mindig az utolsó sikeresen publikált adatállapotot jelzi. "
            "A feltüntetett időpontok célidők; a GitHub Actions tényleges indulása "
            "a szolgáltatás terhelésétől függően késhet."
        )
        zone = "célidő · Budapest"
    else:
        st.write(
            "Data and forecasts are refreshed automatically with GitHub Actions three "
            "times per week. Each run updates available NFL sources, rebuilds features "
            "and forecasts, downloads current odds, and regenerates market comparisons "
            "and the season simulation."
        )
        st.write(
            "Automated data-quality and consistency checks run during every refresh. A "
            "critical failure stops publication, while the platform continues serving the "
            "latest successfully validated data state."
        )
        schedule = (("Tuesday", "08:00"), ("Thursday", "15:00"), ("Sunday", "09:00"))
        footer = (
            "The full refresh can also be started manually from GitHub Actions. The "
            "\"Data updated\" timestamp in the top-right corner always identifies the "
            "latest successfully published data state. Times shown are targets; the "
            "actual GitHub Actions start can be delayed by service load."
        )
        zone = "target · Budapest"
    columns = st.columns(3)
    for column, (day, time) in zip(columns, schedule, strict=True):
        with column:
            st.markdown(
                '<div class="nap-card" style="text-align:center">'
                f'<div class="nap-eyebrow">{day}</div>'
                f'<div class="nap-metric-value blue">{time}</div>'
                f'<div class="nap-muted">{zone}</div></div>',
                unsafe_allow_html=True,
            )
    st.caption(footer)
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
        st.markdown(
            "#### Fizikai adatmodell és adatrétegek"
            if language == "HU" else
            "#### Physical data model and data layers"
        )
        st.write(
            "Az ábra a DuckDB mind a 63 aktuális fizikai tábláját mutatja, "
            "funkcionális csoportokba és adatrétegekbe rendezve. A célja a fizikai "
            "adatstruktúra és az adatáramlás áttekintése; az oszlopszintű séma, "
            "a kulcsok és a kapcsolatok részletes definíciói a technikai dokumentációban találhatók."
            if language == "HU" else
            "The diagram shows all 63 current physical DuckDB tables, organized into "
            "functional groups and data layers. It provides an overview of physical data "
            "structure and flow; column-level schemas, keys and detailed relationships are documented separately."
        )
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


def _source_and_data_management(language: Language) -> None:
    st.markdown(
        "#### Forráskód és adatkezelés"
        if language == "HU" else
        "#### Source code and data handling"
    )
    st.write(
        (
            "A projekt külön privát fejlesztési és publikus alkalmazás-repositoryt "
            "használ. A publikus repository a bemutatható alkalmazáskódot, dokumentációt "
            "és reprodukálható komponenseket tartalmazza. A hozzáférési kulcsok, "
            "lokális adatbázisok és a külső források feltételei miatt nem "
            "továbbterjeszthető adatok nem kerülnek nyilvánosan közzétételre."
        ) if language == "HU" else (
            "The project uses separate private development and public application repositories. "
            "The public repository contains presentable application code, documentation and "
            "reproducible components. Access keys, local databases and source-derived data that "
            "cannot be redistributed under their usage terms are not published publicly."
        )
    )
    st.write(
        (
            "Az API-kulcsok és hozzáférési tokenek környezeti változókon, illetve "
            "GitHub Secretsen keresztül kezeltek. A publikus alkalmazás számára szükséges, "
            "szűkített adatállapot külön, hozzáférés-védett deployment artifactként készül el."
        ) if language == "HU" else (
            "API keys and access tokens are managed through environment variables and GitHub "
            "Secrets. The reduced data state required by the public application is built as a "
            "separate access-controlled deployment artifact."
        )
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
            "Hogyan ellenőrizzük a modelleket?", "Automatikus frissítés", "Technikai háttér",
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
            "Automatic refresh", "Technical background", "About the project and me",
        )
    st.markdown(f"### {headings[0]}")
    _feature_cards(language)
    st.markdown(f"### {headings[1]}")
    _data_flow(language)
    _quality_assurance(language)
    st.markdown(f"### {headings[2]}")
    _sources(language)
    st.markdown(f"### {headings[3]}")
    _model_inputs(language)
    st.markdown(f"### {headings[4]}")
    _validation(language)
    st.markdown(f"### {headings[5]}")
    _automatic_refresh(language)
    st.markdown(f"### {headings[6]}")
    st.write(
        "A platform három nézetben mutatja be a technikai megvalósítást: az Architektúra "
        "a használt technológiákat, az Adatmodell a DuckDB adatrétegeit és tábláit, az "
        "Adatfrissítési folyamat pedig a teljes frissítés menetét mutatja be."
        if language == "HU" else
        "Three views explain the implementation: Architecture shows the technologies, "
        "Data model shows DuckDB layers and tables, and Data refresh flow shows the complete refresh process."
    )
    _diagrams(language)
    _source_and_data_management(language)
    st.markdown(f"### {headings[7]}")
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
