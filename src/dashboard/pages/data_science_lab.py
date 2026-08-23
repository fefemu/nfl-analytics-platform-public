"""Curated model-governance page for technical readers."""

import pandas as pd
import plotly.express as px
import streamlit as st

from src.dashboard.components import empty_state, status_pill
from src.dashboard.i18n import DEFAULT_LANGUAGE, Language, tr


MODEL_LABELS = {
    "elo": {"HU": "Elo baseline", "EN": "Elo baseline"},
    "logistic_elo_plus_qb": {
        "HU": "Elo + irányító",
        "EN": "Elo + quarterback",
    },
    "logistic_elo_qb_post_bye": {
        "HU": "Elo + irányító + pihenési helyzet",
        "EN": "Elo + quarterback + rest context",
    },
    "logistic_elo_qb_unit_burdens": {
        "HU": "Elo + irányító + sérülési terhelés",
        "EN": "Elo + quarterback + injury burden",
    },
    "logistic_full_core": {
        "HU": "Teljes logisztikus modell",
        "EN": "Full logistic model",
    },
}


def _workflow(language: Language) -> None:
    labels = (
        (
            ("1", "Adatok", "Mérkőzések, play-by-play, nfelo, irányítók, sérülések, időjárás és oddsok."),
            ("2", "Feature-ök", "A kickoff előtt ismert adatokból számított, időrendhelyes modellváltozók."),
            ("3", "Modellek", "Külön modellek a győzelmi esélyre, a pontkülönbségre és az összpontszámra."),
            ("4", "Validáció", "Backtest, holdout teszt és 2026-os forward test."),
            ("5", "Előrejelzések", "Győzelmi valószínűség, Spread, Total és várt csapatpontszámok."),
        ) if language == "HU" else (
            ("1", "Data", "Games, play-by-play, nfelo, quarterbacks, injuries, weather and odds."),
            ("2", "Features", "Chronologically valid model inputs known before kickoff."),
            ("3", "Models", "Separate models for win probability, margin and total points."),
            ("4", "Validation", "Backtest, holdout test and the 2026 forward test."),
            ("5", "Predictions", "Win probability, Spread, Total and implied team scores."),
        )
    )
    columns = st.columns(5)
    for column, (number, title, body) in zip(columns, labels, strict=True):
        with column:
            st.markdown(
                '<div class="nap-card">'
                f'<div class="nap-eyebrow">{number}</div>'
                f'<div class="nap-panel-title">{title}</div>'
                f'<div class="nap-muted">{body}</div></div>',
                unsafe_allow_html=True,
            )


def _prediction_layers(language: Language) -> None:
    """Explain the three production targets in end-user language."""
    if language == "HU":
        st.write(
            "A platform három külön kérdésre három külön modellt használ. Egy meccs "
            "győztesének esélye, várható pontkülönbsége és várható összpontszáma nem "
            "ugyanaz a célváltozó, ezért nem is ugyanabból a képletből készül."
        )
        cards = (
            (
                "Győzelmi valószínűség",
                "Ki nyeri a mérkőzést? A modell mindkét csapathoz 0–100% közötti "
                "győzelmi esélyt rendel.",
            ),
            (
                "Várható pontkülönbség",
                "Mekkora különbség várható a két csapat között? A betting megfelelője "
                "a Spread.",
            ),
            (
                "Várható összpontszám",
                "Hány pont várható összesen a mérkőzésen? A betting megfelelője a Total.",
            ),
        )
    else:
        st.write(
            "The platform uses separate models for win probability, expected margin "
            "and expected combined points because these are different prediction targets."
        )
        cards = (
            ("Win probability", "Who wins the game? Each team receives a probability between 0% and 100%."),
            ("Expected margin", "How large is the expected difference between the teams? Its betting equivalent is the Spread."),
            ("Expected total points", "How many combined points are expected in the game? Its betting equivalent is the Total."),
        )
    columns = st.columns(3)
    for column, (title, body) in zip(columns, cards, strict=True):
        with column:
            st.markdown(f"#### {title}")
            st.write(body)


def _validation_flow(language: Language) -> None:
    st.markdown("### 03 — Hogyan validáljuk a modelleket?" if language == "HU" else "### 03 — How are models validated?")
    steps = (
        (
            ("1. Backtest", "Időrendi teszt korábbi szezonokon: a modell mindig csak a vizsgált szezon előtti adatokból tanul."),
            ("2. Holdout teszt", "Egyszeri ellenőrzés a fejlesztés és modellválasztás során félretett 2025-ös adatokon."),
            ("3. Forward test", "A kickoff előtt rögzített 2026-os előrejelzések későbbi értékelése valóban ismeretlen eredményeken."),
        ) if language == "HU" else (
            ("1. Backtest", "Chronological tests where every model learns only from seasons before the one being evaluated."),
            ("2. Holdout test", "A one-time check on 2025 data kept out of development and candidate selection."),
            ("3. Forward test", "Later evaluation of pre-kickoff 2026 forecasts on genuinely unseen outcomes."),
        )
    )
    columns = st.columns(3)
    for column, (title, body) in zip(columns, steps, strict=True):
        with column:
            st.markdown(f"#### {title}")
            st.write(body)
    with st.expander("Validációs biztosítékok" if language == "HU" else "Validation safeguards"):
        st.markdown(
            "- **Időrendi validáció:** nincs véletlenszerű train-test split.\n"
            "- **Adatszivárgás elleni védelem:** a rolling feature-ök egy meccsel eltolva készülnek.\n"
            "- **Azonos tesztminta:** a jelölteket ugyanazokon a meccseken hasonlítjuk össze.\n"
            "- **Külön adatútvonalak:** a teljes és hiányos adatokra használt modelleket külön értékeljük."
            if language == "HU" else
            "- **Chronological validation:** no random train-test split.\n"
            "- **Leakage protection:** rolling features are shifted by one game.\n"
            "- **Identical samples:** candidates are compared on the same games.\n"
            "- **Separate data paths:** complete-input and missing-input models are evaluated separately."
        )


def _scorecard(scorecard: pd.DataFrame, language: Language) -> None:
    if scorecard.empty:
        return
    columns = [
        column for column in (
            "model_name", "model_version", "game_count", "accuracy",
            "brier_score", "log_loss", "worst_season_brier_score",
            "brier_score_season_std",
        ) if column in scorecard
    ]
    st.markdown(f"### {tr(language, 'model_comparison')}")
    st.caption(
        "Lower is better for Brier score and log loss. Accuracy is contextual, not the selection metric."
        if language == "EN" else
        "A Brier score és a log loss esetében az alacsonyabb érték jobb. "
        "Az accuracy hasznos kiegészítő információ, de önmagában nem elég "
        "egy probability model kiválasztásához."
    )
    st.dataframe(scorecard[columns], hide_index=True, width="stretch")

    if "brier_score" in scorecard and scorecard["brier_score"].notna().any():
        best = scorecard.loc[scorecard["brier_score"].astype(float).idxmin()]
        model = str(best.get("model_name", "–"))
        brier = float(best["brier_score"])
        accuracy = best.get("accuracy")
        conclusion = (
            f"A táblában a legalacsonyabb Brier score-t a **{model}** érte el "
            f"(**{brier:.4f}**). "
            + (
            f"Az ehhez tartozó találati arány **{float(accuracy):.1%}**. "
                if pd.notna(accuracy) else ""
            )
            + "Ez a vizsgált időszak átlagos eredménye; a szezononkénti chart "
            "mutatja meg, mennyire stabil ugyanez az előny."
            if language == "HU" else
            f"The lowest Brier score in the table belongs to **{model}** "
            f"(**{brier:.4f}**). "
            + (
                f"Its corresponding accuracy is **{float(accuracy):.1%}**. "
                if pd.notna(accuracy) else ""
            )
            + "This is the average result across the evaluated period; the season chart "
            "shows whether that advantage is stable over time."
        )
        st.success(conclusion)


def _build_season_chart(seasons: pd.DataFrame):
    """Build the season-level Brier score comparison figure."""

    figure = px.line(
        seasons,
        x="validation_season",
        y="brier_score",
        color="model_name",
        markers=True,
        labels={"validation_season": "Season", "brier_score": "Brier score"},
    )
    return figure


def _season_chart(seasons: pd.DataFrame, language: Language) -> None:
    required = {"validation_season", "model_name", "brier_score"}
    if seasons.empty or not required.issubset(seasons.columns):
        return
    st.markdown(f"### {tr(language, 'season_stability')}")
    figure = _build_season_chart(seasons)
    figure.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 10, "r": 10, "t": 25, "b": 10},
        height=390,
        legend_title_text="Model",
    )
    st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})


def _methodology_guide(language: Language) -> None:
    st.markdown("### Hogyan teszteljük a modelleket?" if language == "HU" else "### How are the models tested?")
    if language == "HU":
        st.markdown(
            "1. **Időrendi validáció:** a modell mindig csak korábbi szezonokból "
            "tanul, majd egy későbbi szezont jelez előre. Nincs véletlenszerű train-test split.\n"
            "2. **Adatszivárgás elleni védelem:** minden rolling feature egy mérkőzéssel el van "
            "tolva, így a modell nem láthatja az előrejelzett mérkőzés eredményét.\n"
            "3. **Jelöltek összehasonlítása:** ugyanazon mérkőzéseken hasonlítjuk össze a modelleket, "
            "nem eltérő vagy kényelmesen kiválasztott mintákon.\n"
            "4. **Holdout:** a 2025-ös szezont a fejlesztés alatt zárva tartottuk, és csak a "
            "végleges jelölt kiválasztása után nyitottuk meg egyszer.\n"
            "5. **Forward test:** a 2026-os előrejelzések valódi jövőbeli becslések. Ezeket "
            "a kezdőrúgás előtt archiválva lehet majd tisztán kiértékelni.\n"
            "6. **Primary és fallback routing:** a teljes és a hiányos aktuális adatokra "
            "használt modelleket külön teszteljük."
        )
    else:
        st.markdown(
            "Models are evaluated with expanding chronological validation: each fold trains "
            "on earlier seasons and predicts a later one. Rolling features are shifted to "
            "prevent leakage, candidates are compared on identical games, and primary and "
            "fallback routing are evaluated separately. The 2025 holdout was opened once "
            "after selection; 2026 is the prospective forward test."
        )


def _model_guide(scorecard: pd.DataFrame, language: Language) -> None:
    """Explain the actual governance candidates and production routing."""

    st.markdown(
        "### Milyen modelleket hasonlítunk össze?"
        if language == "HU" else
        "### Which models are being compared?"
    )
    if language == "HU":
        st.write(
            "A scorecard sorai eltérő információkészletet használnak. Így látható, "
            "hogy egy új változócsoport valóban javít-e az egyszerűbb kiindulási "
            "modellhez képest."
        )
        explanations = {
            "elo": (
                "**Elo – csapaterősségi alapmodell.** A korábbi eredményekből "
                "folyamatosan frissülő csapaterősséget és a hazai pálya előnyét "
                "használja. Jó viszonyítási alap, de külön nem ismeri a kezdő "
                "irányítót vagy a sérült kulcsjátékosokat."
            ),
            "logistic_elo_plus_qb": (
                "**Logistic Elo + QB.** Logistic regression, amely az Elo-különbség "
                "mellé beemeli a várható kezdő irányítók értékelésének különbségét. "
                "Az adatokból tanulja meg, hogy ez a két hatás együtt milyen győzelmi "
                "valószínűséget jelent."
            ),
            "logistic_elo_qb_post_bye": (
                "**Logistic Elo + QB + bye week.** Az előző modellhez hozzáadja, "
                "hogy a csapat pihenőhét után játszik-e. Azt vizsgálja, hogy a "
                "hosszabb felkészülési idő ad-e stabil előnyt."
            ),
            "logistic_elo_qb_unit_burdens": (
                "**Logistic Elo + QB + egységszintű sérülési terhelés.** Külön méri "
                "a támadó-, védő- és special teams egységek hiányzó játékosokból "
                "eredő terhelését. Nemcsak a sérültek száma, hanem a játékos szerepe "
                "és korábbi snap-részesedése is számít."
            ),
            "logistic_full_core": (
                "**Teljes alap-feature készlet.** Elo és QB mellett a legutóbbi négy "
                "mérkőzés támadó- és védőhatékonyságát is használja: EPA/play, "
                "success rate, explosive play, sack és turnover mutatókat. Részletesebb, "
                "de a több változó nagyobb zaj- és túlillesztési kockázatot jelent."
            ),
        }
    else:
        st.write(
            "Each scorecard row uses a different information set, showing whether "
            "added complexity produces stable improvement over a simple baseline."
        )
        explanations = {
            "elo": "**Elo baseline.** Rolling team strength and home-field advantage, without explicit QB or injury input.",
            "logistic_elo_plus_qb": "**Logistic Elo + QB.** Learns probability from Elo difference and expected starting-QB rating difference.",
            "logistic_elo_qb_post_bye": "**Logistic Elo + QB + bye.** Adds whether either team is returning from a bye week.",
            "logistic_elo_qb_unit_burdens": "**Logistic Elo + QB + unit injuries.** Adds role- and snap-weighted offense, defense and special-teams injury burden.",
            "logistic_full_core": "**Full core feature set.** Adds last-four-game EPA, success, explosive-play, sack and turnover indicators.",
        }

    available = (
        set(scorecard["model_name"].astype(str))
        if "model_name" in scorecard else set(explanations)
    )
    for model_name, explanation in explanations.items():
        if model_name in available:
            with st.expander(model_name):
                st.markdown(explanation)

    if language == "HU":
        st.markdown("#### A jelenlegi éles modell")
        st.write(
            "A felső kártyán látható **external_nfelo_probability_routing** a későbbi "
            "külső nfelo-audit eredménye. A primary útvonal 70%-ban egy külső Elo-, "
            "QB- és sérülési változókat használó logistic regression, 30%-ban pedig "
            "az nfelo publikált valószínűsége. Hiányos sérülési adatnál a validált "
            "fallback modell külső Elo- és QB adjustment alapján becsül. A blend célja, "
            "hogy az egyedi modell jelét egy független külső becslés stabilizálja."
        )
    else:
        st.markdown("#### Current production routing")
        st.write(
            "The registered **external_nfelo_probability_routing** follows the later "
            "external-nfelo audit. Its primary route blends a 70% external Elo/QB/injury "
            "logistic model with 30% published nfelo probability. The validated fallback "
            "uses external Elo and QB adjustment when complete injury inputs are unavailable."
        )


def _production_interpretation(language: Language) -> None:
    st.markdown("### Hogyan olvasd az eredményt?" if language == "HU" else "### How to interpret the result")
    st.write(
        (
            "Az éles modell nem attól jó, hogy minden mérkőzés győztesét eltalálja. "
            "A cél az, hogy a 60%-os események hosszú távon nagyjából 60%-ban, a "
            "70%-osak pedig nagyjából 70%-ban következzenek be. Ezt nevezzük calibrationnek. "
            "A betting felhasználásban nem pusztán a várható győztest keressük: azt "
            "vizsgáljuk, hogy a modell által becsült valószínűség és az oddsokból számított, "
            "fogadóirodai margin nélküli piaci valószínűség közötti eltérés elég nagy-e."
        ) if language == "HU" else (
            "A useful production model does not need to predict every winner. It should be "
            "calibrated: events assigned 60% should occur about 60% of the time over a large "
            "sample. Betting analysis therefore compares model probability with no-vig market "
            "probability rather than simply selecting the expected winner."
        )
    )


def _metric_table(language: Language) -> None:
    st.markdown(
        "### 02 — Hogyan mérjük a modell teljesítményét?"
        if language == "HU" else
        "### 02 — How is model performance measured?"
    )
    rows = (
        [
            ["Brier score", "A valószínűségi becslések pontossága", "Győzelmi modell", "Alacsonyabb"],
            ["Log loss", "A magabiztos tévedéseket erősen büntető valószínűségi hiba", "Győzelmi modell", "Alacsonyabb"],
            ["Accuracy", "A helyesen eltalált győztesek aránya", "Kiegészítő mutató", "Magasabb"],
            ["MAE", "Az átlagos abszolút előrejelzési hiba pontban", "Spread / Total", "Alacsonyabb"],
            ["RMSE", "A nagy mellélövéseket erősebben büntető hiba pontban", "Spread / Total", "Alacsonyabb"],
        ] if language == "HU" else [
            ["Brier score", "Accuracy of probability estimates", "Win probability", "Lower"],
            ["Log loss", "Probability error with a strong penalty for confident mistakes", "Win probability", "Lower"],
            ["Accuracy", "Share of correctly selected winners", "Supporting metric", "Higher"],
            ["MAE", "Mean absolute prediction error in points", "Spread / Total", "Lower"],
            ["RMSE", "Point error that penalizes large misses more strongly", "Spread / Total", "Lower"],
        ]
    )
    columns = (
        ["Mérőszám", "Mit mér?", "Mire használjuk?", "Jobb irány"]
        if language == "HU" else
        ["Metric", "What does it measure?", "Used for", "Better direction"]
    )
    st.dataframe(pd.DataFrame(rows, columns=columns), hide_index=True, width="stretch")


def _candidate_progression(scorecard: pd.DataFrame, language: Language) -> None:
    st.markdown(
        "### 04 — Hogyan fejlesztjük a modelleket?"
        if language == "HU" else
        "### 04 — How are the models developed?"
    )
    st.write(
        (
            "A fejlesztés során az egyszerűbb kiindulási modellhez lépésenként adunk új "
            "információkat. Az összetettebb modell csak akkor előrelépés, ha későbbi "
            "szezonokon is stabilan jobb eredményt ad."
        ) if language == "HU" else (
            "Development adds information step by step to a simpler baseline. Extra "
            "complexity is useful only when improvement remains stable in later seasons."
        )
    )
    descriptions = {
        "elo": {
            "HU": "A folyamatosan frissülő csapaterősség és a hazai pálya előnye.",
            "EN": "Continuously updated team strength and home-field advantage.",
        },
        "logistic_elo_plus_qb": {
            "HU": "Az Elo mellé bekerül a várható kezdő irányítók értékelésének különbsége.",
            "EN": "Adds the expected starting-quarterback rating difference to Elo.",
        },
        "logistic_elo_qb_post_bye": {
            "HU": "Az Elo és az irányító mellett azt is jelzi, ha valamelyik csapat bye week után játszik.",
            "EN": "Adds whether either team plays after a bye week.",
        },
        "logistic_elo_qb_unit_burdens": {
            "HU": "Az Elo és az irányító mellé támadó-, védő- és special teams sérülési terhelés kerül.",
            "EN": "Adds offense, defense and special-teams injury burden to Elo and quarterback context.",
        },
        "logistic_full_core": {
            "HU": "Az előző jelek mellett rövid távú EPA, success rate, explosive play, sack és turnover mutatókat is használ.",
            "EN": "Also uses recent EPA, success rate, explosive-play, sack and turnover indicators.",
        },
    }
    available = ["elo"] + (
        scorecard["model_name"].astype(str).tolist()
        if "model_name" in scorecard else list(descriptions)[1:]
    )
    for index, model_id in enumerate(dict.fromkeys(available), start=1):
        if model_id not in descriptions:
            continue
        label = MODEL_LABELS.get(model_id, {language: model_id})[language]
        st.markdown(f"**{index}. {label}** — {descriptions[model_id][language]}")


def _results_section(
    scorecard: pd.DataFrame,
    seasons: pd.DataFrame,
    language: Language,
) -> None:
    st.markdown(
        "### 05 — Mit mutatnak az eredmények?"
        if language == "HU" else
        "### 05 — What do the results show?"
    )
    st.write(
        (
            "A győzelmi modelleknél a Brier score az elsődleges összehasonlítási mutató: "
            "az alacsonyabb érték pontosabb valószínűségi becslést jelent. Az Accuracy "
            "hasznos kiegészítés, de önmagában nem választ modellt."
        ) if language == "HU" else (
            "Brier score is the primary comparison metric for win models: lower values "
            "mean better probability estimates. Accuracy is context, not the selection rule."
        )
    )
    if not scorecard.empty:
        table = scorecard.copy()
        if "model_name" in table:
            table["model_name"] = table["model_name"].map(
                lambda value: MODEL_LABELS.get(str(value), {language: str(value)})[language]
            )
        rename = {
            "model_name": "Modell" if language == "HU" else "Model",
            "model_version": "Verzió" if language == "HU" else "Version",
            "game_count": "Vizsgált meccsek" if language == "HU" else "Games",
            "accuracy": "Accuracy",
            "brier_score": "Brier score",
            "log_loss": "Log loss",
            "worst_season_brier_score": "Legrosszabb szezon Brier score" if language == "HU" else "Worst-season Brier score",
            "brier_score_season_std": "Szezonok közötti szórás" if language == "HU" else "Season variability",
        }
        visible = [column for column in rename if column in table.columns]
        table = table[visible].rename(columns=rename)
        brier_column = "Brier score"
        if brier_column in table and table[brier_column].notna().any():
            best_index = table[brier_column].astype(float).idxmin()
            model_column = "Modell" if language == "HU" else "Model"
            table.loc[best_index, model_column] = "★ " + str(table.loc[best_index, model_column])
        st.dataframe(table, hide_index=True, width="stretch")
    if not seasons.empty:
        st.markdown("#### Szezononkénti stabilitás" if language == "HU" else "#### Stability by season")
        st.caption(
            "Nemcsak az átlag számít: egy jó modell előnye lehetőleg több szezonban is megmarad, és nem egyetlen kiugró évből származik."
            if language == "HU" else
            "Average performance is not enough: a useful model should remain reasonably stable across seasons rather than rely on one exceptional year."
        )
        _season_chart(seasons.assign(model_name=seasons["model_name"].map(
            lambda value: MODEL_LABELS.get(str(value), {language: str(value)})[language]
        )), language)


def _production_model_card(registry: pd.DataFrame, language: Language) -> None:
    if registry.empty:
        return
    row = registry.iloc[0]
    logistic_weight = float(row.get("logistic_weight", 0.7))
    elo_weight = float(row.get("elo_weight", 0.3))
    st.markdown(
        "### 06 — Melyik modell fut élesben?"
        if language == "HU" else
        "### 06 — Which model is in production?"
    )
    title = "Logistic + nfelo ensemble" if language == "HU" else "Logistic + nfelo ensemble"
    status = "2026 forward test"
    st.markdown(
        '<div class="nap-card">'
        f'<div class="nap-eyebrow">{"Aktuális győzelmi modell" if language == "HU" else "Current win model"}</div>'
        f'<div class="nap-metric-value blue">{title}</div>'
        f'<div class="nap-muted">{logistic_weight:.0%} Logistic Regression · {elo_weight:.0%} publikált nfelo</div>'
        f'<div class="nap-divider"></div>{status_pill(status.upper(), True)}'
        f'<div class="nap-muted">{"Verzió" if language == "HU" else "Version"} {row.get("model_version", "–")}</div></div>',
        unsafe_allow_html=True,
    )
    st.write(
        (
            "Az ensemble a saját, csapaterősséget, irányítóhelyzetet és egységszintű "
            "sérülési terhelést használó logistic modellt egy független nfelo-becsléssel "
            "kombinálja. A blend célja a saját modell jelének stabilizálása."
        ) if language == "HU" else (
            "The ensemble combines the platform's team-strength, quarterback and injury "
            "logistic model with an independent published nfelo estimate to improve stability."
        )
    )
    first, second = st.columns(2)
    with first:
        st.markdown("#### Elsődleges modell" if language == "HU" else "#### Primary model")
        st.write(
            "Külső nfelo csapaterősség, a saját és nfelo QB adjustment, valamint támadó-, védő- és special teams sérülési terhelés."
            if language == "HU" else
            "External nfelo team strength, platform and nfelo QB adjustment, plus offense, defense and special-teams injury burden."
        )
    with second:
        st.markdown("#### Hiányos adatok esetén" if language == "HU" else "#### When inputs are incomplete")
        st.write(
            "A külön validált tartalék modell külső nfelo csapaterősséget és nfelo QB adjustmentet használ."
            if language == "HU" else
            "A separately validated fallback model uses external nfelo team strength and nfelo QB adjustment."
        )


IMPACT_LABELS = {
    "home_field": {"all_games": {"EN": "All games", "HU": "Minden meccs"}},
    "rest": {
        "home_3_plus": {"EN": "Home: 3+ extra rest days", "HU": "Hazai csapat: legalább 3 nappal több pihenés"},
        "away_3_plus": {"EN": "Away: 3+ extra rest days", "HU": "Vendégcsapat: legalább 3 nappal több pihenés"},
        "similar": {"EN": "Rest difference within 2 days", "HU": "Legfeljebb 2 nap pihenéskülönbség"},
        "home_post_bye": {"EN": "Home team only after a bye", "HU": "Csak a hazai csapat érkezett bye weekről"},
    },
    "qb": {
        "home_advantage": {"EN": "Meaningful home-QB advantage", "HU": "Jelentős hazai QB-előny"},
        "away_advantage": {"EN": "Meaningful away-QB advantage", "HU": "Jelentős vendég QB-előny"},
        "similar": {"EN": "Similar QB ratings", "HU": "Hasonló QB-rating"},
    },
    "injury": {
        "home_more_injured": {"EN": "Home team more injured", "HU": "A hazai csapat sérültebb"},
        "away_more_injured": {"EN": "Away team more injured", "HU": "A vendégcsapat sérültebb"},
        "similar": {"EN": "Similar injury burden", "HU": "Hasonló sérülési terhelés"},
    },
    "weather": {
        "indoor": {"EN": "Indoor", "HU": "Fedett stadion"},
        "freezing": {"EN": "Freezing", "HU": "Fagypont körüli vagy hidegebb"},
        "high_wind": {"EN": "High wind", "HU": "Erős szél"},
        "other_exposed": {"EN": "Other outdoor", "HU": "Egyéb szabadtéri"},
    },
}


def _impact_chart(data: pd.DataFrame, topic: str, language: Language):
    subset = data.loc[data["topic"] == topic].copy()
    subset["label"] = subset["segment"].map(
        lambda value: IMPACT_LABELS[topic][value][language]
    )
    value = "average_total" if topic == "weather" else "win_rate"
    subset["display_value"] = subset[value] * (100 if value == "win_rate" else 1)
    axis_title = (
        ("Hazai győzelmi arány" if language == "HU" else "Home win rate")
        if value == "win_rate" else
        ("Átlagos összpontszám" if language == "HU" else "Average total points")
    )
    figure = px.bar(
        subset, x="display_value", y="label", orientation="h",
        text="display_value", custom_data=["game_count"],
        labels={"display_value": axis_title, "label": ""},
    )
    suffix = "%" if value == "win_rate" else (" pont" if language == "HU" else " points")
    games_label = "Meccsek" if language == "HU" else "Games"
    figure.update_traces(
        marker_color="#42a5ff", texttemplate="%{text:.1f}" + suffix,
        hovertemplate="%{y}<br>%{x:.1f}" + suffix + f"<br>{games_label}: %{{customdata[0]}}<extra></extra>",
    )
    figure.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", height=300,
        margin={"l": 10, "r": 20, "t": 15, "b": 10}, showlegend=False,
    )
    return figure


def _segment(data: pd.DataFrame, topic: str, segment: str) -> pd.Series:
    return data.loc[
        (data["topic"] == topic) & (data["segment"] == segment)
    ].iloc[0]


def _sample_table(rows: pd.DataFrame, topic: str, language: Language) -> None:
    table = rows.copy()
    table["Csoport" if language == "HU" else "Group"] = table["segment"].map(
        lambda value: IMPACT_LABELS[topic][value][language]
    )
    sample_label = "Mintaszám" if language == "HU" else "Games"
    table[sample_label] = table["game_count"].astype(int)
    if topic == "weather":
        metric_label = "Átlagos összpontszám" if language == "HU" else "Average total"
        table[metric_label] = table["average_total"].round(1)
        columns = ["Csoport" if language == "HU" else "Group", sample_label, metric_label]
    else:
        win_label = "Hazai győzelmi arány" if language == "HU" else "Home win rate"
        margin_label = "Átlagos hazai pontkülönbség" if language == "HU" else "Average home margin"
        table[win_label] = table["win_rate"].map(lambda value: f"{value:.1%}")
        table[margin_label] = table["average_margin"].map(lambda value: f"{value:+.2f}")
        columns = ["Csoport" if language == "HU" else "Group", sample_label, win_label, margin_label]
    st.dataframe(table[columns], hide_index=True, width="stretch")


def _home_field_panel(data: pd.DataFrame, language: Language) -> None:
    row = _segment(data, "home_field", "all_games")
    if language == "HU":
        st.markdown("#### A kérdés")
        st.write(
            "Van-e általános előnye annak a csapatnak, amelyik hazai pályán játszik? "
            "Itt nem két külön csoportot hasonlítunk össze: minden NFL-meccsnek van hazai "
            "csapata, ezért a teljes minta hazai eredményét foglaljuk össze."
        )
    else:
        st.markdown("#### The question")
        st.write(
            "Do teams have a general advantage when playing at home? This is a league-wide "
            "summary rather than a two-group comparison because every game has a home team."
        )
    columns = st.columns(3)
    columns[0].metric(
        "Hazai győzelmi arány" if language == "HU" else "Home win rate",
        f"{row['win_rate']:.1%}",
        help=("A döntetlenek nem számítanak hazai győzelemnek." if language == "HU" else "Ties are not counted as home wins."),
    )
    columns[1].metric(
        "Átlagos hazai pontkülönbség" if language == "HU" else "Average home margin",
        f"{row['average_margin']:+.2f}",
        help=("Hazai pontok mínusz vendégpontok." if language == "HU" else "Home points minus away points."),
    )
    columns[2].metric("Vizsgált meccsek" if language == "HU" else "Games", f"{int(row['game_count']):,}")
    st.success(
        (
            "**Mit jelent ez?** A vizsgált időszakban a hazai csapat valamivel gyakrabban "
            "nyert, mint 50%, és meccsenként átlagosan 1,72 ponttal szerzett többet. "
            "Ez ligaátlag: nem jelenti azt, hogy minden csapat ugyanekkora hazai előnyt élvez."
        ) if language == "HU" else (
            "**Interpretation:** home teams won somewhat more often than 50% and scored "
            "1.72 more points per game on average. This is a league average, not a fixed advantage for every team."
        )
    )


def _comparison_panel(data: pd.DataFrame, topic: str, language: Language) -> None:
    introductions = {
        "rest": {
            "HU": ("**Kérdés:** jobban teljesít-e a hazai csapat, ha több ideje volt pihenni? "
                   "A +3 nap azt jelenti, hogy a hazai csapat legalább három nappal többet pihent az ellenfelénél. "
                   "A bye week csoportban csak a hazai csapat érkezett pihenőhétről."),
            "EN": "**Question:** does the home team perform better with more rest? +3 days means at least three additional rest days versus its opponent.",
        },
        "qb": {
            "HU": ("**Kérdés:** hogyan változik az eredmény, ha az egyik várható kezdő QB ratingje lényegesen jobb? "
                   "A rating nem egyszerű rangsor: a korábbi passzjátékból becsült, folyamatos értékelés. "
                   "A ±0,75-ig terjedő különbséget hasonló QB-helyzetnek tekintjük."),
            "EN": "**Question:** how do results change when one expected starting QB has a meaningfully higher historical rating? Differences within ±0.75 are treated as similar.",
        },
        "injury": {
            "HU": ("**Kérdés:** kapcsolatban áll-e a nagyobb sérülési terhelés a gyengébb eredménnyel? "
                   "A mutató nemcsak a hiányzók számát, hanem a játékos szerepét és korábbi snap-részesedését is figyelembe veszi. "
                   "A +2 legalább két burden-ponttal nagyobb terhelést jelent."),
            "EN": "**Question:** is greater role- and snap-weighted injury burden associated with poorer results? A difference of 2+ defines the more-injured group.",
        },
        "weather": {
            "HU": ("**Kérdés:** milyen környezetben születik több vagy kevesebb pont? "
                   "Itt nem a hazai győzelmi arányt, hanem a két csapat együttes pontszámát hasonlítjuk össze. "
                   "Egy meccs egyszerre lehet például szabadtéri és erős szeles; a kategóriák ezért nem mindenhol kizárólagosak."),
            "EN": "**Question:** which environments are associated with higher or lower combined scoring? Categories can overlap, such as outdoor and high wind.",
        },
    }
    st.markdown(introductions[topic][language])
    rows = data.loc[data["topic"] == topic]
    st.plotly_chart(
        _impact_chart(data, topic, language), width="stretch",
        config={"displayModeBar": False},
    )
    _sample_table(rows, topic, language)

    if language == "HU":
        conclusions = {
            "rest": "**Mit látunk?** A több pihenéssel rendelkező hazai csapatok nyers eredménye jobb, a csak hazai post-bye csoport pedig még erősebb. Ez nem bizonyítja, hogy kizárólag a pihenés okozta a különbséget.",
            "qb": "**Mit látunk?** A QB-rating előnye erősen együtt mozog a győzelmi aránnyal és a pontkülönbséggel. Ezért a QB-információ a production model egyik fontos bemenete.",
            "injury": "**Mit látunk?** Amelyik oldal nagyobb sérülési terheléssel érkezik, annak nyers eredménye rosszabb. A csapaterősség és a QB-helyzet egy részt megmagyarázhat ebből, ezért a modell ezeket együtt kezeli.",
            "weather": "**Mit látunk?** Fedett stadionban született a legtöbb pont, erős szélben a legkevesebb. A nyers különbség alapján az időjárás elsősorban a Totals model számára lehet hasznos.",
        }
    else:
        conclusions = {
            "rest": "**What we see:** home teams with additional rest have stronger raw results, especially when only they return from a bye. This does not prove rest alone caused the difference.",
            "qb": "**What we see:** QB-rating advantage is strongly associated with win rate and scoring margin, supporting its use as a production-model input.",
            "injury": "**What we see:** the side carrying greater injury burden has poorer raw results. Team strength and QB context may explain part of the gap.",
            "weather": "**What we see:** indoor games score highest and high-wind games lowest, making weather especially relevant to the Totals model.",
        }
    st.success(conclusions[topic])
    st.caption(
        "Korlát: ez kontrollváltozók nélküli, leíró összehasonlítás. Nem fogadási szabály és nem önálló prediction."
        if language == "HU" else
        "Limitation: this is an uncontrolled descriptive comparison, not a betting rule or standalone prediction."
    )


def _impact_analysis(data: pd.DataFrame, language: Language) -> None:
    if data.empty:
        return
    st.markdown("### 07 — Hatásvizsgálatok" if language == "HU" else "### 07 — Impact analysis")
    st.info(
        "A predikció mellett a történelmi adatokat külön elemzésekhez is használjuk. "
        "Ezek azt vizsgálják, hogy bizonyos körülmények között hogyan változott a "
        "csapatok teljesítménye. Az eredmények összefüggéseket mutatnak, nem "
        "feltétlenül ok-okozati kapcsolatot."
        if language == "HU" else
        "Historical data is also used for separate research analyses. These examine how "
        "team performance changed under particular circumstances and show associations, "
        "not necessarily causal effects."
    )
    tabs = st.tabs(
        ["Hazai pálya", "Pihenés", "Irányító (QB)", "Sérülések", "Időjárás"]
        if language == "HU" else
        ["Home field", "Rest", "QB", "Injuries", "Weather"]
    )
    with tabs[0]:
        _home_field_panel(data, language)
    for tab, topic in zip(tabs[1:], ("rest", "qb", "injury", "weather"), strict=True):
        with tab:
            _comparison_panel(data, topic, language)


def render_data_science_lab(
    products: dict[str, pd.DataFrame],
    language: Language = DEFAULT_LANGUAGE,
) -> None:
    """Render production governance without exposing raw database tables."""

    if not products:
        empty_state(
            "Data Science Lab is not available" if language == "EN" else "A Data Science Lab nem elérhető",
            "Build the reporting pipeline to populate governance evidence."
            if language == "EN" else
            "Futtasd a reporting pipeline-t a governance eredmények felépítéséhez.",
        )
        return
    st.caption(
        "Ez az oldal bemutatja a platform mögötti modellezési módszereket, a modellek "
        "teljesítményének mérését és azokat az elemzéseket, amelyekkel az egyes tényezők "
        "mérkőzésekre gyakorolt hatását vizsgáljuk."
        if language == "HU" else
        "This page explains the modeling methods behind the platform, how model quality "
        "is measured, and the analyses used to study factors associated with NFL results."
    )
    registry = products.get("production_model_registry", pd.DataFrame())
    scorecard = products.get("model_governance_scorecard", pd.DataFrame())
    seasons = products.get("model_governance_season_results", pd.DataFrame())
    impacts = products.get("historical_impact_summary", pd.DataFrame())
    _workflow(language)
    st.markdown("### 01 — Mit jeleznek előre a modellek?" if language == "HU" else "### 01 — What do the models predict?")
    _prediction_layers(language)
    _metric_table(language)
    _validation_flow(language)
    _candidate_progression(scorecard, language)
    _results_section(scorecard, seasons, language)
    _production_model_card(registry, language)
    _impact_analysis(impacts, language)
    st.info(
        "Chronological validation trains only on earlier seasons. The 2025 holdout is a historical audit; 2026 is the prospective forward test."
        if language == "EN" else
        "A charton látható fejlesztési eredmények nem jelentik azt, hogy a 2026-os "
        "teljesítmény garantált. A 2026-os szezon a prospective forward test: csak a "
        "kickoff előtt rögzített predictionök számítanak majd az értékelésbe."
    )
