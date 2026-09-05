# NFL Analytics Platform

> **Source-available portfolio project.** Copyright (c) 2026 Ferenc Kaizer.
> All rights reserved; this repository is not currently distributed under an
> open-source license. See [COPYRIGHT.md](COPYRIGHT.md) and
> [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

An end-to-end NFL data engineering, predictive modeling, season simulation and
betting-market analytics platform with a bilingual public application.

## Live Application

**[Open the NFL Analytics Platform](https://nfl-analytics-platform.streamlit.app/)**

Languages: English / Hungarian

The application provides:

- weekly matchup forecasts and model-implied scores;
- matchup-level Game Center analysis;
- Moneyline, Spread and Total market comparison;
- a forward-only Betting Board;
- team and roster views;
- a 10,000-run Monte Carlo season simulator;
- a Data Science Lab covering validation, model selection, ensemble routing and
  fallback behavior.

## What This Project Demonstrates

- End-to-end data pipelines from external sources to analytics-ready datasets.
- Layered DuckDB architecture: **RAW → PROCESSED → ANALYTICS → OUTPUT**.
- Leakage-safe feature engineering and chronological model validation.
- Production models for win probability, expected margin and total points.
- Explicit, validated fallback models for incomplete current-season data.
- Betting-market comparison using no-vig consensus probabilities, model–market probability gaps and expected value.
- Automated testing, data-quality gates and scheduled GitHub Actions refreshes.
- An interactive, responsive and bilingual Streamlit application.

## Current Status

| Item | Current state |
|---|---|
| Version | 0.2.0 |
| Production season | 2026 forward test |
| Historical training and evaluation | 2018–2025 |
| Historical games | 2,227 |
| Current schedule | 272 regular-season games |
| Automated tests | **1,290+ passing** |
| Modeling pipeline | 33 validated steps |
| Season simulation | 10,000 dynamic Monte Carlo runs |
| Production refresh | GitHub Actions, three scheduled runs per week |
| Publication policy | Only validated snapshots are published |

Probability, Spread and Totals models are frozen for the 2026 forward test.
Missing current inputs use separately validated fallback routes instead of
silent or artificial zero-value imputation.

## Architecture

The platform separates source ingestion, transformation, modeling and public
serving responsibilities:

~~~text
External sources
      ↓
RAW — source-aligned snapshots
      ↓
PROCESSED — cleaned and normalized entities
      ↓
ANALYTICS — features, models, predictions and simulations
      ↓
Validation gates and automated regression tests
      ↓
Private, read-only deployment snapshot
      ↓
Bilingual Streamlit application
~~~

The full physical data model currently contains 63 DuckDB tables. The public
application reads a compact deployment database containing only the validated
tables required for serving.

### Technology Stack

- **Data and processing:** Python, pandas, NumPy, DuckDB, SQL
- **Modeling:** scikit-learn, XGBoost, logistic regression, Ridge, Elo
- **Application:** Streamlit, HTML/CSS
- **Testing and automation:** pytest, Git, GitHub Actions
- **Analytics:** privacy-conscious Google Analytics 4 integration

## Models and Validation

The platform predicts three related targets:

- game win probability;
- expected home margin;
- expected total points.

Candidate models are evaluated using chronological backtests, a locked 2025
holdout and the untouched 2026 forward test. Probability evaluation emphasizes
Brier score, log loss and calibration in addition to accuracy. Spread and
Totals candidates use time-aware out-of-sample error metrics.

The production probability forecast combines a selected in-house logistic
component with the public nflElo reference. If required current inputs are
unavailable or incomplete, routing switches to a model validated for the
available feature set. Spread and Totals follow the same explicit
primary/fallback principle.

The repository also documents rejected candidates. A model is not promoted
because it wins on one development split; it must remain stable on unseen time
periods and satisfy the production quality gates.

## Quality and Automation

The suite contains **more than 1,290 automated tests** covering:

- ingestion and normalization;
- DuckDB transactions and transformations;
- leakage-safe feature engineering;
- model and prediction pipelines;
- fallback and routing behavior;
- betting-market calculations;
- season simulation;
- dashboard repositories, view models and application logic;
- deployment snapshot construction and publication.

Three different quality layers are kept separate:

| Quality layer | Purpose |
|---|---|
| Code quality | Automated unit, integration and regression tests |
| Data quality | Schema, completeness, consistency and publication gates |
| Model quality | Backtest, holdout, calibration and forward-test monitoring |

Production refreshes run on GitHub Actions every Tuesday at 08:00, Thursday at
15:00 and Sunday at 15:00 Budapest time. A refresh restores the operational
database, updates sources and odds, rebuilds predictions and simulations, runs
the complete regression suite, validates the output and publishes a new
snapshot. If a critical step fails, the previous validated public snapshot
remains unchanged.

The workflow can also be started manually for exceptional injury, roster or
market changes.

## Public Repository and Data Access

This repository contains publicly shareable source code, tests and
documentation. It intentionally excludes:

- API credentials and tokens;
- generated local databases;
- private operational state;
- third-party-derived datasets whose terms do not permit redistribution.

The hosted application downloads an access-controlled deployment artifact from
a separate private repository using a read-only fine-grained token. Credentials
are supplied through environment variables and GitHub or Streamlit secrets and
are never committed.

The compact dashboard database must not be uploaded to a public GitHub Release
because it may include bookmaker-derived data.

## Data Sources

The project uses documented external sources including:

- nflverse schedules, play-by-play, player, roster and team identity data;
- nfelo / nfelopoints reference ratings;
- depth-chart, injury, snap-count, venue and weather inputs;
- The Odds API for current bookmaker prices.

Source attribution, licensing and redistribution boundaries are documented in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Run Locally

Create and activate a Python environment, then install the pinned dependencies:

~~~powershell
python -m pip install -r requirements.txt
~~~

Run the complete test suite:

~~~powershell
python -m pytest -q
~~~

Start the local application:

~~~powershell
python -m streamlit run streamlit_app.py
~~~

Build the compact dashboard database after the modeling and odds pipelines
have completed:

~~~powershell
python -m src.deployment.build_dashboard_snapshot
~~~

Run the full production refresh only after configuring the required API and
publication credentials:

~~~powershell
python -m src.pipeline.run_production_refresh --online --refresh-sources --publish
~~~

Operational modes, audit behavior, forward snapshots and publication rules are
documented in
[docs/operations/in_season_refresh.md](docs/operations/in_season_refresh.md).

## Technical Documentation

The README is the portfolio-level overview. Detailed implementation evidence is
kept in [docs/](docs/README.md):

- [modeling and prediction](docs/data_model/modeling_and_prediction.md);
- [model governance and probability blending](docs/data_model/model_governance_and_blending.md);
- [injury and player usage](docs/data_model/injury_and_player_usage.md);
- [current production predictions](docs/data_model/current_production_predictions.md);
- [Spread predictions](docs/data_model/current_spread_predictions.md);
- [Totals predictions](docs/data_model/current_totals_predictions.md);
- [current betting board](docs/data_model/current_betting_board.md);
- [historical market evaluation](docs/data_model/historical_market_evaluation.md);
- [Streamlit product blueprint](docs/product/streamlit_ui_blueprint.md);
- [production refresh runbook](docs/operations/in_season_refresh.md).

The repository-level [sql/](sql/) directory contains the reproducible
data-quality suites for the main physical and analytical layers.

## Forward-Test Policy

The selected production models remain frozen during the 2026 forward test.
Changes are promoted only after time-based evaluation and explicit governance
review. Pregame predictions and market observations are archived so that future
performance and closing-line movement can be evaluated prospectively.

## Disclaimer

Predictions are probabilistic estimates, not guaranteed outcomes. A positive
model–market probability gap or expected value does not guarantee that an individual wager will be
profitable. The platform is an independent analytics project and is not
affiliated with the NFL or its teams.
