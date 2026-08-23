# NFL Analytics Platform Documentation

**Project:** NFL Analytics Platform  
**Status:** Active  
**Last Updated:** 2026-08-09

---

## Purpose

This directory contains the architecture, data-model, modeling, governance and project-planning documentation for the NFL Analytics Platform.

The documentation is intended for:

- project development;
- data engineering review;
- data science review;
- portfolio presentation;
- future Streamlit users;
- operational maintenance.

---

## Project Documents

### [Project Vision](Project_Vision.md)

Describes the long-term product vision, intended users and planned public analytics platform.

### [Project Charter](Project_Charter.md)

Defines project scope, objectives, deliverables, constraints and success criteria.

### [Project Roadmap](Project_Roadmap.md)

Tracks completed platform layers and future milestones, including:

- prediction modeling;
- injury and weather features;
- spread and totals models;
- betting expected value;
- season simulation;
- Streamlit publication.

### [Glossary](Glossary.md)

Defines the main data engineering, football analytics, modeling, betting and simulation terminology used throughout the repository.

---

## Data-Model Documentation

### [Processed Schedule](data_model/processed_schedule.md)

Documents the canonical processed NFL schedule, historical team-code handling, game identifiers and schedule-related data-quality rules.

### [Modeling and Prediction](data_model/modeling_and_prediction.md)

Documents:

- game modeling dataset;
- time-based splits;
- Elo baseline;
- QB features;
- rolling team features;
- logistic and boosting experiments;
- current game predictions;
- season simulation.

### [Injury and Player Usage](data_model/injury_and_player_usage.md)

Documents the complete injury and player-importance lineage:

- injury reports;
- depth charts;
- snap counts;
- player-directory crosswalk;
- leakage-safe historical usage;
- player injury impact;
- team injury burden;
- game-level injury features;
- missing-data semantics;
- current production blend inputs.

### [Model Governance and Probability Blending](data_model/model_governance_and_blending.md)

Documents:

- common model eligibility;
- expanding-window validation;
- frozen candidate models;
- accuracy, Brier score and log loss;
- 2025 historical audit;
- Elo and injury probability blending;
- selected 70/30 production candidate;
- 2026 forward-test policy;
- Streamlit Data Science Lab reporting design.

---

## Architecture Overview

The platform follows three main DuckDB layers.

### Raw Layer

The raw layer contains source-aligned data loaded from local files or external APIs.

Examples:

- schedules;
- odds snapshots;
- play-by-play data;
- injury reports;
- depth charts;
- snap counts;
- player directory.

Raw tables preserve source information and provide a reproducible ingestion boundary.

### Processed Layer

The processed layer standardizes and connects source records.

Examples:

- canonical schedule;
- player-game injury status;
- player-game depth charts;
- player-game snap counts;
- team-game efficiency;
- QB game performance.

Processed tables resolve identifiers, normalize values and establish stable business keys.

### Analytics Layer

The analytics layer contains model- and reporting-ready outputs.

Examples:

- Elo ratings and predictions;
- rolling team features;
- QB rating history;
- injury impact;
- team injury burden;
- game injury features;
- modeling datasets;
- governance scorecards;
- current predictions;
- season simulations.

---

## Current Production Candidate

The selected probability model for the 2026 forward test is:

`elo_injury_logistic_blend`

Version:

`0.2.0`

Components:

- 70% injury-enhanced logistic probability;
- 30% Elo probability.

The injury-enhanced logistic component uses:

1. Elo rating difference;
2. listed-QB rating difference;
3. non-QB offensive injury-burden difference;
4. defensive injury-burden difference;
5. special-teams injury-burden difference.

When complete current injury information is unavailable, the production policy uses Elo fallback.

The active prediction mode must be exposed as either:

- `BLEND`;
- `ELO_FALLBACK`.

The next untouched evaluation period is the 2026 season.

---

## Model Governance Reporting

The following generated DuckDB tables support model review and the future Data Science Lab:

- `analytics.model_governance_scorecard`
- `analytics.model_governance_season_results`
- `analytics.model_blend_weight_grid`
- `analytics.model_blend_scorecard`
- `analytics.production_model_registry`

These tables provide:

- aggregate candidate rankings;
- season-level results;
- blend-weight sensitivity;
- historical audit results;
- selected production configuration.

---

## SQL Quality Checks

SQL quality checks are stored in the repository-level `sql` directory.

### Platform and Schedule

- `001_data_quality_checks.sql`
- `002_schema_inspection.sql`
- `003_processed_schedule_quality_checks.sql`

### Odds and Markets

- `004_odds_data_quality_checks.sql`
- `005_processed_odds_quality_checks.sql`
- `006_best_odds_quality_checks.sql`
- `007_odds_event_bridge_quality_checks.sql`
- `008_current_market_board_quality_checks.sql`

### Elo and Team Efficiency

- `009_elo_quality_checks.sql`
- `010_team_game_efficiency_quality_checks.sql`
- `011_rolling_team_features_quality_checks.sql`

### Quarterback Features

- `012_qb_game_performance_quality_checks.sql`
- `013_qb_ratings_quality_checks.sql`
- `014_game_qb_features_quality_checks.sql`

### Modeling and Prediction

- `015_game_modeling_dataset_quality_checks.sql`
- `016_modeling_game_splits_quality_checks.sql`
- `017_game_schedule_features_quality_checks.sql`
- `018_current_game_predictions_quality_checks.sql`
- `019_prediction_explanations_quality_checks.sql`
- `020_season_simulation_quality_checks.sql`
- `026_logistic_feature_contributions_quality_checks.sql`

### [Current Production Predictions](data_model/current_production_predictions.md)

Documents the live pregame prediction layer:

- the selected 70% injury-logistic and 30% Elo blend;
- automatic Elo fallback for incomplete QB or injury inputs;
- prediction routing and model metadata;
- auditable probability explanations;
- current limitations and season-simulation policy.

### [Current Production Spread Predictions](data_model/current_spread_predictions.md)

Documents:

- chronological spread model selection;
- Ridge regularization evaluation;
- injury-feature rejection;
- final 2025 holdout performance;
- Elo + QB primary model;
- Elo-only fallback;
- current production routing;
- `analytics.current_game_spread_predictions`.

### [Current Production Totals Predictions](data_model/current_totals_predictions.md)

Documents:

- chronological totals model selection;
- normalized venue and weather features;
- leakage-safe league scoring environment;
- 64-game versus 128-game window comparison;
- expanding-window validation;
- final 2025 holdout performance;
- primary and early-season fallback routing;
- `analytics.current_game_total_predictions`.

### [Current Model-Implied Team Scores](data_model/current_game_score_predictions.md)

Documents:

- the exact Spread + Totals score identities;
- source metadata and prediction lineage;
- nonnegative-score and winner validation;
- `analytics.current_game_score_predictions`.

### [Historical Market Evaluation](data_model/historical_market_evaluation.md)

Documents:

- 2021–2024 expanding-window OOF market evaluation;
- flat-stake ROI and drawdown;
- edge-bucket and opening-to-close movement diagnostics;
- Moneyline synthetic-price limitation;
- `analytics.historical_betting_ledger` and `analytics.historical_betting_performance`.

### Injury, Depth Chart and Player Usage

- `021_player_game_injury_quality_checks.sql`
- `022_depth_chart_quality_checks.sql`
- `023_snap_count_quality_checks.sql`
- `024_injury_feature_quality_checks.sql`

### Model Governance

- `025_model_governance_quality_checks.sql`
- `027_prediction_narratives_quality_checks.sql`
- `028_prediction_data_science_view_quality_checks.sql`
- `029_elo_simulation_benchmark_quality_checks.sql`
- `030_current_spread_predictions_quality_checks.sql`
- `031_current_totals_predictions_quality_checks.sql`
- `032_current_totals_value_quality_checks.sql`
- `033_current_betting_board_quality_checks.sql`
- `034_current_game_score_predictions_quality_checks.sql`
- `035_historical_market_evaluation_quality_checks.sql`
- `036_forward_refresh_quality_checks.sql`

---

## Running a SQL Quality Check

Disconnect DBeaver and close GitHub Desktop before running DuckDB write operations or checks that may conflict with another process.

Example:

`python -m src.utils.run_sql sql/025_model_governance_quality_checks.sql`

A successful quality-check suite should report only `PASS` rows.

---

## Modeling Pipeline

The local modeling pipeline rebuilds the generated DuckDB modeling and reporting chain from existing local source files.

Run:

`python -m src.pipeline.run_modeling_pipeline`

The pipeline currently contains 33 ordered steps, beginning with the external nfelo source refresh.

It rebuilds:

- Elo ratings;
- team efficiency;
- rolling features;
- QB features;
- schedule features;
- injury data;
- depth charts;
- snap history;
- injury impact and burden;
- modeling dataset and splits;
- governance reporting;
- current predictions;
- current season simulation.
- bilingual prediction narratives;
- technical prediction Data Science view;
- current spread predictions;
- frozen-versus-dynamic Elo benchmark.
- normalized venue and weather features;
- leakage-safe league scoring environment;
- current totals predictions;
- model-implied home and away score predictions;
- leakage-safe historical market evaluation and virtual ledger;

The modeling pipeline refreshes the external nfelo source with retry handling. The odds API remains a separate, explicitly invoked pipeline.

This prevents routine rebuilds from:

- consuming odds API credits;
- changing historical source snapshots;
- depending on network availability.

---

## Validation Workflow

Before committing a completed development block:

1. disconnect DBeaver;
2. close GitHub Desktop during DuckDB operations;
3. run targeted tests;
4. rebuild the relevant pipeline;
5. run the applicable SQL quality checks;
6. run the complete test suite;
7. run Git whitespace validation;
8. inspect changed files;
9. exclude the generated DuckDB file from the commit;
10. commit and push through GitHub Desktop;
11. discard only the generated DuckDB modification after the push.

Full test suite:

`python -m pytest -q`

Whitespace validation:

`git diff --check`

The generated file below is not committed:

`data/nfl_analytics.duckdb`

---

## Data Refresh Boundaries

External ingestion and local model rebuilding are separate operations.

### External Refreshes

Examples:

- injury download;
- depth-chart download;
- snap-count download;
- player-directory download;
- odds API refresh.

These operations may use external resources or replace local raw snapshots.

### Local Rebuilds

Examples:

- raw Parquet loading;
- processed table building;
- analytics feature generation;
- model governance;
- current prediction generation;
- simulation.

These operations use existing local sources and do not consume the odds API allowance.

---

## Planned Documentation

Future documentation should include:

- physical DuckDB schema reference;
- entity-relationship diagram;
- weekly production runbook;
- odds API quota runbook;
- current prediction and fallback specification;
- spread and totals model documentation;
- betting expected-value methodology;
- Streamlit application guide;
- 2026 forward-test report;
- Methods and Data Science Lab guide.

---

## Current Project Direction

Operational instructions for explicit online/offline refresh modes, audit history,
immutable forward snapshots and prospective CLV are documented in
[`operations/in_season_refresh.md`](operations/in_season_refresh.md).

The approved application information architecture, page designs, forward-only rules,
responsive behaviour and team-identity policy are documented in
[`product/streamlit_ui_blueprint.md`](product/streamlit_ui_blueprint.md).

The first Streamlit foundation is available through `streamlit_app.py`; it provides
the shared premium shell, navigation, theme, read-only data boundary and safe empty
states. The Weekly Overview, forward-only Betting Board, selected-matchup Game Center,
Monte Carlo Season Simulator, Data Science Lab and About are connected product pages.
The global EN/HU control translates UI copy while established technical terms remain
unchanged.

The production probability, explanation, spread, totals and season-simulation layers are operational.

Betting expected value across Moneyline, Spread and Totals markets and the combined betting board are operational.

The roadmap then continues with:

- quota-aware refresh scheduling;
- Streamlit publication of forward-only current candidates;
- About, privacy-conscious Google Analytics and public deployment;
- 2026 forward monitoring.

---

## Summary

The documentation set now covers the project vision, roadmap, terminology, core data models, injury and player-usage system, model governance and selected production blend.

The repository’s SQL quality suites provide reproducible validation for every major layer from schedule ingestion through model-governance reporting.

The selected 2026 production candidate is a 70% injury-enhanced logistic and 30% Elo probability blend with explicit Elo fallback when injury coverage is incomplete.
