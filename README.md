# NFL Analytics Platform

> **Source-available portfolio project.** Copyright (c) 2026 Ferenc Kaizer.
> All rights reserved; this repository is not currently distributed under an
> open-source license. See [COPYRIGHT.md](COPYRIGHT.md) and
> [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

> End-to-end NFL data engineering, predictive modeling, season simulation and betting analytics platform.

---

## About

The NFL Analytics Platform is a long-term portfolio project combining data engineering, leakage-safe feature engineering, statistical modeling, machine learning, sports simulation and betting analytics in one reproducible system.

The project demonstrates the complete lifecycle of a modern Data Science product:

- source ingestion;
- raw, processed and analytics data layers;
- data quality validation;
- historical model development;
- time-based evaluation;
- production prediction generation;
- Monte Carlo season simulation;
- explainable model outputs;
- future betting and public dashboard layers.

---

## Current Status

| Item | Value |
|------|-------|
| Version | 0.2.0 |
| Phase | Production probability, spread, totals and simulation |
| Automated tests | 1,145 passing |
| Historical seasons | 2018–2025 |
| Historical games | 2,227 |
| Current schedule | 272 regular-season games |
| Probability model | External nfelo + QB/injury logistic routing |
| Spread model | External nfelo + external QB Ridge |
| Totals model | EPA + weather + QB + league-environment Ridge |
| Missing-data policy | Explicit probability, spread and totals fallbacks |
| Season simulations | 10,000 dynamic runs from external nfelo ratings |
| Modeling pipeline | 33 validated steps |

The repository contains a complete historical modeling dataset, leakage-safe Elo, QB, player-usage, injury, weather, venue and league scoring-environment features.

The selected probability, spread and totals models are frozen for the 2026 forward test. External nfelo ratings replaced the weaker internal Elo production signal after chronological backtests and the locked 2025 holdout. Missing current inputs are handled through separately validated fallback models rather than silent imputation.

All current 2026 preseason games use probability, spread and totals fallbacks because complete weekly rolling, listed-QB, injury and observed-weather inputs are not yet available.

Production probability explanations, exact logistic feature contributions, bilingual EN/HU narratives, the technical Data Science view, production spread and totals predictions, and dynamic-versus-frozen Elo simulation benchmarking are operational.

Moneyline, Spread and Totals expected-value tables, a combined betting board, model-implied team scores, a leakage-safe historical benchmark and an audited in-season refresh with immutable forward snapshots are operational. The premium Streamlit application foundation is operational; page-level product features and scheduling remain in progress.

---

## Core Architecture

The main DuckDB layers are:

- `raw`: source-aligned ingestion tables;
- `processed`: cleaned and normalized business entities;
- `analytics`: features, ratings, predictions, simulations and reporting-ready outputs.

Important generated analytics tables include:

- `analytics.elo_game_predictions`
- `analytics.current_elo_ratings`
- `analytics.rolling_team_features`
- `analytics.qb_rating_history`
- `analytics.current_qb_ratings`
- `analytics.game_qb_features`
- `analytics.game_qb_audit`
- `analytics.game_schedule_features`
- `analytics.game_modeling_dataset`
- `analytics.modeling_game_splits`
- `analytics.current_game_predictions`
- `analytics.current_game_prediction_explanations`
- `analytics.current_season_simulation_summary`
- `analytics.current_season_win_distribution`
- `analytics.current_game_logistic_feature_contributions`
- `analytics.current_game_prediction_narratives`
- `analytics.current_game_prediction_data_science_view`
- `analytics.current_game_spread_predictions`
- `analytics.current_game_total_predictions`
- `analytics.current_game_score_predictions`
- `analytics.current_season_elo_benchmark_summary`
- `analytics.current_season_elo_benchmark_team_comparison`

The DuckDB file under `data/nfl_analytics.duckdb` is generated locally and is not committed.

---

## Streamlit Application

The bilingual premium Streamlit application, Weekly Overview, Game Center, Season
Simulator, forward-only Betting Board, Data Science Lab and About page are operational
locally. The interface includes
nflverse-sourced team identity, matchup probabilities, implied scores, bilingual
narratives, market comparison, Monte Carlo expected wins and uncertainty, candidate
filters, routing labels and stale-odds warnings.

Run the local application:

```powershell
python -m streamlit run streamlit_app.py
```

Build the compact public dashboard database after the modeling and odds
pipelines have completed:

```powershell
python -m src.deployment.build_dashboard_snapshot
```

The generated `data/deployment/dashboard.duckdb` contains only the tables used
by the public application and is excluded from Git. Local development keeps
using `data/nfl_analytics.duckdb`. A hosted application can use an explicit
local artifact path through `NFL_ANALYTICS_DASHBOARD_DATABASE`, a generic HTTPS
artifact URL through `NFL_ANALYTICS_DASHBOARD_DATABASE_URL`, or a private
GitHub Release asset. Remote artifacts are downloaded into the runtime cache
and opened read-only.

The recommended hosted configuration uses a private GitHub repository and a
fine-grained, read-only token stored only in the hosting provider's secret
manager:

```text
NFL_ANALYTICS_DASHBOARD_GITHUB_REPOSITORY=fefemu/nfl-analytics-platform-data
NFL_ANALYTICS_DASHBOARD_GITHUB_TOKEN=<read-only fine-grained token>
NFL_ANALYTICS_DASHBOARD_GITHUB_ASSET=dashboard.duckdb
```

At startup the application discovers the latest Release, downloads the named
asset through the authenticated GitHub API and caches it by asset URL. Publishing
a new Release therefore updates the hosted data without committing the database
to the public source repository. Scope the token to the private data repository
only, grant it read-only repository contents access and never commit it to Git.

> **Publication warning:** do not attach `dashboard.duckdb` to a public GitHub
> Release when it contains current bookmaker data or third-party source rows.
> The Odds API permits user-facing analytical displays, but not downloadable
> database dumps. Keep the deployment artifact in access-controlled storage and
> expose only the rendered dashboard. See
> [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

The public dashboard supports privacy-conscious Google Analytics 4 page views.
Store the web-stream Measurement ID in the hosting provider's secret manager:

```text
NFL_ANALYTICS_GA4_MEASUREMENT_ID=G-XXXXXXXXXX
```

At process startup, `sitecustomize.py` injects a consent-aware tag into the
outer Streamlit HTML shell. No Google tag loads before the visitor accepts the
one-time analytics banner. The choice is persisted in browser local storage and
can be changed through the small Privacy control. Google Signals, advertising
storage and personalization remain disabled. Page and EN/HU language changes
are tracked without counting same-page Streamlit reruns. If the setting is
absent or invalid, analytics stays disabled and the application continues.

The scheduled artifact refresh remains.

---

## Schedule Pipeline

The schedule pipeline loads source Parquet data into the raw DuckDB layer and builds the validated processed schedule table.

Downloaded source data is intentionally excluded from Git. After a fresh clone,
download the current nflverse schedule first:

```powershell
python -m src.ingestion.download_historical_data
```

Run the schedule pipeline:

```powershell
python -m src.pipeline.run_schedule_pipeline
```

Pipeline steps:

1. Load schedule data into `raw.schedule`.
2. Build and validate `processed.schedule`.

Run the processed schedule quality checks:

```powershell
python -m src.utils.run_sql sql/003_processed_schedule_quality_checks.sql
```

All quality checks should return `issue_count = 0` and `status = PASS`.

---

## Odds Pipeline

The odds pipeline retrieves current NFL Moneyline, Spread and Totals markets from The Odds API.

It:

- stores timestamped raw JSON snapshots;
- normalizes bookmaker markets;
- calculates implied and no-vig probabilities;
- selects the best available price by market line;
- connects external events to schedule game identifiers;
- publishes the latest market board.

Create a local `.env` file based on `.env.example`:

```dotenv
ODDS_API_KEY=your_real_api_key
```

Run the complete current odds pipeline:

```powershell
python -m src.pipeline.run_odds_pipeline
```

Successful runs populate:

- `raw.odds_snapshots`
- `raw.odds_events`
- `raw.odds_markets`
- `processed.odds_market_outcomes`
- `analytics.best_odds_by_line`
- `analytics.odds_event_schedule_bridge`
- `analytics.current_market_board`

Rebuild the odds layers from an existing snapshot without consuming API credits:

```powershell
python -m src.pipeline.run_odds_snapshot_pipeline data/raw/odds/example_snapshot.json
```

Run odds quality checks:

```powershell
python -m src.utils.run_sql sql/004_odds_data_quality_checks.sql
```

```powershell
python -m src.utils.run_sql sql/005_processed_odds_quality_checks.sql
```

```powershell
python -m src.utils.run_sql sql/006_best_odds_quality_checks.sql
```

```powershell
python -m src.utils.run_sql sql/007_odds_event_bridge_quality_checks.sql
```

```powershell
python -m src.utils.run_sql sql/008_current_market_board_quality_checks.sql
```

The local `.env` file and raw odds snapshots are excluded from Git.

---

## Historical Feature Engineering

The historical feature layer covers 2018–2025.

Implemented feature groups include:

- offensive and defensive EPA;
- success rate;
- explosive play rate;
- sack and turnover rates;
- competitive-game efficiency;
- designed-rush and dropback efficiency;
- leakage-safe rolling windows;
- quarterback game performance;
- opponent-adjusted QB EPA;
- empirical-Bayes QB rating shrinkage;
- QB uncertainty;
- rest-day differences;
- short-week indicators;
- extended-rest indicators;
- post-bye indicators.

Historical starter information and actual primary QB information are stored separately so that retrospective audit fields do not leak into pregame model features.

---

The injury ingestion layer:

- downloads season-level injury reports through `nflreadpy`;
- normalizes the 2018–2024 and 2025 source-schema variants;
- preserves timestamped status changes;
- stores canonical season-level Parquet files locally;
- loads 45,337 source records into `raw.injury_reports`.

The processed injury layer creates:

- `processed.player_game_injury_status`;
- one final weekly injury status per player and game;
- standardized `Out`, `Doubtful` and `Questionable` statuses;
- standardized practice-participation statuses;
- explicit availability and practice flags;
- snapshot provenance and source-timestamp metadata.

The processed injury table contains 45,318 player-game records. Seventeen injury records from the cancelled 2022 Week 17 Buffalo–Cincinnati game are preserved in the raw layer but excluded from the player-game layer because the game was not completed or modeled.

The source provides timestamps for every 2018–2024 injury record. The 2025 injury source contains final weekly status but no `date_modified` field, so it is treated as a final game-day report rather than a general historical as-of snapshot.

The depth-chart layer normalizes two source generations:

- 2018–2024 weekly NFL depth charts with complete GSIS identifiers, offense, defense and special-teams roles, and depth ranks 1–3;
- 2025–2026 timestamped ESPN depth charts with ESPN identifiers, approximately 99% GSIS coverage, position slots and daily snapshot history.

The raw source generations are stored separately:

- `raw.depth_charts_legacy`;
- `raw.depth_charts_espn`.

They are processed into:

- `processed.player_game_depth_chart_legacy`;
- `processed.player_game_depth_chart_espn`;
- `processed.player_game_depth_chart`.

The unified table contains 321,407 player-game role records across 4,998 team-games. It includes 140,208 listed starter roles and 382 role records that use an ESPN identifier because no GSIS identifier is available.

Legacy weekly records are connected directly to their season, week, team and scheduled game. Exact source duplicates are consolidated, and the best listed rank is selected when one player-role has conflicting ranks.

For timestamped ESPN data, each team-game uses the latest team snapshot whose UTC calendar date is no later than the scheduled game date. This prevents a later depth-chart snapshot from leaking into an earlier game.

A player may have multiple valid role records in the same game, such as an offensive role and a special-teams role. The table therefore uses player-game-role grain rather than one row per player-game.

The next feature-engineering step is leakage-safe recent snap share. QB injuries will use the existing internal QB rating and replacement-quality difference. Non-QB injuries will initially combine depth rank, prior-game snap share and position importance before position-specific player ratings are developed.
---

## Modeling Dataset

The game-level modeling layer contains one row per game and combines:

- schedule;
- pregame Elo;
- rolling offense and defense features;
- leakage-safe listed-QB features;
- schedule-context features;
- binary home-win target;
- reproducible time-based split assignments.

Configured development split:

| Split | Seasons | Core-eligible games |
|-------|---------|--------------------:|
| Train | 2018–2022 | 1,043 |
| Validation | 2023–2024 | 410 |
| Final holdout | 2025 | 215 |

The 2025 holdout has been opened and is no longer treated as an untouched development holdout.

---

## Model Evaluation

Primary model-selection metrics:

1. Brier score;
2. log loss;
3. accuracy as a secondary metric.

Evaluated model families include:

- Elo;
- logistic regression;
- histogram gradient boosting;
- XGBoost.

The leading validation logistic candidate used:

- Elo rating difference;
- listed-QB rating difference;
- post-bye difference.

It achieved the following 2023–2024 validation result:

| Model | Accuracy | Brier | Log loss |
|-------|---------:|------:|---------:|
| Logistic Elo + QB + post-bye | 68.05% | 0.213537 | 0.616473 |
| Elo | 64.88% | 0.219363 | 0.630341 |

The frozen logistic candidate did not generalize to the 2025 holdout:

| Model | Accuracy | Brier | Log loss |
|-------|---------:|------:|---------:|
| Logistic Elo + QB + post-bye | 60.00% | 0.236957 | 0.667263 |
| Elo | 64.19% | 0.230111 | 0.652224 |

That first-generation logistic candidate was rejected for deployment.

Later external-source governance compared internal Elo, external nfelo ratings, QB adjustments, injury-enhanced logistic specifications and probability blends on identical expanding-window samples. The selected 2026 production routing is:

`70% external_nfelo_qb_injury_logistic + 30% published_nfelo_probability`

The production policy uses an external Elo + external QB logistic fallback whenever the complete primary feature set is unavailable.

The next truly prospective evaluation period is the 2026 season.

---

## Production Modeling Pipeline

Before running DuckDB writers, disconnect DBeaver and any other process holding the database file.

Run the complete modeling and prediction pipeline:

```powershell
python -m src.pipeline.run_modeling_pipeline
```

The pipeline executes 33 validated steps:

1. external nfelo game ratings;
2. internal Elo ratings;
3. team-game efficiency;
4. rolling team features;
5. QB game performance;
6. QB ratings;
7. game QB features;
8. game schedule features;
9. normalized venue and weather features;
10. leakage-safe league scoring environment;
11. injury-report loading;
12. player-game injury status;
13. depth-chart loading;
14. legacy player-game depth charts;
15. ESPN player-game depth charts;
16. unified player-game depth charts;
17. snap-count loading;
18. player-directory loading;
19. normalized player-game snap counts;
20. leakage-safe player snap-share history;
21. player-game injury context;
22. player injury impact;
23. team-game injury burden;
24. game injury features;
25. game modeling dataset;
26. modeling split assignments;
27. model-governance reporting;
28. historical OOF market evaluation and virtual ledger;
29. current production probabilities, explanations and narratives;
30. current production spread predictions;
31. current production totals predictions;
32. model-implied home and away score predictions;
33. dynamic current-season simulation and Elo benchmark.

The complete pipeline currently finishes in approximately 80 seconds on the development machine.

---

## Current Game Predictions

The selected production prediction layer uses:

- external nfelo ratings and published probabilities;
- external nfelo QB adjustments;
- listed-QB rating difference;
- offense injury-burden difference;
- defense injury-burden difference;
- special-teams injury-burden difference;
- 70% primary logistic and 30% published nfelo probability weights;
- explicit external Elo + external QB logistic fallback;
- versioned model metadata.

Current outputs are stored in:

- `analytics.current_game_predictions`;
- `analytics.current_game_prediction_explanations`;
- `analytics.current_game_logistic_feature_contributions`.

The explanation layer provides:

- final Elo, logistic and blended probabilities;
- the adjustment from Elo to the final production probability;
- prediction mode and fallback reason;
- exact standardized feature values;
- fitted logistic coefficients;
- signed log-odds contributions;
- contribution ranking;
- intercept and complete logistic probability reconstruction.

Run the related quality checks:

```powershell
python -m src.utils.run_sql sql/018_current_game_predictions_quality_checks.sql
python -m src.utils.run_sql sql/019_prediction_explanations_quality_checks.sql
python -m src.utils.run_sql sql/026_logistic_feature_contributions_quality_checks.sql
```

---

## Current Spread Predictions

The selected production spread model estimates expected home scoring margin with:

- external nfelo rating difference;
- external nfelo QB adjustment difference;
- standardized Ridge regression;
- `alpha = 10`.

The external feature pair covers all current games; listed-QB availability remains audit metadata and no longer changes Spread routing.

The frozen model achieved:

- validation MAE: `10.197`;
- 2025 holdout MAE: `9.999`;
- holdout RMSE: `12.795`;
- holdout R-squared: `0.153`;
- holdout MAE improvement over the constant baseline: `7.57%`.

Current outputs are stored in:

- `analytics.current_game_spread_predictions`.

Run the related quality checks:

```powershell
python -m src.utils.run_sql sql/030_current_spread_predictions_quality_checks.sql
```

Detailed methodology and routing documentation:

- `docs/data_model/current_spread_predictions.md`

---

## Current Totals Predictions

The selected production totals model estimates expected combined score with:

- four-game offensive EPA aggregate;
- four-game defensive EPA-allowed aggregate;
- venue and continuous weather context;
- listed-QB rating aggregate;
- the previous 64-game league scoring environment;
- standardized Ridge regression with `alpha = 100`.

The locked primary model achieved:

- 2025 holdout MAE: `10.733582`;
- holdout RMSE: `13.665821`;
- holdout bias: `0.327409`;
- holdout R-squared: `0.034299`;
- `2.783633%` MAE improvement over the constant baseline.

Games without complete four-game rolling and listed-QB inputs use the validation-selected fallback:

- previous 64-game league scoring environment;
- indoor indicator;
- current home-plus-away Elo rating;
- Ridge `alpha = 1`.

The preseason production build currently provides 272 fallback predictions with an expected-total range of approximately `40.21–52.41`.

Current outputs are stored in:

- `analytics.current_game_total_predictions`.

Run the related quality checks:

```powershell
python -m src.utils.run_sql sql/031_current_totals_predictions_quality_checks.sql
```

Detailed methodology and routing documentation:

- `docs/data_model/current_totals_predictions.md`

---

## Model-Implied Team Scores

The Spread and Totals outputs are combined into internally consistent team-score estimates:

- implied home score = `(predicted total + predicted home margin) / 2`;
- implied away score = `(predicted total - predicted home margin) / 2`.

The 272 current games are stored in `analytics.current_game_score_predictions` with both source models, routing modes and generation timestamps preserved for audit.

Run the related quality checks:

```powershell
python -m src.utils.run_sql sql/034_current_game_score_predictions_quality_checks.sql
```

Detailed documentation:

- `docs/data_model/current_game_score_predictions.md`

---

## Historical Market Evaluation

The 2021–2024 benchmark settles leakage-safe expanding-window OOF selections against the closing market. It reports flat-stake ROI, win/loss/push counts, maximum drawdown, edge buckets and available opening-to-close movement.

Outputs:

- `analytics.historical_betting_ledger`;
- `analytics.historical_betting_performance`.

Moneyline uses a clearly labeled synthetic no-vig closing price because complete historical bookmaker prices are unavailable. Spread and Totals use recorded closing prices. The report is diagnostic evidence, not an automated betting recommendation.

```powershell
python -m src.utils.run_sql sql/035_historical_market_evaluation_quality_checks.sql
```

Detailed documentation:

- `docs/data_model/historical_market_evaluation.md`

---

## Dynamic Season Simulation

The current regular season is simulated 10,000 times.

Each simulated game:

1. calculates its probability from the current simulated Elo state;
2. samples a winner;
3. transfers Elo points between the teams;
4. uses the updated ratings in later games.

The simulator preserves completed real-world wins, losses and ties during the season and simulates only the remaining games.

Dashboard-ready outputs:

- `analytics.current_season_simulation_summary`
- `analytics.current_season_win_distribution`

The summary includes:

- expected wins and losses;
- existing ties;
- median wins;
- 10th and 90th percentiles;
- most likely win total;
- minimum and maximum simulated wins;
- expected final Elo.

Run the simulation directly:

```powershell
python -m src.simulation.build_current_season_simulation --simulations 10000
```

Run its SQL quality checks:

```powershell
python -m src.utils.run_sql sql/020_season_simulation_quality_checks.sql
```

---

## Data Quality Checks

SQL quality checks are stored in the `sql` directory.

Recent model and prediction checks:

```powershell
python -m src.utils.run_sql sql/009_elo_quality_checks.sql
```

```powershell
python -m src.utils.run_sql sql/010_team_game_efficiency_quality_checks.sql
```

```powershell
python -m src.utils.run_sql sql/011_rolling_team_features_quality_checks.sql
```

```powershell
python -m src.utils.run_sql sql/012_qb_game_performance_quality_checks.sql
```

```powershell
python -m src.utils.run_sql sql/013_qb_ratings_quality_checks.sql
```

```powershell
python -m src.utils.run_sql sql/014_game_qb_features_quality_checks.sql
```

```powershell
python -m src.utils.run_sql sql/015_game_modeling_dataset_quality_checks.sql
```

```powershell
python -m src.utils.run_sql sql/016_modeling_game_splits_quality_checks.sql
```

```powershell
python -m src.utils.run_sql sql/017_game_schedule_features_quality_checks.sql
```

All checks should return `PASS`.

---

## Tests

Run the complete automated test suite:

```powershell
python -m pytest -q
```

The suite currently contains more than 350 tests covering:

- ingestion and normalization;
- DuckDB transaction handling;
- schedule and odds pipelines;
- Elo and QB ratings;
- leakage-safe rolling features;
- modeling datasets and splits;
- logistic, boosting and XGBoost experiments;
- calibration and holdout evaluation;
- current predictions and explanations;
- dynamic season simulation;
- pipeline dependency order.

Run the diff formatting check before every commit:

```powershell
git diff --check
```

---

## Development Workflow

1. Disconnect DBeaver before DuckDB writes.
2. Make one bounded change.
3. Run targeted tests.
4. Run the complete test suite.
5. Run `git diff --check`.
6. Commit source code, tests, SQL and documentation.
7. Do not commit `data/nfl_analytics.duckdb`.
8. Push through GitHub Desktop.
9. Discard the generated DuckDB modification after the commit.

---

## Planned Work

Near-term roadmap:

1. finalize refresh timing and quota-aware Windows Task Scheduler policy;
2. build the Streamlit Game Center, Season Simulator and Data Science Lab;
3. add privacy-conscious Google Analytics;
4. monitor the frozen models and prospective CLV during the 2026 forward test.

The production model remains replaceable. New feature groups are promoted only when they improve time-based probability metrics and remain stable outside their development period.

---

## Documentation

Additional project documentation is available in the `docs` directory.
