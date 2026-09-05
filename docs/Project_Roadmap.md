# Project Roadmap

**Project:** NFL Analytics Platform
**Version:** 0.1.0
**Status:** Active
**Last Updated:** 2026-09-04

---

## Purpose

This document defines the implementation order and major delivery milestones of the NFL Analytics Platform.

The roadmap is intentionally evidence-driven. A feature or model is promoted only after leakage-safe construction, time-based evaluation and reproducible validation.

---

## Current Position

The historical data platform, injury feature engineering, weather and venue context, model governance, bilingual prediction explanations, technical Data Science view, dynamic and frozen Elo simulation benchmarking, and production probability, spread and totals predictions are complete.

The current production system provides:

- historical schedule and play-by-play data;
- bookmaker odds ingestion and market normalization;
- leakage-safe Elo and QB ratings;
- rolling offense and defense features;
- schedule-context features;
- historical injury reports and player-game injury status;
- normalized legacy NFL and timestamped ESPN depth charts;
- player snap counts and leakage-safe prior-game snap share;
- player-level injury impact and team-game injury burden;
- a game-level modeling dataset;
- expanding-window model governance;
- external nfelo source ingestion with complete historical modeling coverage;
- a selected 70% external Elo/QB/injury logistic and 30% published-nfelo blend;
- automatic external Elo + external QB logistic fallback;
- exact standardized logistic feature contributions;
- current production game probabilities and explanations;
- 10,000-run dynamic regular-season simulation.
- bilingual EN/HU game narratives;
- a technical prediction Data Science view;
- frozen-versus-dynamic Elo simulation benchmarking;
- a validated external nfelo + external QB production spread model;
- current production spread predictions.
- normalized venue and weather features;
- leakage-safe 32-, 64- and 128-game league scoring environments;
- a validated EPA + weather + QB production totals model;
- an early-season league-environment + indoor + Elo totals fallback;
- current production totals predictions;
- Moneyline, Spread and Totals expected-value tables;
- a combined cross-market betting board;
- a leakage-safe 2021–2024 historical closing-market benchmark and virtual ledger;
- flat-stake ROI, probability-gap bucket, drawdown and opening-to-close movement reporting;
- algebraically consistent model-implied home and away scores;
- external-nfelo initialized dynamic and frozen season simulations.

The current development area is automated in-season refresh. Prospective snapshot-based CLV tracking and Streamlit publication follow afterward.

---

## Milestone Summary

| Milestone | Goal | Status |
|-----------|------|--------|
| M1 | Project foundation and repository standards | Completed |
| M2 | Historical schedule and DuckDB data platform | Completed |
| M3 | Odds ingestion and market analytics | Completed |
| M4 | Elo, PBP, rolling and QB feature engineering | Completed |
| M5 | Modeling dataset and time-based evaluation | Completed |
| M6 | First-generation production model selection | Completed |
| M7 | Current game prediction and explanation pipeline | Completed |
| M8 | Dynamic regular-season simulation | Completed |
| M9 | Injury, starter and roster context | Completed |
| M10 | Weather and venue context | Completed |
| M11 | Spread and totals models | Completed |
| M12 | Betting edge and expected-value engine | Completed |
| M13 | Automated in-season refresh workflows | Completed for first generation |
| M14 | Streamlit public application | Planned |
| M15 | Version 1.0 documentation and release | Planned |

---

## Prioritized Backlog — Betting Methodology and 2026 Forward Performance

**Added:** 2026-09-04

**Priority:** High

**Scope constraint:** Reuse the existing snapshot/archive pipeline and do not change production prediction models.

### Delivery sequence

| Phase | Scope | Estimate | Target order |
|------|-------|----------|--------------|
| P0 | Audit the append-only forward archive, locked price/model-version fields, snapshot cadence and post-kickoff join boundaries | Completed 2026-09-05 | Completed before the first 2026 regular-season kickoff |
| P1 | Rename user-facing Edge to Model–market probability gap / Modell–piac eltérés; separate its tooltip and meaning from executable-price EV | Completed 2026-09-05 | Completed |
| P2 | Document equal-weighted no-vig consensus construction and clarify the Totals → Spread → implied team-score relationship in the app and public documentation | Completed 2026-09-05 | Completed |
| P3 | Define and implement reproducible prospective CLV, including both line and price movement for Spread and Total | 3–5 person-days | After the archive audit |
| P4 | Build the forward settlement and performance layer for locked 2026 observations: W/L/push, units, ROI, win rate, average odds, Brier, Log Loss and drawdown | 3–4 person-days | After completed games exist |
| P5 | Add a bilingual Forward Performance / Live Results view with market, week, season and selection-scope filters plus explicit small-sample warnings | 2–3 person-days | After the performance layer |
| P6 | End-to-end tests, time-safety/reproducibility checks, documentation reconciliation and public-repository update | 1.5–2.5 person-days | Release gate |
| P7 | Benchmark out-of-sample Brier/Log Loss against the same-game de-vigged closing market and the expected Brier distribution implied by the frozen probabilities | 2–4 person-days | Evaluation-only; after common-sample market coverage is audited |

Estimated total: **11–19 person-days**, or roughly **2–3 focused working weeks**. Allow **3–4 calendar weeks** if delivered alongside production refresh monitoring and unrelated UI work. P1 and P2 can ship independently in approximately **1.5–2.5 person-days**; the Forward Performance page must not be presented as evidence of profitability until genuine settled forward observations exist.

### Methodology presentation

- replace standalone user-facing `Edge` terminology with `Model–market probability gap`;
- use `Modell–piac valószínűségi eltérés` in Hungarian and retain `%` as the display convention;
- define the gap as model probability minus equal-weighted consensus no-vig probability;
- state explicitly that the gap measures disagreement with the market, not expected betting profit;
- keep EV separate and calculate it from model probability and the best available executable price;
- document that equivalent markets and matching lines are normalized bookmaker by bookmaker before the no-vig probabilities are averaged;
- retain bookmaker weighting, market-quality weighting and closing-market weighting as future research rather than changing aggregation in this backlog item;
- explain that the Totals Ridge model predicts combined points, the Spread model predicts margin, and displayed team scores are algebraically implied by those two predictions.

### Forward data and CLV

- preserve the original locked prediction, executable price, line, timestamp, selection scope, model name and model version;
- store later and final pre-kickoff market observations separately from the locked observation;
- define Moneyline CLV using locked versus closing executable price and a documented implied-probability representation;
- define Spread and Total CLV using both line movement and price movement, without treating different lines as odds-only comparisons;
- store all operands required to reproduce every displayed CLV value;
- join final game results only after completion and never recalculate archived predictions with a newer model;
- audit whether the existing latest-pregame lateral comparison is sufficiently close to kickoff to qualify as the platform's closing observation;
- keep historical backtest observations physically and semantically separate from the 2026 live forward test.

### Forward performance output

At minimum, provide:

- tracked selection/prediction count and explicit sample size;
- wins, losses and pushes where applicable;
- win rate, average odds, flat-stake units and flat-stake ROI;
- average CLV and positive-CLV percentage;
- running drawdown and maximum drawdown;
- Moneyline Brier score and Log Loss;
- breakdowns by Moneyline, Spread and Total, with optional week, season and positive-EV/recommendation scope filters;
- a prominent warning that a small forward sample is not evidence of sustainable profitability.

### Acceptance gates

- no misleading standalone Edge label remains in the application or public documentation;
- probability disagreement and executable-price EV are visibly distinct;
- odds-consensus and implied-score methodology are documented consistently;
- forward ROI contains only genuinely locked, settled forward observations;
- CLV can be reproduced from stored locked and closing operands, including line changes;
- backtest and live-forward results cannot be confused in storage, calculations or UI;
- automated unit, data-quality and integration tests cover settlement, pushes, line/price CLV, drawdown, filtering and missing-close handling;
- existing automated tests remain green and bilingual public documentation is updated.

### P7 — Brier calibration baseline and closing-market benchmark

- use the frozen out-of-sample prediction set without refitting or optimizing the model;
- simulate outcomes as independent Bernoulli draws from the fixed model probabilities and report the mean, median and reference intervals of the resulting Brier distribution;
- also report the analytical expected Brier under those probabilities as a reproducibility check;
- calculate model and de-vigged closing-market Brier score and Log Loss on exactly the same eligible games;
- report paired model-minus-market differences with an appropriate paired bootstrap interval;
- compare model reliability, closing-market reliability and the simulated reference range where bin sizes are sufficient;
- label simulation ranges as reference intervals rather than confidence intervals for model skill;
- explain that a good absolute Brier score does not imply an exploitable betting edge or profitability;
- keep this task evaluation-only and do not tune the production model against the benchmark sample.

---

## Research Backlog — Offseason Cold Starts and Weather Uncertainty

**Added:** 2026-09-04

**Priority:** Medium, with discovery work before model changes

### R1 — Preseason roster continuity

- quantify returning offensive, defensive and special-teams snaps by team and position group;
- capture important arrivals, departures and projected starter changes;
- test whether continuity and turnover features improve Week 1–4 predictions under leakage-safe historical validation;
- expose additional early-season uncertainty for teams with major roster turnover;
- do not promote the feature unless it improves chronological validation and a protected holdout.

Estimated effort: **4–7 person-days** for source audit, feature engineering and evaluation.

### R2 — Player-value unit ratings

- investigate player-level value estimates that can be aggregated into offense, defense and special-teams unit strength;
- keep talent/value measurement separate from the existing player-level injury and availability impact;
- account for depth order and expected usage without inventing ratings for uncovered players;
- evaluate whether bottom-up unit ratings add value beyond regressed nfelo team strength.

Estimated effort: **8–15 person-days**, strongly dependent on reliable historical roster and player-value coverage.

### R3 — QB transition and scheme-fit uncertainty

- retain the existing recency-weighted, opponent-adjusted QB rating as the player-performance baseline;
- investigate explicit new-team, new-coach, new-coordinator and system-transition indicators;
- model transition uncertainty before attempting a directional scheme-fit adjustment;
- require time-safe historical coaching and starter data plus chronological validation.

Estimated effort: **5–10 person-days** after data-source feasibility is confirmed.

### R4 — Timestamped pregame weather forecasts

- ingest timestamped forecasts associated with game and expected kickoff time rather than using future actual weather;
- refresh forecasts during scheduled production runs while preserving each forecast vintage;
- record forecast horizon, issue time, source and missingness;
- use neutral weather values and the existing availability indicator when no eligible forecast exists;
- evaluate forecast-horizon reliability and consider uncertainty/shrinkage for distant forecasts;
- ensure locked predictions retain the exact forecast vintage used.

Estimated effort: **5–8 person-days** for ingestion, archival, modeling integration, validation and documentation, excluding any paid-source procurement.

### Recommended order

1. R1 roster continuity, because it directly addresses the largest Week 1 cold-start limitation.
2. R4 timestamped weather forecasts, because the current system deliberately avoids treating future actual weather as known.
3. R3 QB transition uncertainty.
4. R2 full player-value unit ratings, after historical source feasibility is established.

---

## M1 — Project Foundation

**Status:** Completed

Delivered:

- repository structure;
- Python virtual environment;
- dependency management;
- configuration conventions;
- logging;
- pytest foundation;
- Git and generated-data rules;
- project charter, vision and roadmap.

---

## M2 — Historical Data Platform

**Status:** Completed

Delivered:

- historical schedule ingestion;
- raw and processed DuckDB layers;
- processed schedule validation;
- reproducible schedule pipeline;
- historical team-code handling;
- Parquet-based source storage;
- independent SQL quality checks.

---

## M3 — Odds and Market Analytics

**Status:** Completed

Delivered:

- The Odds API integration;
- Moneyline, Spread and Totals ingestion;
- timestamped raw JSON snapshots;
- normalized bookmaker outcomes;
- implied probabilities;
- no-vig probabilities;
- best available prices;
- odds-to-schedule event bridge;
- current market board;
- offline snapshot rebuild pipeline.

Future extension:

- quota-aware automated refresh scheduling;
- opening, intermediate and closing snapshots;
- closing-line-value tracking.

---

## M4 — Feature Engineering

**Status:** Completed for first generation

Delivered:

- 2018–2025 play-by-play ingestion;
- team-game offensive and defensive efficiency;
- EPA per play;
- success rate;
- explosive play rate;
- sack and turnover features;
- competitive-game efficiency;
- leakage-safe rolling windows;
- QB game performance;
- opponent adjustment;
- empirical-Bayes shrinkage;
- QB uncertainty;
- pregame QB ratings;
- schedule rest features;
- short-week indicators;
- extended-rest indicators;
- post-bye indicators.

Future extensions:

- injury burden;
- confirmed starter status;
- roster continuity;
- weather;
- venue and roof state;
- travel and time-zone context.

---

## M5 — Modeling Dataset and Evaluation

**Status:** Completed

Delivered:

- one row per historical game;
- pregame-only feature construction;
- train split for 2018–2022;
- external validation split for 2023–2024;
- final 2025 holdout;
- core and extended eligibility flags;
- feature-group ablation;
- regularization tuning;
- expanding-window cross-validation;
- calibration analysis;
- season-level diagnostics.

Primary selection metrics:

1. Brier score;
2. log loss;
3. accuracy as a secondary metric.

---

## M6 — First-Generation Model Selection

**Status:** Completed

Evaluated:

- Elo;
- logistic regression;
- histogram gradient boosting;
- XGBoost;
- raw, sigmoid and isotonic calibration.

The leading validation logistic model used:

- Elo rating difference;
- listed-QB rating difference;
- post-bye difference.

The frozen logistic candidate improved 2023–2024 validation metrics but failed to generalize to the 2025 holdout.

The first-generation Elo, QB and post-bye logistic candidate was rejected after the 2025 audit.

Later injury-enhanced governance evaluated frozen candidates and Elo/logistic probability blends across identical expanding-window samples.

Final production decision:

- production model: `elo_injury_logistic_blend`;
- logistic component: `logistic_elo_qb_unit_burdens`;
- logistic weight: 70%;
- Elo weight: 30%;
- missing QB or injury inputs: Elo fallback;
- model version: `0.2.0`;
- 2025: no longer an untouched holdout;
- next prospective test period: 2026.

The selected model remains replaceable when new feature groups demonstrate stable time-based improvement.

---

## M7 — Current Prediction and Explanation Pipeline

**Status:** Completed

Delivered:

- current Elo team ratings;
- offseason rating regression;
- home-field advantage;
- neutral-site handling;
- current game probabilities;
- predicted winners;
- model versioning;
- prediction timestamps;
- structured user-facing explanations;
- technical log-odds decomposition;
- explicit blend and Elo fallback reporting;
- exact standardized logistic feature values;
- fitted coefficient and log-odds contributions;
- contribution ranking;
- intercept and probability reconstruction;
- independent SQL validation.

Current tables:

- `analytics.current_game_predictions`
- `analytics.current_game_prediction_explanations`
- `analytics.current_game_logistic_feature_contributions`

---

## M8 — Dynamic Regular-Season Simulation

**Status:** Completed

Delivered:

- full 272-game regular-season schedule;
- dynamic Elo updates after every simulated game;
- reproducible random seeds;
- 10,000 simulation runs;
- completed in-season record preservation;
- win, loss and tie handling;
- expected records;
- median and percentile win totals;
- most likely win totals;
- expected final Elo;
- full team win distributions;
- dashboard-ready DuckDB outputs.

Current tables:

- `analytics.current_season_simulation_summary`
- `analytics.current_season_win_distribution`

The first version covers the regular season only. Playoff seeding and NFL tiebreakers are deferred to a later extension.

---

## M9 — Injury, Starter and Roster Context

**Status:** Completed

Completed:

1. nflverse injury-source audit;
2. 2018–2025 historical injury coverage assessment;
3. injury source-schema and timestamp audit;
4. canonical season-level injury Parquet ingestion;
5. `raw.injury_reports`;
6. `processed.player_game_injury_status`;
7. final weekly injury snapshot selection;
8. standardized injury and practice statuses;
9. explicit handling of the cancelled 2022 Buffalo–Cincinnati game;
10. legacy NFL and timestamped ESPN depth-chart source audit;
11. canonical 2018–2026 depth-chart Parquet ingestion;
12. `raw.depth_charts_legacy`;
13. `raw.depth_charts_espn`;
14. `processed.player_game_depth_chart_legacy`;
15. `processed.player_game_depth_chart_espn`;
16. unified `processed.player_game_depth_chart`;
17. source-independent starter, backup and reserve tiers;
18. player identifier provenance and ESPN fallback identity;
19. leakage-safe pregame ESPN snapshot selection;
20. Python and independent SQL data-quality coverage.

Current injury coverage:

- 45,337 raw injury records;
- 45,318 processed player-game records;
- complete source timestamps for 2018–2024;
- final weekly status without source timestamps for 2025;
- two player-games selected from multiple timestamped source snapshots;
- 17 injury keys from the cancelled 2022 Week 17 Buffalo–Cincinnati game retained only in raw storage.

Current depth-chart coverage:

- 258,942 raw legacy NFL role records for 2018–2024;
- 951,797 raw timestamped ESPN role records for 2025–2026;
- 321,407 unified player-game role records;
- 4,998 covered team-games;
- 140,208 listed starter roles;
- 382 processed role records without a GSIS identifier;
- offense, defense and special-teams roles;
- normalized starter, primary-backup and reserve tiers;
- full scheduled team-game coverage for the available source seasons.

Injury-to-depth matching:

- 2018: 96.47%;
- 2019: 96.49%;
- 2020: 92.79%;
- 2021: 92.84%;
- 2022: 94.77%;
- 2023: 88.94%;
- 2024: 87.98%;
- 2025: 99.57%.

Unmatched injury players remain valid injury records. They will be represented through explicit missingness rather than silently removed from later team-game aggregates.

Completed after the initial injury and depth-chart foundation:

1. 2018–2025 historical player snap-count ingestion;
2. player identifier normalization;
3. normalized player-game snap participation;
4. leakage-safe prior-game offense, defense and special-teams snap share;
5. injury, depth-chart and recent-usage context;
6. QB replacement-quality handling;
7. non-QB depth, usage and position-importance impact;
8. team-game injury burden;
9. game-level injury features;
10. modeling-dataset integration;
11. feature ablation and expanding-window governance;
12. selected production probability blend;
13. automatic Elo fallback for incomplete current inputs.

Leakage policy:

- 2018–2024 injury timestamps are preserved;
- 2025 final weekly injury reports are treated as game-day information only;
- timestamped ESPN depth charts use the latest snapshot available no later than the scheduled game date;
- snap counts from the game being predicted must never enter that game’s pregame features;
- rolling snap-share features use only completed prior games;
- retrospective actual participation remains an audit or training-label field and must not enter pregame features.

---

## M10 — Weather and Venue Context

**Status:** Completed for first generation

Planned features:

- stadium location;
- indoor, outdoor and retractable-roof classification;
- roof state when available;
- temperature;
- wind speed;
- precipitation;
- snow;
- extreme-weather indicators;
- forecast timestamp;
- time remaining until kickoff.

Weather will be evaluated separately for:

- Moneyline;
- Spread;
- Totals.

The strongest expected use case is the Totals model.

---

## M11 — Spread and Totals Models

**Status:** Completed

### Spread Model

Completed:

1. continuous home-margin target audit;
2. constant baseline;
3. Elo-only Ridge candidate;
4. Elo + listed-QB Ridge candidate;
5. Elo + QB + injury candidate;
6. identical-sample candidate comparison;
7. expanding-window 2021–2024 backtest;
8. Ridge alpha grid;
9. paired injury-value bootstrap;
10. fold-level coefficient diagnostics;
11. frozen model specification;
12. one-time 2025 holdout evaluation;
13. Elo-only missing-QB fallback;
14. current production prediction builder;
15. DuckDB persistence;
16. independent SQL quality checks;
17. production pipeline integration.

Selected primary model:

- `ridge_elo_qb_spread`;
- Elo rating difference;
- listed-QB rating difference;
- `StandardScaler`;
- Ridge `alpha = 100`.

Selected fallback:

- `ridge_elo_spread`;
- Elo rating difference;
- Ridge `alpha = 10`.

Final 2025 holdout:

- 278 games;
- MAE `9.998817`;
- RMSE `12.794622`;
- bias `-0.657993`;
- R-squared `0.152643`;
- `7.573697%` MAE improvement over the constant baseline.

Production table:

- `analytics.current_game_spread_predictions`.

### Totals Model

Completed:

1. combined-score target and baseline audit;
2. four-game and eight-game rolling coverage analysis;
3. pace and historical scoring-form candidates;
4. normalized venue and weather feature engineering;
5. leakage-safe 32-, 64- and 128-game league scoring environments;
6. EPA, weather, QB and scoring-environment candidate comparison;
7. expanding-window 2021–2024 backtest;
8. Ridge alpha selection;
9. paired 64-game versus 128-game bootstrap;
10. fold-level standardized coefficient diagnostics;
11. frozen primary model specification;
12. corrected complete-window holdout protocol;
13. one-time official 2025 holdout evaluation;
14. production-safe early-season fallback selection;
15. current production prediction builder;
16. DuckDB persistence;
17. independent SQL quality checks;
18. 30-step pipeline integration.

Selected primary model:

- `ridge_epa_weather_qb_league_64_totals`;
- four-game offensive EPA aggregate;
- four-game defensive EPA-allowed aggregate;
- indoor and continuous weather context;
- listed-QB rating aggregate;
- previous 64-game league scoring average;
- `StandardScaler`;
- Ridge `alpha = 100`.

Final 2025 holdout:

- 215 games;
- MAE `10.733582`;
- RMSE `13.665821`;
- bias `0.327409`;
- R-squared `0.034299`;
- `2.783633%` MAE improvement over the constant baseline.

Selected fallback:

- `ridge_league_64_indoor_elo_totals`;
- previous 64-game league scoring average;
- indoor indicator;
- current home-plus-away Elo rating;
- Ridge `alpha = 1`.

Production table:

- `analytics.current_game_total_predictions`.

The current preseason build provides predictions for all 272 scheduled games through explicit fallback routing.

The joint spread and totals outputs now produce algebraically consistent model-implied home and away score estimates for every current game.

---

## M12 — Model–Market Probability Gap and Expected Value

**Status:** Completed for first generation

Delivered:

- model probability;
- no-vig market probability;
- model–market probability gap;
- available decimal odds;
- expected value;
- bookmaker and timestamp;
- current-board edge and expected-value fields;
- closing-market historical benchmark using expanding-window OOF predictions;
- virtual flat-stake betting ledger;
- ROI, win rate, push rate and maximum drawdown;
- probability-gap bucket analysis;
- opening-to-close market-movement diagnostics where available.

Moneyline historical return uses a synthetic no-vig closing price because complete bookmaker Moneyline prices are unavailable in the external history. Spread and Totals use recorded closing prices. True prospective CLV depends on timestamped in-season snapshots and must not be inferred from this retrospective benchmark.

The first release will not place real bets automatically.

---

## M13 — Automated In-Season Refresh

**Status:** Completed for first generation

Delivered:

- weekly schedule and result processing;
- postgame Elo and rolling-feature rebuild;
- injury and starter refresh;
- weather refresh;
- quota-aware odds snapshots;
- prediction regeneration;
- season simulation regeneration;
- dashboard data refresh.

Additional operational controls:

- explicit online versus offline odds mode;
- refresh success/failure audit history;
- immutable future market-board archive;
- positive-EV forward candidate flag;
- prospective later-snapshot CLV comparison;
- post-kickoff archive rejection;
- idempotent snapshot reprocessing.

The safe refresh runner is operational. Scheduling can initially use Windows Task Scheduler and later move to a hosted background job; production scheduling remains intentionally disabled until source-update timing and API quota policy are finalized.

---

## M14 — Streamlit Application

**Status:** In progress

The product information architecture and first-release UI baseline are defined in
`docs/product/streamlit_ui_blueprint.md`. The implementation is split into six
incremental UI blocks so the forward-only public Betting Board can ship before the
more technical pages. The visual target is a premium dark analytics command center
with a reusable component system, not a minimally styled default Streamlit dashboard.

Completed foundation:

- six-page application shell and navigation;
- centralized dark theme and responsive visual tokens;
- reusable KPI, status, empty-state and team-identity components;
- read-only DuckDB repository and health state;
- canonical 32-team badge registry with historical aliases;
- graceful missing-database and missing-table states;
- local browser smoke test at desktop viewport.

Completed first product pages:

- Week-selectable overview with all matchup probabilities, model-implied scores,
  Spread and Totals forecasts;
- forward-only Betting Board across Moneyline, Spread and Totals;
- one strongest candidate per game and market in the card view;
- EV, bookmaker-coverage and market filters;
- explicit primary/fallback and stale-market presentation;
- detailed offer table for technical review.
- selected-matchup Game Center with nflverse-sourced team identity;
- accessible opposing win-probability bar and model-implied score;
- Spread and Totals prediction tiles with explicit routing state;
- bilingual model narrative and optional current market comparison;
- technical model and routing identifiers behind an expander.
- ranked expected-wins outlook for all 32 teams;
- selected-team Monte Carlo win distribution and P10–P90 interval;
- 8+, 10+, 12+ and 14+ win probabilities;
- dynamic-versus-frozen Elo comparison when benchmark data is available;
- explicit simulation uncertainty and playoff-probability limitation.
- global EN/HU selector backed by centralized UI copy;
- terminology policy that preserves established analytics terms such as pipeline,
  Brier score, calibration, fallback, expected value, Monte Carlo and Elo;
- curated Data Science Lab with production model, governance comparison and
  season-level Brier score evidence;
- bilingual About page with methodology, sources, limitations, privacy and
  responsible-use disclosures.

Planned product areas:

- richer calibration and feature-contribution charts as monitoring history grows;
- SHAP if a nonlinear production model is promoted;
- deployment, product analytics and automated public refresh.

### Deferred UI Backlog

#### In-place Weekly Overview navigation

- Open matchup details from Weekly Overview in the current browser tab instead
  of a new tab or window.

#### Game Center market evaluation cards

- Replace the current multi-row market comparison with three responsive cards:
  Moneyline, Spread and Total.
- Show exactly one model-preferred outcome per market type while keeping model
  preference separate from betting value. A preferred outcome must remain the
  preferred outcome even when its current price produces zero or negative edge.
- Reuse existing prediction, market and value outputs; do not introduce a new
  prediction model or UI-only probability calculation.
- Each card must show the preferred outcome, model probability, no-vig market
  probability, edge in percentage points, best available odds, bookmaker and a
  value status.
- Reuse an existing project value/edge threshold where available. Otherwise,
  implement configurable thresholds for `VALUE`, `SMALL EDGE` and `NO VALUE`
  rather than hardcoded presentation logic.
- Spread and Total must resolve opposing outcomes for the same line into one
  preferred-side card instead of showing both sides separately.
- Missing odds or market probability must produce an explicit unavailable/no
  current market state without suppressing the model-preferred outcome.
- Add bilingual tooltips for model probability, no-vig market probability,
  edge and best odds, including the warning that positive edge does not
  guarantee profit.
- Preserve the existing Game Center probability, model-score, margin, Total,
  model-explanation and technical-routing sections.
- Keep Game Center focused on one matchup and all three model opinions, while
  Betting Board remains the ranked, next-week cross-game value view.
- Before implementation, audit the current DuckDB output tables and document
  the source fields, preferred-side selection, status thresholds and missing
  market fallback used by the cards.

### Product Analytics

- privacy-conscious Google Analytics integration;
- total users and sessions;
- traffic source and campaign attribution;
- country- and region-level audience reporting;
- device-category and screen-size reporting;
- page and feature usage;
- returning-user trends;
- no collection of model inputs or sensitive user data;
- consent and privacy disclosure appropriate to the deployment region.

---

## M15 — Version 1.0 Release

**Status:** Planned

Release requirements:

- reproducible end-to-end pipelines;
- automated tests;
- SQL quality checks;
- data model documentation;
- methods documentation;
- ER diagram;
- public Streamlit deployment;
- model limitations;
- responsible betting disclaimer;
- refresh runbook;
- clean repository and release tag.

---

## Development Principles

1. Pregame predictions may use only information available before kickoff.
2. Time-based validation takes priority over random train-test splits.
3. Brier score and log loss take priority over headline accuracy.
4. A complex model must beat simpler baselines outside its training period.
5. Market data and independent football-model data must remain distinguishable.
6. Every generated table requires Python validation and SQL quality checks.
7. Documentation is part of the definition of done.
8. Generated DuckDB changes are never committed.
