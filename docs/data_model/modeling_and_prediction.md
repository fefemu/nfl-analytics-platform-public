# Modeling and Prediction Data Model

**Project:** NFL Analytics Platform  
**Version:** 0.1.0  
**Status:** Active  
**Last Updated:** 2026-08-02

---

## Purpose

This document describes the main historical feature, modeling, current prediction and season simulation tables of the NFL Analytics Platform.

It defines:

- table grain;
- source dependencies;
- important fields;
- leakage boundaries;
- refresh behavior;
- downstream consumers.

---

## Data Lineage Overview

The primary modeling lineage is:

1. `processed.schedule`
2. `analytics.elo_game_predictions`
3. `processed.team_game_efficiency`
4. `analytics.rolling_team_features`
5. `processed.qb_game_performance`
6. `analytics.qb_rating_history`
7. `analytics.current_qb_ratings`
8. `analytics.game_qb_features`
9. `analytics.game_schedule_features`
10. `analytics.game_modeling_dataset`
11. `analytics.modeling_game_splits`

The current production lineage is:

1. `processed.schedule`
2. `analytics.current_elo_ratings`
3. `analytics.current_game_predictions`
4. `analytics.current_game_prediction_explanations`
5. `analytics.current_season_simulation_summary`
6. `analytics.current_season_win_distribution`

All tables in this document are generated and may be rebuilt by the modeling pipeline.

---

## Historical Rating Tables

### `analytics.elo_game_predictions`

**Grain:** One row per completed NFL game.

**Purpose:** Store the complete leakage-safe historical Elo calculation for every processed game.

Important fields include:

- `game_id`
- `season`
- `game_type`
- `week`
- pregame home and away Elo ratings;
- applied home-field advantage;
- home and away win probabilities;
- actual binary or tied game result;
- postgame home and away Elo ratings;
- rating change;
- model parameters.

Leakage rule:

The probability and pregame rating columns are calculated before the current game result updates either team.

---

### `analytics.current_elo_ratings`

**Grain:** One row per current NFL team.

**Purpose:** Store each team’s latest Elo state after all currently completed games.

Important fields:

- `team`
- `elo_rank`
- `elo_rating`
- `games_played`
- `last_game_id`
- `as_of_gameday`
- `last_completed_season`

Primary consumers:

- current game predictions;
- dynamic season simulation;
- dashboard team-strength views.

At a season boundary, the current rating is regressed toward 1500 using the production season-retention parameter before future-season probabilities are calculated.

---

## Team Efficiency Tables

### `processed.team_game_efficiency`

**Grain:** One row per team per completed game.

**Purpose:** Aggregate play-by-play performance into team-game offensive and defensive observations.

Feature families include:

- EPA per play;
- competitive EPA;
- dropback EPA;
- designed-rush EPA;
- early-down EPA;
- success rate;
- explosive play rate;
- sack rate;
- turnover rate;
- defensive EPA allowed;
- defensive success rate allowed;
- generated sacks and turnovers.

A completed game normally contributes two rows: one for each participating team.

---

### `analytics.rolling_team_features`

**Grain:** One row per team per game.

**Purpose:** Store leakage-safe rolling team form before each game.

Current windows:

- last 4 games;
- last 8 games.

Leakage rule:

Every rolling feature is shifted so that the current game is excluded. Only earlier team-game observations may contribute to the feature.

Completeness flags identify whether the required history exists for each window.

---

## Quarterback Tables

### `processed.qb_game_performance`

**Grain:** One row per qualifying quarterback per game.

**Purpose:** Aggregate QB play-by-play performance and identify the retrospective primary quarterback.

Important concepts:

- dropbacks;
- EPA per dropback;
- opponent adjustment;
- team dropback share;
- primary-QB flag;
- historical team and player identifiers.

The primary-QB identity is retrospective and is not directly eligible as a pregame prediction feature.

---

### `analytics.qb_rating_history`

**Grain:** One row per quarterback-game observation.

**Purpose:** Store the quarterback’s rating state before and after each game.

The rating method includes:

- time decay;
- opponent-adjusted performance;
- 365-day half-life;
- empirical-Bayes shrinkage;
- effective dropback count;
- uncertainty estimate.

Leakage rule:

The pregame QB rating is calculated from games occurring before the current game.

---

### `analytics.current_qb_ratings`

**Grain:** One row per known quarterback.

**Purpose:** Store the most recent QB rating state for current prediction and starter workflows.

This table does not by itself prove that a quarterback will start a future game. A separate timestamped starter snapshot is required for production usage.

---

### `analytics.game_qb_features`

**Grain:** One row per completed game.

**Purpose:** Attach pregame listed-QB ratings to the home and away teams.

Important fields:

- listed home and away QB identifiers;
- listed home and away QB names;
- listed QB ratings;
- rating availability flags;
- effective dropbacks;
- uncertainty;
- listed-QB rating difference.

Only listed pregame QB information is eligible for historical model features.

---

### `analytics.game_qb_audit`

**Grain:** One row per completed game.

**Purpose:** Compare listed pregame quarterbacks with retrospective actual primary quarterbacks.

This table supports data quality and starter-accuracy analysis.

Leakage boundary:

Actual primary QB identity must remain in the audit layer and must not be joined into pregame prediction features.

---

## Schedule Context

### `analytics.game_schedule_features`

**Grain:** One row per scheduled game.

**Purpose:** Store deterministic pregame schedule context.

Important fields:

- home and away rest days;
- rest-days difference;
- short-week flags;
- short-week difference;
- extended-rest flags;
- extended-rest difference;
- post-bye flags;
- post-bye difference.

Current definitions:

- short week: at most 6 rest days;
- extended rest: at least 9 rest days;
- post-bye: at least 13 rest days.

These values are derived entirely from the known schedule and therefore do not require game outcomes.

---
## Injury and Starter Context

### `raw.injury_reports`

**Grain:** One source record per player, team, week and available source snapshot.

**Purpose:** Preserve historical nflverse injury and practice reports without model-facing status cleaning.

Source coverage:

- seasons 2018–2025;
- 45,337 source records;
- 32 teams per season;
- official injury-report status;
- practice-participation status;
- primary and secondary injury descriptions;
- GSIS player identifiers.

Schema history:

- 2018–2024 contains UTC `date_modified` timestamps;
- 2025 contains `season_type` but no source modification timestamp;
- ingestion normalizes both versions into one 17-column Parquet and DuckDB schema.

Multiple timestamped rows for the same player-team-week are valid source history. Exact duplicate snapshot keys are rejected.

---

### `processed.player_game_injury_status`

**Grain:** One row per player, team and scheduled game.

**Purpose:** Connect weekly injury reports to a game, select the final available source snapshot and standardize model-facing availability fields.

Important fields:

- `game_id`;
- team and opponent;
- home/away indicator;
- GSIS player ID;
- position and player name;
- cleaned injury-report status;
- cleaned practice-participation status;
- `is_out`;
- `is_doubtful`;
- `is_questionable`;
- `did_not_practice`;
- `limited_practice`;
- `full_practice`;
- source modification timestamp;
- source timestamp availability flag;
- source snapshot count.

Current row count: 45,318.

Two 2024 player-game records contain multiple timestamped source snapshots. The latest timestamp is selected while the complete source history remains in the raw layer.

Seventeen player-week injury keys from the cancelled 2022 Week 17 Buffalo–Cincinnati game do not have a completed schedule game. They remain in `raw.injury_reports` and are explicitly excluded from the processed player-game table. Any other unexplained schedule join failure causes the build to fail.

Leakage boundary:

The processed table represents the final available weekly game-day report. The 2025 source has no modification timestamp and must not be represented as an earlier daily as-of snapshot.

---

### Depth-Chart Source Generations

The depth-chart layer normalizes two materially different source generations.

#### `raw.depth_charts_legacy`

**Grain:** One source row per listed player, weekly team depth chart and role.

**Purpose:** Preserve the 2018–2024 weekly NFL depth-chart source without removing source duplicates or conflicting role ranks.

Source characteristics:

- seasons 2018–2024;
- 258,942 raw records;
- complete GSIS player identifiers;
- weekly season, game-type, team and week keys;
- offense, defense and special-teams formations;
- listed depth ranks 1–3;
- historical franchise codes are preserved.

Exact duplicate source rows remain in the raw table. Super Bowl bye snapshots may have no week and are preserved in raw storage but do not represent scheduled player-games.

#### `raw.depth_charts_espn`

**Grain:** One source row per timestamped ESPN team depth-chart role.

**Purpose:** Preserve timestamped ESPN depth-chart history for current and future seasons.

Source characteristics:

- source seasons 2025–2026;
- 951,797 raw records;
- 357 distinct snapshot timestamps;
- 32 teams;
- complete ESPN player identifiers;
- approximately 99% GSIS identifier coverage;
- position groups, position slots and listed position ranks.

The raw ESPN table retains the source season derived from the season-level Parquet filename. This is necessary because timestamp history can extend into the following calendar year.

---

### `processed.player_game_depth_chart_legacy`

**Grain:** One row per player, scheduled game and listed legacy depth-chart role.

**Purpose:** Connect weekly NFL depth charts to scheduled games and consolidate duplicate or conflicting source records.

Important behavior:

- joins by season, game type, week and team;
- preserves multiple legitimate roles for one player-game;
- removes exact source duplication at the processed role grain;
- selects the smallest listed rank when one player-role has conflicting ranks;
- derives starter, primary-backup and reserve flags;
- records source row count and conflicting-rank provenance.

Current size:

- 225,962 player-game role records;
- 3,884 team-games;
- 379 roles consolidated from conflicting source ranks.

A rank of 1 represents a listed starter role, rank 2 a primary backup role and rank 3 a reserve role. These are pregame depth-chart designations, not retrospective proof of who actually started or played.

---

### `processed.player_game_depth_chart_espn`

**Grain:** One row per player, scheduled game and timestamped ESPN depth-chart role.

**Purpose:** Select a leakage-safe pregame ESPN snapshot and connect its player roles to scheduled games.

Snapshot rule:

For every team-game, the builder selects the latest available team snapshot whose UTC calendar date is no later than the scheduled game date.

Using a date-level cutoff avoids claiming false timestamp precision because the schedule stores local kickoff time without a timezone. A post-game-date snapshot is never eligible for an earlier game.

Important behavior:

- uses GSIS ID when available;
- otherwise creates a stable player key from the ESPN ID;
- preserves position group, position name and position slot;
- derives starter, primary-backup and reserve flags;
- preserves the selected source timestamp;
- keeps multiple legitimate roles for one player-game.

Current size:

- 95,445 player-game role records;
- 1,114 team-games;
- 382 role records without a GSIS ID;
- no conflicting processed role ranks.

---

### `processed.player_game_depth_chart`

**Grain:** One row per player, scheduled game and depth-chart role.

**Purpose:** Provide one source-independent depth-chart business table across the legacy NFL and timestamped ESPN generations.

The unified table contains:

- 321,407 player-game role records;
- 4,998 team-games;
- 140,208 listed starter roles;
- 382 role records using ESPN identity fallback.

Important fields:

- game, season, week and team context;
- team and opponent;
- home/away indicator;
- source-independent `player_key`;
- GSIS and ESPN identifiers;
- player name and position;
- formation and depth-chart position;
- ESPN position slot where available;
- depth rank and normalized depth tier;
- starter, primary-backup and reserve flags;
- offense, defense and special-teams role flags;
- identifier provenance;
- source generation;
- source snapshot timestamp and availability flag;
- source record and conflicting-rank metadata.

One player may have multiple valid rows in the same game. For example, the same player may hold both an offensive role and a special-teams role. Downstream player-level aggregation must therefore consolidate roles before joining to one-row-per-player injury or snap-share data.

The unified layer does not itself estimate player quality. It provides pregame role importance that will later be combined with prior-game snap share, position importance, injury status and QB replacement quality.

---

## Historical Modeling Layer

### `analytics.game_modeling_dataset`

**Grain:** One row per completed historical game.

**Purpose:** Combine schedule, target, Elo, rolling team features, QB features and schedule context into a single modeling table.

Major field groups:

- identifiers and dates;
- home and away teams;
- binary home-win target;
- Elo pregame ratings and probability;
- rolling offense differences;
- rolling defense differences;
- listed-QB rating difference;
- QB uncertainty and coverage;
- rest and post-bye context;
- history completeness flags.

Leakage rules:

- every feature must represent information available before kickoff;
- current-game statistics cannot enter rolling features;
- actual primary QB fields are excluded;
- the target is used only for training and evaluation.

---

### `analytics.modeling_game_splits`

**Grain:** One row per modeling game.

**Purpose:** Assign reproducible chronological development periods and model eligibility.

Current split assignment:

| Split | Seasons |
|-------|---------|
| Train | 2018–2022 |
| Validation | 2023–2024 |
| Holdout | 2025 |

Important fields:

- `split_name`
- `split_order`
- binary-target eligibility;
- short-history completeness;
- long-history completeness;
- both-QB-rating availability;
- core-model eligibility;
- extended-model eligibility.

The 2025 holdout has been opened for the frozen first-generation model and is no longer considered untouched.

---

## Model Specifications

### Rejected Logistic Candidate

Model name:

`logistic_elo_qb_post_bye`

Version:

`0.1.0`

Features:

- Elo rating difference;
- listed-QB rating difference;
- post-bye difference.

Status:

`rejected_on_2025_holdout`

The model remains in the repository for reproducibility, diagnostics and Data Science Lab reporting.

---

### Production Elo Model

Model name:

`elo`

Version:

`1.0.0`

Parameters:

- K-factor: 45;
- home-field advantage: 50 Elo points;
- season retention: 60%;
- classification threshold: 0.5.

Status:

`production`

---

## Current Prediction Layer

### `analytics.current_game_predictions`

**Grain:** One row per upcoming regular-season or postseason game.

**Purpose:** Store versioned production Elo probabilities for the current schedule.

Important fields:

- game identity and kickoff information;
- home and away teams;
- neutral-site flag;
- current Elo ratings;
- season-regressed pregame Elo ratings;
- applied home-field advantage;
- home and away win probabilities;
- predicted winner;
- rating as-of dates;
- model name and version;
- prediction generation timestamp.

Refresh behavior:

The table is replaced whenever the current prediction builder runs.

If a game is completed, it is removed from this current prediction table during the next rebuild.

---

### `analytics.current_game_prediction_explanations`

**Grain:** One row per current game prediction.

**Purpose:** Store a structured explanation of each production Elo probability.

User-facing fields include:

- favorite;
- underdog;
- favorite win probability;
- matchup label;
- neutral-site probability;
- home-field probability lift.

Technical fields include:

- raw Elo rating edge;
- adjusted Elo rating edge;
- team-strength log-odds contribution;
- home-field log-odds contribution;
- total home log odds.

The technical components exactly reconstruct the stored home-win probability.

Refresh behavior:

The table is rebuilt in the same transaction as current game predictions so that prediction and explanation versions cannot diverge.

---

## Season Simulation Layer

### `analytics.current_season_simulation_summary`

**Grain:** One row per NFL team for the current simulated season.

**Purpose:** Store dashboard-ready team-level Monte Carlo results.

Important fields:

- season;
- team;
- total games;
- expected wins;
- expected losses;
- existing ties;
- median wins;
- 10th-percentile wins;
- 90th-percentile wins;
- most likely wins;
- minimum and maximum simulated wins;
- expected final Elo;
- simulation count;
- random seed;
- model version;
- generation timestamp.

The current implementation simulates the regular season only.

---

### `analytics.current_season_win_distribution`

**Grain:** One row per team and simulated win total.

**Purpose:** Store the full probability distribution of regular-season wins.

Important fields:

- season;
- team;
- win total;
- number of simulations producing the win total;
- probability;
- total simulations;
- random seed;
- model metadata;
- generation timestamp.

For every team:

- simulation counts sum to the configured run count;
- probabilities sum to one;
- expected wins reconstructed from the distribution match the summary table.

---

## Dynamic Simulation Behavior

The simulation uses the current production Elo state.

For every remaining game in chronological order:

1. read both teams’ current simulated ratings;
2. apply home-field advantage unless the site is neutral;
3. calculate the home-win probability;
4. sample the winner;
5. transfer Elo points using the production K-factor;
6. carry updated ratings into later games.

During the season:

- completed real wins, losses and ties are preserved;
- only remaining games are simulated;
- current Elo already reflects completed results.

---

## Refresh and Dependency Order

The modeling pipeline builds the relevant tables in this order:

1. Elo ratings;
2. team-game efficiency;
3. rolling team features;
4. QB game performance;
5. QB ratings;
6. game QB features;
7. game schedule features;
8. game modeling dataset;
9. modeling splits;
10. current predictions and explanations;
11. current season simulation.
The injury and depth-chart context currently refreshes separately:

1. download canonical season-level injury Parquet files;
2. load `raw.injury_reports`;
3. build `processed.player_game_injury_status`;
4. download legacy NFL and timestamped ESPN depth-chart Parquet files;
5. load `raw.depth_charts_legacy` and `raw.depth_charts_espn`;
6. build the legacy and ESPN player-game depth-chart tables;
7. build `processed.player_game_depth_chart`;
8. run the independent injury and depth-chart SQL quality checks.

This context will join the main modeling pipeline only after leakage-safe snap-share and player-impact features are available.

DuckDB writers must not run while DBeaver or another process holds the database file.

---

## Data Quality

Every materialized table is validated in Python during its build.

Independent SQL quality checks are available under:

- `sql/009_elo_quality_checks.sql`
- `sql/010_team_game_efficiency_quality_checks.sql`
- `sql/011_rolling_team_features_quality_checks.sql`
- `sql/012_qb_game_performance_quality_checks.sql`
- `sql/013_qb_ratings_quality_checks.sql`
- `sql/014_game_qb_features_quality_checks.sql`
- `sql/015_game_modeling_dataset_quality_checks.sql`
- `sql/016_modeling_game_splits_quality_checks.sql`
- `sql/017_game_schedule_features_quality_checks.sql`
- `sql/018_current_game_predictions_quality_checks.sql`
- `sql/019_prediction_explanations_quality_checks.sql`
- `sql/020_season_simulation_quality_checks.sql`
- `sql/021_injury_reports_quality_checks.sql`
- `sql/022_depth_chart_quality_checks.sql`

A successful quality-check query returns `issue_count = 0` and `status = PASS` for every check.