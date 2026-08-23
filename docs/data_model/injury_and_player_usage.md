# Injury and Player Usage Data Model

**Project:** NFL Analytics Platform  
**Version:** 0.1.0  
**Status:** Active  
**Last Updated:** 2026-08-05

---

## Purpose

This document describes the historical injury-report, depth-chart, snap-count and injury-impact data model of the NFL Analytics Platform.

The layer converts weekly player availability information into leakage-safe player, team and game-level features.

It supports:

- historical model research;
- spread and totals modeling;
- game prediction explanations;
- player and team injury dashboards;
- future weekly production predictions;
- the Streamlit Data Science Lab.

---

## Design Principles

The injury layer follows these rules:

1. Player availability must be tied to a specific scheduled game.
2. Player importance must use only information available before the target game.
3. Depth-chart role and historical snap usage are modeled separately.
4. A veteran changing teams retains career usage history.
5. Same-team usage history is preferred over career history.
6. Quarterback impact is stored separately from generic offensive burden.
7. Missing injury-report data is not treated as a healthy team.
8. Every completed historical game remains in the game-level feature table.
9. Generated DuckDB tables are reproducible from local Parquet sources.
10. External downloads are not triggered by the modeling pipeline.

---

## Data Lineage

### Injury Reports

1. `data/raw/injuries/injury_reports_*.parquet`
2. `raw.injury_reports`
3. `processed.player_game_injury_status`
4. `analytics.player_game_injury_context`
5. `analytics.player_injury_impact`
6. `analytics.team_game_injury_burden`
7. `analytics.game_injury_features`
8. `analytics.game_modeling_dataset`

### Depth Charts

1. `data/raw/depth_charts/legacy/`
2. `data/raw/depth_charts/espn/`
3. `raw.depth_charts_legacy`
4. `raw.depth_charts_espn`
5. `processed.player_game_depth_chart_legacy`
6. `processed.player_game_depth_chart_espn`
7. `processed.player_game_depth_chart`

### Snap Usage

1. `data/raw/snap_counts/snap_counts_*.parquet`
2. `data/raw/players/player_directory.parquet`
3. `raw.player_snap_counts`
4. `raw.player_directory`
5. `processed.player_game_snap_counts`
6. `analytics.player_snap_share_history`

The depth-chart and snap-history branches join the injury branch in `analytics.player_game_injury_context`.

---

## Source Coverage

### Injury Reports

Historical injury reports cover the 2018–2025 seasons.

The raw layer contains:

- 45,337 source rows;
- 32 teams per season;
- regular-season and postseason reports;
- player identifiers;
- listed position;
- primary and secondary injuries;
- game-status designation;
- practice-status designation;
- source modification timestamps where historically available.

After duplicate snapshot consolidation and schedule matching, the processed layer contains 45,318 player-game injury records.

Seventeen injury keys from the cancelled 2022 Buffalo–Cincinnati game are intentionally excluded because no completed scheduled game exists.

### Depth Charts

Two depth-chart source generations are used.

#### Legacy NFL Depth Charts

The legacy source covers 2018–2024 and is organized by:

- season;
- week;
- team;
- formation;
- position;
- player;
- depth rank.

The processed legacy table contains:

- 225,962 player-role rows;
- 3,884 team-games;
- 379 roles consolidated from conflicting source ranks.

#### ESPN Depth Charts

The ESPN source is used from 2025 onward and contains timestamped roster snapshots.

The processed ESPN table contains:

- 95,445 player-role rows;
- 1,114 team-games;
- 382 role rows without a GSIS identifier.

For each scheduled game, the latest eligible snapshot is selected according to the source timing rules.

#### Unified Depth Chart

The two source generations are standardized into `processed.player_game_depth_chart`.

The unified table contains:

- 321,407 player-role rows;
- 4,998 team-games;
- 140,208 starter-role rows;
- 382 rows without a GSIS identifier.

Source generation is retained so downstream users can distinguish legacy NFL and ESPN records.

### Snap Counts

Snap-count data covers 2018–2025.

The raw source contains:

- 205,354 player-game records;
- all 32 teams in every season;
- offense, defense and special-teams snap counts;
- offense, defense and special-teams snap shares;
- 5,084 unique snap-count players.

Three source special-teams shares equal 1.01 because of upstream rounding. These are accepted during ingestion and normalized to 1.00 in the processed layer.

### Player Directory

The player directory contains:

- 25,035 player records;
- complete GSIS identifier coverage;
- 22,554 PFR identifiers;
- 16,759 ESPN identifiers;
- no duplicate GSIS identifiers;
- no duplicate non-null PFR or ESPN identifiers.

The directory links PFR snap-count identifiers to the GSIS identifiers used by injury and depth-chart data.

Of 205,354 snap-count rows:

- 205,148 match the player directory;
- 206 use a controlled fallback player key;
- the unmatched rows represent 24 players.

---

## Processed Injury Status

`processed.player_game_injury_status` creates one record per player, team and scheduled game.

It performs the following operations:

- standardizes source strings and null values;
- maps the report to a scheduled game;
- identifies the opponent and home/away context;
- selects the latest available snapshot;
- records how many source snapshots were consolidated;
- preserves the source modification timestamp;
- excludes known injury reports associated with an unplayed game.

The final table contains 45,318 rows.

Two player-game records required selection from multiple source snapshots.

For example, if a player changes from `Questionable` to `Out`, the latest eligible source snapshot becomes the final player-game status.

---

## Standardized Status Values

### Game Status

The standardized game-status categories are:

- `Out`
- `Doubtful`
- `Questionable`
- `No game status`

Historical counts are:

- `No game status`: 24,184
- `Questionable`: 11,537
- `Out`: 8,331
- `Doubtful`: 1,266

A missing game status does not automatically mean that the player was healthy. It means the source did not provide a formal game designation for that row.

### Practice Status

The standardized practice categories are:

- `Did Not Participate In Practice`
- `Limited Participation in Practice`
- `Full Participation in Practice`
- `No practice status`

Historical counts are:

- full participation: 21,102;
- did not participate: 12,680;
- limited participation: 11,280;
- no practice status: 256.

Whitespace-only and missing values are normalized before downstream scoring.

---

## Player Snap-Share History

`analytics.player_snap_share_history` contains the historical usage information used to estimate player importance.

It contains 205,354 rows and tracks:

- current-game snap counts and shares;
- offense, defense and special-teams usage;
- prior same-team usage;
- prior career usage;
- number of prior observations;
- the timestamp after which the current game becomes available to future rows.

The table includes:

- 5,084 first-career-history rows;
- 9,607 first-team-history rows;
- 928 multi-team player-seasons.

### Leakage Prevention

The target game is never included in its own pregame usage history.

A historical usage row is eligible only when its `available_after_gameday` value is earlier than the target injury record’s game date.

This prevents postgame snap counts from leaking into pregame injury features.

### Players Changing Teams

A veteran joining a new team is not treated as a rookie.

The lookup order is:

1. use prior usage history with the current team when available;
2. otherwise use prior career history from earlier teams;
3. otherwise use the depth-chart fallback.

This preserves known information about an established player while still recognizing that usage on a new team is uncertain.

---

## Player Injury Context

`analytics.player_game_injury_context` combines:

- processed injury status;
- depth-chart role;
- historical snap usage.

The table contains 45,318 rows.

Coverage includes:

- 42,452 depth-chart matches;
- 44,170 prior snap-history matches;
- 43,503 same-team histories;
- 667 career-history fallbacks.

The snap-history source distribution is:

- `TEAM`: 43,503
- `CAREER`: 667
- `NONE`: 1,148

The depth tiers are:

- `STARTER`: 27,821
- `PRIMARY_BACKUP`: 10,818
- `RESERVE`: 3,813
- `UNKNOWN`: 2,866

Depth information provides a controlled importance fallback when historical snap usage is unavailable.

---

## Injury Impact Scoring

`analytics.player_injury_impact` converts player availability and importance into numeric pregame impact scores.

The calculation separates three concepts:

1. availability severity;
2. player importance;
3. unit assignment.

### Availability Severity

Base game-status weights are:

| Game status | Base severity |
|---|---:|
| Out | 1.00 |
| Doubtful | 0.75 |
| Questionable | 0.35 |
| No game status | 0.00 |

Practice information adjusts the severity when a formal game status exists:

| Practice status | Adjustment |
|---|---:|
| Did Not Participate In Practice | +0.10 |
| Limited Participation in Practice | +0.05 |
| Full Participation in Practice | -0.05 |
| No practice status | 0.00 |

The final availability severity is constrained to a valid range.

### Player Importance

Historical snap usage is the preferred source of player importance.

Usage becomes fully reliable after four prior games. Smaller samples are shrunk toward the depth-chart prior so that one or two historical appearances do not receive excessive weight.

Depth-chart fallback weights are:

| Depth tier | Importance |
|---|---:|
| Starter | 1.00 |
| Primary backup | 0.55 |
| Reserve | 0.25 |
| Unknown | 0.40 |

Special-teams usage receives a weight of 0.25 in the generic importance calculation.

### Impact Definition

Conceptually:

`injury impact = availability severity × estimated player importance`

The table stores unit-specific impact values for:

- quarterback;
- non-quarterback offense;
- defense;
- special teams.

The completed table contains:

- 45,318 player-game rows;
- 21,134 rows with positive injury impact;
- 597 quarterback rows with positive impact;
- 20,537 non-quarterback rows with positive impact;
- 1,148 rows using the depth-only importance fallback.

### Quarterback Handling

Quarterbacks are modeled separately.

Positive quarterback injury impact is not included again in generic offensive injury burden. This prevents double counting because the platform already contains a dedicated listed-QB rating feature.

---

## Team-Game Injury Burden

`analytics.team_game_injury_burden` aggregates player impacts to one row per team and scheduled game.

The schedule is the authoritative table spine. Therefore, teams without injury-report records still receive an explicit row.

The table contains:

- 4,454 team-game rows;
- 4,320 rows with positive total burden;
- 571 rows with quarterback burden;
- 4,312 rows with non-quarterback burden;
- 21 rows without injury-report source data.

Stored measures include:

- injured-player counts;
- status-specific counts;
- starter and backup injury counts;
- quarterback burden;
- non-quarterback offensive burden;
- defensive burden;
- special-teams burden;
- total burden;
- top impacted player information;
- injury-report coverage flag.

### Missing-Data Semantics

Missing injury-report data is not interpreted as zero confirmed injuries.

For a team-game without source coverage:

- `has_injury_report_data` is false;
- count and burden fields use neutral zero values to preserve numeric table integrity;
- rates and top-player details remain null where appropriate;
- downstream models receive an explicit coverage indicator.

Twenty-one missing team-game reports correspond to 14 games, including the full 2023 Divisional, Conference and Super Bowl source gap.

---

## Game-Level Injury Features

`analytics.game_injury_features` joins the home and away team burdens into one row per game.

The table contains:

- 2,227 games;
- 2,213 games with complete two-team injury coverage;
- 14 games with incomplete injury coverage.

It stores:

- home injury measures;
- away injury measures;
- home-minus-away burden differences;
- home and away coverage flags;
- a combined complete-coverage flag.

Important modeling features include:

- non-quarterback offensive burden difference;
- defensive burden difference;
- special-teams burden difference;
- quarterback burden difference;
- total burden difference.

The primary injury-enhanced logistic candidate uses the non-quarterback offense, defense and special-teams burden differences alongside Elo and listed-QB rating differences.

---

## Modeling Dataset Integration

`analytics.game_modeling_dataset` retains all 2,227 historical games.

The injury join produces:

- 2,213 complete-coverage games;
- 14 incomplete-coverage games;
- no unexpected null non-quarterback burden differences after controlled normalization.

Keeping all games in the table allows:

- explicit model eligibility rules;
- complete auditability;
- Elo fallback when injury information is incomplete;
- fair comparison across candidate models.

Historical model experiments requiring injury features use only games with complete injury coverage.

---

## Injury Model Candidate

The leading injury-enhanced logistic component is:

`logistic_elo_qb_unit_burdens`

Its five features are:

1. Elo rating difference;
2. listed-QB rating difference;
3. non-QB offensive injury-burden difference;
4. defensive injury-burden difference;
5. special-teams injury-burden difference.

Its regularization parameter is:

`C = 0.1`

### Expanding-Window Evaluation

Across the 2020–2025 governance period:

| Model | Accuracy | Brier | Log loss |
|---|---:|---:|---:|
| Logistic Elo + QB + unit burdens | 64.91% | 0.220788 | 0.632027 |
| Logistic Elo + QB | 64.35% | 0.221349 | 0.633424 |
| Logistic Elo + QB + post-bye | 64.27% | 0.221378 | 0.633254 |
| Logistic full core | 63.56% | 0.221787 | 0.633981 |
| Elo | 63.64% | 0.223252 | 0.638115 |

The injury model is the best standalone candidate across the full six-season governance sample.

However, its 2025 result is materially weaker than Elo. This is why production selection does not rely on the standalone injury model alone.

---

## Selected Production Blend

The selected 2026 forward-test candidate is:

`elo_injury_logistic_blend`

Model version:

`0.2.0`

The blend uses:

- 70% injury-enhanced logistic probability;
- 30% Elo probability.

The selected weight was determined from out-of-fold predictions across 2020–2025.

### Aggregate Governance Result

| Model | Accuracy | Brier | Log loss |
|---|---:|---:|---:|
| 70% injury / 30% Elo blend | 65.23% | 0.220351 | 0.631040 |
| Injury logistic | 64.91% | 0.220788 | 0.632027 |
| Elo | 63.64% | 0.223252 | 0.638115 |

The blend improves both probability quality and winner accuracy over the individual components across the complete governance period.

Weights between approximately 65% and 80% injury probability produce similar results. Therefore, the selected 70% weight is not dependent on a narrow or unstable optimum.

### Historical 2025 Audit

A separate blend selected without using 2025 assigned:

- 90% injury probability;
- 10% Elo probability.

Applied unchanged to 2025, it performed better than the standalone injury model but worse than Elo.

This demonstrates the value of blending while also showing that model performance varies by season.

### Production Fallback

The live prediction policy is:

- use the 70/30 blend when current injury features have complete coverage;
- use Elo when current injury coverage is incomplete;
- expose the active prediction mode in reporting and the user interface.

The next untouched forward-evaluation season is 2026.

---

## Pipeline Integration

The local modeling pipeline rebuilds the complete injury and player-usage chain without calling external download APIs.

Relevant pipeline stages include:

1. raw injury load;
2. player-game injury-status build;
3. raw depth-chart load;
4. legacy depth-chart build;
5. ESPN depth-chart build;
6. unified depth-chart build;
7. raw snap-count load;
8. raw player-directory load;
9. processed player-game snap counts;
10. player snap-share history;
11. player-game injury context;
12. player injury impact;
13. team-game injury burden;
14. game injury features;
15. modeling dataset;
16. modeling splits;
17. model-governance reporting.

External source downloads remain separate operational steps so routine model rebuilds do not consume API credits or unexpectedly replace source snapshots.

---

## Data Quality

The main SQL quality suites are:

- `sql/021_player_game_injury_quality_checks.sql`
- `sql/022_depth_chart_quality_checks.sql`
- `sql/023_snap_count_quality_checks.sql`
- `sql/024_injury_feature_quality_checks.sql`
- `sql/025_model_governance_quality_checks.sql`

The injury-feature quality suite validates:

- row counts;
- business-key uniqueness;
- schedule coverage;
- leakage-safe snap-history timing;
- valid history-source values;
- valid injury-impact ranges;
- quarterback double-counting prevention;
- missing-data semantics;
- game-level coverage;
- modeling-dataset integration.

The current injury-feature suite contains 23 passing checks.

---

## Operational Refresh Strategy

During the season, injury information should be refreshed more frequently than historical model parameters.

A practical weekly process is:

1. refresh schedules and upcoming games;
2. refresh depth charts;
3. refresh injury reports after official practice reports;
4. rebuild injury context and current game features;
5. regenerate current predictions;
6. refresh odds only at controlled decision points;
7. rebuild the market edge table;
8. rerun simulations when material ratings or player availability change.

Injury and depth-chart refreshes do not consume the external odds API quota.

Odds requests should remain separately controlled because the production API allowance is limited.

---

## Known Limitations

1. Injury reports describe availability but do not directly measure player talent.
2. Snap share measures usage, which is related to but not identical to player quality.
3. Depth charts differ structurally between the legacy NFL and ESPN source generations.
4. Some ESPN depth-chart rows lack GSIS identifiers.
5. Twenty-four snap-count players require fallback identifiers.
6. Historical injury timestamps are unavailable for the 2025 source generation.
7. Fourteen historical games lack complete two-team injury-report coverage.
8. Inactive status, transactions and late game-day surprises may require future sources.
9. The selected production blend still requires a true 2026 forward test.
10. Current live prediction generation must explicitly implement the selected blend and Elo fallback policy.

---

## Planned Extensions

Planned future improvements include:

- current-week injury snapshot ingestion;
- starter and inactive confirmations;
- current-game injury-feature generation;
- live 70/30 blend predictions;
- explicit prediction-mode reporting;
- injury contribution explanations;
- spread and totals injury effects;
- position-group sensitivity analysis;
- weather interactions;
- Streamlit injury dashboards;
- Data Science Lab governance visualizations;
- 2026 forward-performance monitoring.

---

## Summary

The injury layer transforms 45,337 raw historical injury-report rows, two depth-chart source generations and 205,354 player-game snap records into leakage-safe game-level injury features.

Player importance prioritizes same-team snap history, falls back to career history after a team change and uses depth-chart role when no prior usage exists.

Quarterback impact remains separate from generic offensive burden, missing source data is explicitly flagged, and all 2,227 games remain auditable.

Across the complete 2020–2025 governance period, the strongest production candidate is a 70% injury-enhanced logistic and 30% Elo probability blend. The model is selected for a controlled 2026 forward test, with Elo used whenever current injury coverage is incomplete.