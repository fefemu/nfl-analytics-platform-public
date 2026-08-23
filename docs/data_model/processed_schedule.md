# Processed Schedule Data Model

## Purpose

The `processed.schedule` dataset is the primary cleaned and standardized
game-level dataset used by downstream feature engineering, machine learning,
betting analytics and reporting.

It contains validated schedule data together with derived business fields
that are independent from any machine learning model.

---

## Load Strategy

The `processed.schedule` dataset is fully rebuilt during every pipeline run.

The source `schedule` table is loaded using a full-refresh strategy, and the
processed dataset is deterministically derived from the current source state.

An incremental load is not used because:

- the dataset is relatively small;
- historical records may be corrected or enriched by the source;
- a full rebuild keeps the transformation logic simple and reproducible;
- the processed dataset can be recreated entirely from the raw schedule data.


## Design Principles

- Raw data remains immutable.
- One record represents one NFL game.
- Derived fields are deterministic.
- No machine learning features are created in this layer.
- No rolling statistics are calculated in this layer.
- The dataset supports Moneyline, Spread and Totals markets.

## Assumptions

- One record represents one NFL game.
- `game_id` uniquely identifies a game.
- Historical source records may be corrected or enriched.
- Future games may not contain final scores.
- Betting market fields may be missing for older seasons or future games.
- Derived fields are calculated only from the current raw schedule state.

## Dataset Structure

The dataset contains four categories of fields:

### 1. Business Keys

Fields that uniquely identify an NFL game.



### 2. Source Fields

Fields copied directly from the raw schedule dataset.

### 3. Derived Fields

Deterministic fields calculated from the source data.

### 4. Future Extensions

Fields reserved for future feature engineering and betting models.

## Field Definitions

### Business Keys

Business keys uniquely identify an NFL game and define the natural grain of the dataset.

One record always represents one NFL game.

Business key fields are mandatory except for `gametime`, which may be unavailable for older historical games.
If any business key is missing or invalid, the pipeline must fail during the processing stage and the dataset must not be created.

| Column | Source | Data Type | Nullable | Description |
|----------|----------|-----------|----------|-------------|
| game_id | Raw | VARCHAR | No | Unique identifier of the NFL game. Primary business key of the dataset. |
| season | Raw | INTEGER | No | NFL season in which the game is played. |
| game_type | Raw | VARCHAR | No | Game type (REG, WC, DIV, CON, SB). |
| week | Raw | INTEGER | No | NFL week number within the season. |
| gameday | Raw | DATE | No | Scheduled game date. Converted from the raw source during processing. |
| gametime | Raw | TIME | Yes | Scheduled kickoff time (local stadium time). May be unavailable for older historical games. |

### Source Fields

Source fields are copied from the raw schedule dataset without business-level aggregation.

They may be standardized or cast to a target data type during processing, but their business meaning remains unchanged.

| Column | Source | Data Type | Nullable | Description |
|---|---|---:|:---:|---|
| weekday | Raw | VARCHAR | No | Day of the week on which the game is scheduled. |
| away_team | Raw | VARCHAR | No | Abbreviation of the away team. |
| home_team | Raw | VARCHAR | No | Abbreviation of the home team. |
| location | Raw | VARCHAR | Yes | Indicates whether the game is played at the home venue or at a neutral site. |
| away_score | Raw | INTEGER | Yes | Final score of the away team. Null for games without a final result. |
| home_score | Raw | INTEGER | Yes | Final score of the home team. Null for games without a final result. |
| overtime | Raw | INTEGER | Yes | Indicates whether the game went to overtime, according to the source dataset. |
| away_rest | Raw | INTEGER | No | Number of days since the away team's previous game. |
| home_rest | Raw | INTEGER | No | Number of days since the home team's previous game. |
| away_moneyline | Raw | INTEGER | Yes | Closing away team moneyline odds. |
| home_moneyline | Raw | INTEGER | Yes | Closing home team moneyline odds. |
| spread_line | Raw | DOUBLE | Yes | Closing point spread. |
| away_spread_odds | Raw | INTEGER | Yes | Closing odds for the away team against the spread. |
| home_spread_odds | Raw | INTEGER | Yes | Closing odds for the home team against the spread. |
| total_line | Raw | DOUBLE | Yes | Closing over/under line. |
| under_odds | Raw | INTEGER | Yes | Closing odds for the under market. |
| over_odds | Raw | INTEGER | Yes | Closing odds for the over market. |
| roof | Raw | VARCHAR | Yes | Stadium roof type provided by the source dataset. |
| surface | Raw | VARCHAR | Yes | Playing surface provided by the source dataset. |
| temp | Raw | INTEGER | Yes | Game-time temperature provided by the source dataset. |
| wind | Raw | INTEGER | Yes | Game-time wind speed provided by the source dataset. |
| away_qb_id | Raw | VARCHAR | Yes | Identifier of the away team's starting quarterback. |
| home_qb_id | Raw | VARCHAR | Yes | Identifier of the home team's starting quarterback. |
| away_qb_name | Raw | VARCHAR | Yes | Name of the away team's starting quarterback. |
| home_qb_name | Raw | VARCHAR | Yes | Name of the home team's starting quarterback. |
| away_coach | Raw | VARCHAR | Yes | Away team's head coach. |
| home_coach | Raw | VARCHAR | Yes | Home team's head coach. |
| referee | Raw | VARCHAR | Yes | Game referee provided by the source dataset. |
| stadium_id | Raw | VARCHAR | Yes | Unique stadium identifier provided by the source dataset. |
| stadium | Raw | VARCHAR | Yes | Stadium name provided by the source dataset. |

### Derived Fields

Derived fields are calculated deterministically from source fields during processing.

They contain reusable business logic but do not include rolling statistics, model features or predictions.

| Column | Source | Data Type | Nullable | Description |
|---|---|---:|:---:|---|
| is_completed | Derived | BOOLEAN | No | Indicates whether both final team scores are available. |
| home_win | Derived | BOOLEAN | Yes | Indicates whether the home team won. Null when the game is not completed. |
| away_win | Derived | BOOLEAN | Yes | Indicates whether the away team won. Null when the game is not completed. |
| is_tie | Derived | BOOLEAN | Yes | Indicates whether the game ended in a tie. Null when the game is not completed. |
| point_differential | Derived | INTEGER | Yes | Home score minus away score. Null when the game is not completed. |
| game_result | Derived | VARCHAR | No | Standardized game outcome. Possible values: `HOME_WIN`, `AWAY_WIN`, `TIE`, `NOT_PLAYED`. |
| home_rest_advantage | Derived | INTEGER | Yes | Home team rest days minus away team rest days. Null when either source value is missing. |
| total_points | Derived | INTEGER | Yes | Sum of the home and away scores. Null when the game is not completed. |
| is_regular_season | Derived | BOOLEAN | No | Indicates whether the game belongs to the regular season. |
| is_playoff | Derived | BOOLEAN | No | Indicates whether the game belongs to the NFL postseason. |

> **Note**
>
> Derived fields must be deterministic and reproducible from the current
> `raw.schedule` dataset.
>
> Features requiring historical aggregation, rolling windows, external data
> sources or machine learning belong to the `features` layer and are intentionally
> excluded from this dataset.


### Future Extensions

The following information is intentionally excluded from the
`processed.schedule` dataset and will be introduced in later layers of
the platform.

| Future Dataset | Purpose |
|----------------|---------|
| `features.game_features` | Rolling statistics, Elo ratings, team form, offensive and defensive metrics, weather-derived features and other machine learning features. |
| `predictions.weekly_predictions` | Predicted win probabilities and model outputs. |
| `betting.bet_recommendations` | Expected value, Kelly criterion, stake sizing, betting recommendations and outcome evaluation. |
| `mart.power_bi` | Business-friendly reporting tables and Power BI semantic model. |
