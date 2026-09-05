# Streamlit Product and UI Blueprint

**Status:** Approved design baseline  
**Scope:** Public, forward-only NFL analytics application  
**Primary language:** Global English/Hungarian UI with shared technical terminology

---

## Product Positioning

The application is an analytics product, not a tipster feed. It explains what the
models currently expect, where those expectations differ from the market, and how
uncertain those estimates are.

The public application must never present reconstructed historical predictions as
past picks. Betting candidates are shown only when they were archived before kickoff.

## Information Architecture

The application uses six pages. Four are primary product destinations and two are
supporting pages.

| Navigation | Page | Primary question |
|---|---|---|
| Home | Weekly Overview | What matters in the current NFL week? |
| Games | Game Center | What does the model expect in this matchup, and why? |
| Betting | Betting Board | Where does the model differ materially from the market? |
| Simulations | Season Simulator | What season outcomes are most likely? |
| Lab | Data Science Lab | How were the models selected and how reliable are they? |
| About | Methodology, data and responsible use | What is this product, how is it built, and what are its limits? |

Desktop navigation uses a persistent left sidebar. On narrow screens the sidebar may
collapse, but page names and the active page must remain visible. The header contains
the data freshness indicator and EN/HU language control; filters stay inside pages.

## Global Application Shell

Every page contains:

- product name and compact wordmark;
- active season/week context;
- latest successful refresh time and odds snapshot time;
- data-state badge: `LIVE DATA`, `PRESEASON/FALLBACK`, `STALE` or `UNAVAILABLE`;
- EN/HU language selector;
- methodology and responsible-use links in the footer;
- a clear non-affiliation notice.

The application must fail softly. A missing optional table hides only the affected
component. A missing core table produces an explanatory empty state with the expected
refresh command, never a Python stack trace.

## Page Designs

### 1. Weekly Overview

Purpose: give a useful answer within ten seconds without requiring betting knowledge.

Top row:

- week selector;
- games remaining;
- closest matchup;
- largest model favorite;
- highest predicted total.

Main content:

1. featured matchup card;
2. compact schedule cards ordered by kickoff;
3. top model-versus-market differences, explicitly labelled as candidates;
4. expected-wins snapshot from the season simulation;
5. data coverage and fallback notice.

### 2. Game Center

One selected game is the entire page context.

```text
┌─────────────────────────────────────────────────────────────┐
│ Away badge  TEAM     kickoff / venue      TEAM  Home badge  │
│    42.1%                win probability              57.9%   │
├──────────────────────┬──────────────────────┬───────────────┤
│ Implied score        │ Model spread         │ Model total   │
│ AWAY 20.8–24.6 HOME  │ HOME -3.8            │ 45.4          │
├──────────────────────┴──────────────────────┴───────────────┤
│ Why the model leans this way — plain-language narrative     │
├─────────────────────────────────────────────────────────────┤
│ Market comparison: Moneyline | Spread | Totals              │
├─────────────────────────────────────────────────────────────┤
│ Expand: technical drivers, feature contributions, routing   │
└─────────────────────────────────────────────────────────────┘
```

Required states:

- primary-model inputs complete;
- fallback prediction with a visible reason;
- market available;
- market unavailable;
- game already started, therefore no current betting candidate.

### 3. Betting Board

Purpose: decision-ready comparison, not a promise of profit.

Default filters are conservative and user-visible:

- future games only;
- positive EV only;
- minimum bookmaker coverage;
- market selector: Moneyline, Spread, Totals;
- week, team and bookmaker;
- minimum model–market probability gap and EV;
- primary/fallback prediction mode.

The default view is one best candidate card per game and market. An expandable table
shows all lines and books. This prevents multiple slightly different lines from making
one game dominate the page.

Candidate fields:

- matchup and kickoff;
- market/outcome/line;
- best available odds and bookmaker count;
- model probability and no-vig market probability;
- model–market probability gap and expected value;
- prediction mode and freshness;
- later, prospective CLV state.

Use `Candidate`, never `Guaranteed pick`, `Lock` or similar language. Kelly is hidden
from the default public view and may appear only inside a technical expander.

### 4. Season Simulator

Top section:

- expected wins table for all 32 teams;
- P10–P90 range;
- most likely win total;
- selected team control.

Team detail:

- win-distribution chart;
- expected record;
- probability of selected win thresholds;
- dynamic-versus-frozen Elo comparison;
- concise explanation that simulation is a distribution, not a forecast certainty.

The first release does not claim playoff probability until playoff qualification and
tiebreaker rules are explicitly implemented and tested.

### 5. Data Science Lab

This page is technical but curated. It should not expose raw database tables.

Sections:

1. model registry and selected production versions;
2. chronological validation and holdout results;
3. probability calibration, Brier score and log loss;
4. Spread/Totals MAE and RMSE;
5. feature contribution examples;
6. external nfelo comparison;
7. leakage controls, routing and data lineage.

Charts always include sample size, evaluation period, metric direction and a short
plain-language interpretation.

### 6. About

Contains:

- project purpose and portfolio context;
- methodology summary;
- data-source attribution;
- update cadence;
- limitations and responsible betting statement;
- privacy and product-analytics disclosure;
- explicit statement that the product is not affiliated with or endorsed by the NFL
  or its clubs.

## Visual System

The visual target is a premium analytics command center: dense enough for serious
analysis, but with a strong hierarchy and broadcast-quality presentation. The supplied
dark Market Board concept is the reference direction. The application must not look
like an unstyled stack of default Streamlit widgets.

### Reference layout language

- fixed dark navigation rail with icon, label and high-contrast active state;
- large page title paired with compact KPI tiles;
- deep navy data surfaces separated by subtle blue borders;
- dense, readable table as the central workspace;
- context panel on the right for the selected game;
- supporting charts in a consistent card grid;
- green, blue, amber and red semantic highlights against a restrained background;
- compact status pills such as `VALUE`, `WATCH`, `NO EDGE`, `FALLBACK` and `STALE`;
- small info icons and tooltips for technical terms;
- generous spacing around page regions even when individual tables are dense.

The sample concept is a design-language reference rather than a literal single-page
specification. Model diagnostics such as calibration and Brier score belong mainly in
the Data Science Lab; the Betting Board keeps only the information needed to evaluate
current candidates. This preserves the premium appearance without overloading every
screen.

### Core component library

The first UI block must create reusable, styled components rather than page-specific
markup:

- `AppHeader`: title, season/week, refresh and snapshot freshness;
- `MetricTile`: icon, label, main value, optional trend and tooltip;
- `StatusPill`: semantic state with text, color and icon;
- `TeamIdentity`: badge/helmet/logo adapter plus name and abbreviation;
- `MatchupHeader`: two-team presentation and kickoff context;
- `ProbabilityBar`: opposing colors, midpoint and accessible labels;
- `PredictionTile`: probability, Spread, Total or implied score;
- `FactorBar`: signed contribution with neutral zero baseline;
- `CandidateRow` and `CandidateCard`: desktop and mobile variants;
- `ChartCard`: title, help text, freshness and consistent Plotly styling;
- `EmptyState` and `DataWarning`: missing, stale or fallback data;
- `ResponsibleUsePanel`: persistent compact disclaimer.

### Visual tokens

Initial tokens should be centralized and adjustable before individual pages are built:

| Token | Intended use |
|---|---|
| `surface.canvas` | near-black navy application background |
| `surface.sidebar` | slightly lighter navigation rail |
| `surface.card` | blue-black card and table surface |
| `border.subtle` | low-contrast blue-gray separation |
| `text.primary` | off-white headings and key values |
| `text.secondary` | cool gray labels and supporting text |
| `accent.model` | electric blue model probability and charts |
| `accent.positive` | mint green positive edge and healthy state |
| `accent.warning` | amber watch, fallback and stale state |
| `accent.negative` | coral red negative edge and error state |

Exact hex values are selected during the UI foundation build and verified for
contrast. They must live in one theme/config module rather than being repeated across
pages.

### Streamlit implementation quality

The premium result should be achieved with a controlled theme, reusable component
renderers, Plotly templates and narrowly scoped CSS. Avoid fragile selectors tied to
Streamlit's generated internal class names. Default widgets may remain for accessible
inputs, but their placement, labels and surrounding components follow the design
system. The page should use the full viewport width on desktop and intentionally
collapse to a card-first layout on mobile.

The visual style should feel like a modern broadcast analytics desk rather than a
sportsbook.

- deep navy background or header;
- off-white content surfaces;
- muted slate secondary text;
- cyan/blue for model information;
- amber for attention and fallback states;
- green/red may supplement positive/negative values but never be the only signal;
- tabular numerals for odds, probabilities and model metrics;
- restrained shadows and rounded cards;
- WCAG-conscious contrast and keyboard-accessible controls.

Team colors are accents, not page backgrounds. Opponents must remain equally legible
and neutral presentation must be preserved.

## Team Identity and Helmet Strategy

The UI supports three interchangeable identity modes:

1. `BADGE` — fallback: team abbreviation in a team-color circular badge;
2. `GENERIC_HELMET` — optional: an original, unbranded helmet silhouette colored by
   team palette, with the abbreviation beside it;
3. `NFLVERSE_REMOTE_LOGO` — default: remotely loaded logo URL from nflverse team
   metadata, with visible source attribution and a non-affiliation notice.

The nflverse `teams` release is the selected metadata source for team names, colors
and remote logo URLs. The repository does not copy or redistribute the image files.
The interface visibly attributes nflverse and states that team marks belong to their
respective owners and that this project is not affiliated with the NFL or its clubs.

Implementation preparation:

- one team-brand registry keyed by the canonical project codes;
- historical aliases normalized before lookup (`OAK→LV`, `SD→LAC`, `STL→LA`);
- local fallback asset for every team;
- logo provider isolated behind one function;
- broken remote image URLs must never break a page;
- asset provenance, retrieval date and display authorization recorded in a manifest;
- no implication of NFL or club endorsement.

## Data Contracts by Page

| Page | Required analytics products |
|---|---|
| Weekly Overview | current predictions, score predictions, season simulation summary, refresh audit |
| Game Center | probability, Spread, Totals, implied score, narratives, explanations, feature contributions, market board |
| Betting Board | current betting board, forward archive, prospective CLV view |
| Season Simulator | season simulation summary and win distribution, Elo benchmark comparison |
| Data Science Lab | governance scorecards, holdout/backtest summaries and prediction data-science view |
| About | static documentation and refresh metadata |

UI data access should be centralized in a read-only repository layer. Pages must not
embed ad-hoc SQL or modify DuckDB.

## Forward-Only and Freshness Rules

- Current public candidates require `commence_time > fetched_at` and current time
  before kickoff.
- Historical archive rows are used for monitoring and CLV, not rendered as historical
  picks unless a future audited-results product is deliberately approved.
- The latest successful refresh controls the global freshness badge.
- Old odds are visibly stale rather than silently presented as current.
- Fallback predictions remain visible but clearly labelled.

## Product Analytics Plan

Privacy-conscious analytics is added only after the page shell is stable.

Measure:

- page views and navigation path;
- Game Center matchup selection;
- market and filter usage;
- methodology and disclaimer expansion;
- simulator team selection;
- device class and coarse traffic source.

Do not send game-level model inputs, selected bookmaker, exact candidate identifiers,
IP-derived fine location, or free-form user content as custom analytics parameters.

## Responsive Behaviour

- desktop: sidebar plus two- or three-column cards;
- tablet: two-column cards;
- mobile: single-column cards, collapsed filters and no horizontally essential table;
- wide technical tables receive a card summary before the expandable full table;
- charts use readable labels at 360-pixel width.

## Delivery Sequence

### UI-1 — Foundation

- application shell and navigation;
- theme tokens and reusable components;
- read-only DuckDB repository;
- team badge registry and asset adapter;
- freshness, missing-data and error states.

### UI-2 — Weekly Overview and Betting Board

- forward-only query layer;
- filters, candidate cards and detailed table;
- responsible-use presentation;
- desktop and mobile smoke tests.

### UI-3 — Game Center

- matchup selector;
- probability, Spread, Totals and implied scores;
- narrative and technical expanders;
- market comparison.

### UI-4 — Season Simulator

- expected-wins standings;
- team distribution view;
- dynamic/frozen comparison.

### UI-5 — Data Science Lab and About

- curated governance views;
- methodology, lineage, attribution and limitations.

### UI-6 — Product Analytics and Deployment

- consent/privacy decision;
- Google Analytics events;
- hosting configuration;
- refresh scheduling and production smoke checks.

## First-Release Acceptance Criteria

- six-page navigation works on desktop and mobile;
- every public betting row is pregame and from the current forward workflow;
- all timestamps identify their timezone and data freshness;
- fallback modes are never visually indistinguishable from primary modes;
- one broken optional asset cannot break a page;
- no generated DuckDB file is bundled in Git;
- empty database and missing-table states are understandable;
- the application contains methodology, privacy, non-affiliation and responsible-use
  disclosures;
- automated tests cover query filtering and view-model calculations independently of
  Streamlit rendering.

## Decisions Before Implementation

The baseline recommendation is:

- six pages with four primary product destinations;
- English-first UI with EN/HU narratives;
- dark broadcast-analytics visual style;
- team badges or original generic helmets at launch;
- official team logos only after an explicit rights/usage decision;
- Google Analytics after the application shell, not before it;
- no public historical picks in the first release.
