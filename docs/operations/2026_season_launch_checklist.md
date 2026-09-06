# 2026 Season Launch Checklist

**Window:** 2026-09-06 through the first completed regular-season game  
**Timezone:** Europe/Budapest  
**Change policy:** Keep production models frozen; allow only operational or correctness fixes.

## Automated and offline release gate — completed 2026-09-06

- private regression suite passes;
- public regression suite passes without private deployment modules;
- latest private operational release restores into an isolated path;
- the 24-table dashboard snapshot builds from the restored release;
- forward archive and forward performance SQL quality gates pass;
- offline odds processing keeps every stage on the explicitly supplied database;
- the default local database hash remains unchanged during the isolation test;
- no Odds API market request or release publication is used by the offline gate.

## Tuesday 2026-09-08 — production dispatcher gate

At the first Cloudflare invocation between 08:00 and 08:14 Budapest time, verify:

1. Cloudflare cron execution is successful.
2. Worker log contains `GitHub repository_dispatch: HTTP 204`.
3. Worker log contains `Odds API events lookup: HTTP 200`.
4. GitHub creates a `repository_dispatch` run for **Weekly production refresh**.
5. The refresh guard allows exactly one production refresh and suppresses redundant attempts.
6. The workflow restores operational state, completes validation and publishes a new release.
7. The public dashboard reports the new refresh timestamp and remains functional on desktop and mobile.

If the dispatch fails, use the per-task Worker status and response body to identify
the failed branch. A manual `workflow_dispatch` run is the recovery path; it
bypasses the scheduled freshness guard.

## Thursday 2026-09-10 — first kickoff-near capture

The first game is scheduled for 02:20 CEST. The expected capture invocation is
01:15 CEST, 65 minutes before kickoff.

Verify:

1. `Odds API events lookup: HTTP 200`;
2. kickoff dispatch returns HTTP 204;
3. **Kickoff-near odds capture** completes successfully;
4. the stored fetch is 45–75 minutes before kickoff;
5. `is_closing_snapshot = true` and `is_clv = true` only for eligible observations;
6. repeated heartbeat invocations do not create a conflicting locked observation.

## First postgame refresh

- the completed game settles to WIN, LOSS or PUSH from its final score;
- flat-stake units and ROI use the immutable entry price;
- the Live Results navigation appears only after at least one settled row exists;
- historical backtests remain separate from the 2026 forward sample;
- any delayed result feed leaves the observation pending rather than fabricating a result.
