# Cloudflare production-refresh dispatcher

The Cloudflare Worker provides a scheduler independent of GitHub Actions' native
schedule delivery. It dispatches the existing production workflow at these local
times:

- Tuesday 08:00 Europe/Budapest
- Thursday 15:00 Europe/Budapest
- Sunday 09:00 Europe/Budapest

The Worker wakes every 15 minutes and dispatches only during the first 15 minutes
of a configured local-time window. `Intl.DateTimeFormat` performs the Budapest
DST conversion. GitHub's native schedules remain enabled as delayed fallback
attempts. The production refresh guard treats both sources as scheduled events
and suppresses another refresh when a successful release exists within 12 hours.
Manual `workflow_dispatch` runs always bypass the guard.

The heartbeat also checks the Odds API upcoming-events endpoint without consuming
market quota. When NFL games are 52.5–67.5 minutes from kickoff, it sends one
`kickoff-odds-capture` dispatch for that kickoff window. The separate workflow
downloads one full NFL slate for the US region and three markets (3 credits),
rebuilds only odds/market/archive layers and republishes the validated state. It
does not retrain models or refresh football-data sources.

## One-time setup

1. Create a fine-grained GitHub personal access token restricted to the
   `fefemu/nfl-analytics-platform` repository with **Contents: Read and write**.
   This is required by GitHub's repository-dispatch endpoint.
2. Open a terminal in `deployment/cloudflare-refresh-dispatcher`.
3. Run `npm install` and `npx wrangler login`.
4. Store the token with `npx wrangler secret put GITHUB_TOKEN`.
5. Store the Odds API key with `npx wrangler secret put ODDS_API_KEY`.
6. Run `npm test`.
7. Deploy with `npm run deploy`.

Never add either secret to `wrangler.jsonc`, `.env`, source code, logs or the
repository. Wrangler stores production values as encrypted Worker secrets.

## Verification

After deployment, use Cloudflare **Workers & Pages → the Worker → Triggers** to
confirm the `*/15 * * * *` Cron Trigger. Use **Observability → Logs** to verify
dispatches. In GitHub Actions, externally triggered runs show the
`repository_dispatch` event. A successful API request returns HTTP 204.
Every scheduled execution logs `Odds API events lookup: HTTP <status>`. During
a configured production window it also logs
`GitHub repository_dispatch: HTTP <status>`. Non-success responses include the
response body and rejected task name before the aggregate scheduled error.

Kickoff dispatches appear under **Kickoff-near odds capture**. A comparison is
marked closing/CLV only when its `kickoff_capture_*` fetch is 45–75 minutes before
kickoff.

For local schedule simulation, run `npm run dev` and invoke Wrangler's scheduled
test route with a chosen `time` query parameter. Do not use a production token in
local `.dev.vars` unless it is excluded from source control.
