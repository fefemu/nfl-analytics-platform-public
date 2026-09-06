import assert from "node:assert/strict";
import test from "node:test";

import { budapestClock, dispatchGitHub, shouldDispatch, upcomingKickoffWindow } from "../src/index.js";

test("Budapest schedule works in summer and winter time", () => {
  assert.equal(shouldDispatch(new Date("2026-09-01T06:00:00Z")), true);
  assert.equal(shouldDispatch(new Date("2026-09-03T13:00:00Z")), true);
  assert.equal(shouldDispatch(new Date("2026-09-06T07:00:00Z")), true);
  assert.equal(shouldDispatch(new Date("2026-01-06T07:00:00Z")), true);
  assert.equal(shouldDispatch(new Date("2026-09-03T13:15:00Z")), false);
  assert.deepEqual(budapestClock(new Date("2026-09-03T13:05:00Z")), {
    weekday: 4,
    hour: 15,
    minute: 5,
  });
});

test("dispatcher sends the expected authenticated repository event", async () => {
  let request;
  const fakeFetch = async (url, options) => {
    request = { url, options };
    return { ok: true, status: 204, text: async () => "" };
  };
  await dispatchGitHub(
    {
      GITHUB_TOKEN: "secret",
      GITHUB_OWNER: "fefemu",
      GITHUB_REPOSITORY: "nfl-analytics-platform",
    },
    new Date("2026-09-03T13:00:00Z"),
    fakeFetch,
  );
  assert.equal(
    request.url,
    "https://api.github.com/repos/fefemu/nfl-analytics-platform/dispatches",
  );
  assert.equal(request.options.method, "POST");
  assert.equal(request.options.headers.Authorization, "Bearer secret");
  const payload = JSON.parse(request.options.body);
  assert.equal(payload.event_type, "scheduled-production-refresh");
  assert.equal(payload.client_payload.source, "cloudflare-cron");
});

test("dispatcher rejects missing secrets before making a request", async () => {
  await assert.rejects(
    dispatchGitHub({}, new Date(), async () => assert.fail("fetch must not run")),
    /Missing required binding: GITHUB_TOKEN/,
  );
});

test("kickoff lookup returns all games in the single sixty-minute window", async () => {
  const fakeFetch = async () => ({
    ok: true,
    json: async () => [
      { id: "b", commence_time: "2026-09-13T17:00:00Z" },
      { id: "a", commence_time: "2026-09-13T17:00:00Z" },
      { id: "later", commence_time: "2026-09-13T20:25:00Z" },
    ],
  });
  const result = await upcomingKickoffWindow(
    { ODDS_API_KEY: "secret" }, new Date("2026-09-13T16:00:00Z"), fakeFetch,
  );
  assert.deepEqual(result.event_ids, ["a", "b"]);
  assert.deepEqual(result.kickoff_times, ["2026-09-13T17:00:00.000Z"]);
});

test("kickoff lookup is empty outside the narrow idempotent window", async () => {
  const fakeFetch = async () => ({
    ok: true,
    json: async () => [{ id: "a", commence_time: "2026-09-13T17:00:00Z" }],
  });
  assert.equal(await upcomingKickoffWindow(
    { ODDS_API_KEY: "secret" }, new Date("2026-09-13T15:45:00Z"), fakeFetch,
  ), null);
});
