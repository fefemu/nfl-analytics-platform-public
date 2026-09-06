import assert from "node:assert/strict";
import test from "node:test";

import {
  budapestClock,
  dispatchGitHub,
  runScheduledTasks,
  shouldDispatch,
  upcomingKickoffWindow,
} from "../src/index.js";

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
    status: 200,
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
    status: 200,
    json: async () => [{ id: "a", commence_time: "2026-09-13T17:00:00Z" }],
  });
  assert.equal(await upcomingKickoffWindow(
    { ODDS_API_KEY: "secret" }, new Date("2026-09-13T15:45:00Z"), fakeFetch,
  ), null);
});

test("production dispatch completes even when kickoff odds lookup fails", async () => {
  const calls = [];
  const fakeFetch = async (url) => {
    calls.push(url);
    if (url.startsWith("https://api.github.com/")) {
      await new Promise((resolve) => setTimeout(resolve, 5));
      calls.push("github-completed");
      return { ok: true, status: 204, text: async () => "" };
    }
    return {
      ok: false,
      status: 401,
      text: async () => '{"error_code":"INVALID_KEY"}',
    };
  };

  await assert.rejects(
    runScheduledTasks(
      {
        GITHUB_TOKEN: "github-secret",
        GITHUB_OWNER: "fefemu",
        GITHUB_REPOSITORY: "nfl-analytics-platform",
        ODDS_API_KEY: "invalid-odds-secret",
      },
      new Date("2026-09-06T07:00:00Z"),
      fakeFetch,
    ),
    /scheduled dispatcher tasks failed/,
  );
  assert.equal(calls.filter((url) => url.startsWith?.("https://api.github.com/")).length, 1);
  assert.ok(calls.includes("github-completed"));
});

test("scheduled tasks log each HTTP status and the rejected task reason", async () => {
  const logs = [];
  const errors = [];
  const originalLog = console.log;
  const originalError = console.error;
  console.log = (message) => logs.push(message);
  console.error = (message) => errors.push(message);

  try {
    const fakeFetch = async (url) => {
      if (url.startsWith("https://api.github.com/")) {
        return { ok: true, status: 204, text: async () => "" };
      }
      return {
        ok: false,
        status: 429,
        text: async () => '{"message":"quota exceeded"}',
      };
    };

    await assert.rejects(
      runScheduledTasks(
        {
          GITHUB_TOKEN: "github-secret",
          GITHUB_OWNER: "fefemu",
          GITHUB_REPOSITORY: "nfl-analytics-platform",
          ODDS_API_KEY: "odds-secret",
        },
        new Date("2026-09-06T07:00:00Z"),
        fakeFetch,
      ),
      (error) => {
        assert.equal(error instanceof AggregateError, true);
        assert.match(error.errors[0].message, /kickoff odds lookup failed: Odds API events lookup failed \(HTTP 429\)/);
        return true;
      },
    );
  } finally {
    console.log = originalLog;
    console.error = originalError;
  }

  assert.ok(logs.includes("GitHub repository_dispatch: HTTP 204"));
  assert.ok(logs.includes("Odds API events lookup: HTTP 429"));
  assert.ok(errors.includes('Odds API events lookup response body: {"message":"quota exceeded"}'));
  assert.ok(errors.some((message) => message.includes("Scheduled task rejected: kickoff odds lookup:")));
});
