const BUDAPEST_TIME_ZONE = "Europe/Budapest";
const EVENT_TYPE = "scheduled-production-refresh";
const KICKOFF_EVENT_TYPE = "kickoff-odds-capture";
const NFL_EVENTS_URL = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/events";
const CAPTURE_MIN_MINUTES = 52.5;
const CAPTURE_MAX_MINUTES = 67.5;
const TARGET_HOURS = new Map([
  [2, 8],
  [4, 15],
  [0, 9],
]);

const WEEKDAYS = new Map([
  ["Sun", 0], ["Mon", 1], ["Tue", 2], ["Wed", 3],
  ["Thu", 4], ["Fri", 5], ["Sat", 6],
]);

export function budapestClock(date) {
  const parts = new Intl.DateTimeFormat("en-GB", {
    timeZone: BUDAPEST_TIME_ZONE,
    weekday: "short",
    hour: "2-digit",
    minute: "2-digit",
    hourCycle: "h23",
  }).formatToParts(date);
  const values = Object.fromEntries(parts.map(({ type, value }) => [type, value]));
  return {
    weekday: WEEKDAYS.get(values.weekday),
    hour: Number(values.hour),
    minute: Number(values.minute),
  };
}

export function shouldDispatch(date) {
  const local = budapestClock(date);
  return TARGET_HOURS.get(local.weekday) === local.hour && local.minute < 15;
}

export async function dispatchGitHub(env, scheduledAt, fetchImpl = fetch, eventType = EVENT_TYPE, extraPayload = {}) {
  for (const name of ["GITHUB_TOKEN", "GITHUB_OWNER", "GITHUB_REPOSITORY"]) {
    if (!env[name]) throw new Error(`Missing required binding: ${name}`);
  }
  let response;
  try {
    response = await fetchImpl(
      `https://api.github.com/repos/${env.GITHUB_OWNER}/${env.GITHUB_REPOSITORY}/dispatches`,
      {
      method: "POST",
      headers: {
        Accept: "application/vnd.github+json",
        Authorization: `Bearer ${env.GITHUB_TOKEN}`,
        "Content-Type": "application/json",
        "User-Agent": "NFL-Analytics-Refresh-Dispatcher/1.0",
        "X-GitHub-Api-Version": "2026-03-10",
      },
      body: JSON.stringify({
        event_type: eventType,
        client_payload: {
          source: "cloudflare-cron",
          scheduled_at: scheduledAt.toISOString(),
          timezone: BUDAPEST_TIME_ZONE,
          ...extraPayload,
        },
      }),
      },
    );
  } catch (error) {
    console.error(`GitHub repository_dispatch: request failed: ${error instanceof Error ? error.message : String(error)}`);
    throw error;
  }
  console.log(`GitHub repository_dispatch: HTTP ${response.status}`);
  if (!response.ok) {
    const detail = await response.text();
    console.error(`GitHub repository_dispatch response body: ${detail || "<empty>"}`);
    throw new Error(`GitHub repository_dispatch failed (HTTP ${response.status}): ${detail}`);
  }
}

export async function upcomingKickoffWindow(env, scheduledAt, fetchImpl = fetch) {
  if (!env.ODDS_API_KEY) throw new Error("Missing required binding: ODDS_API_KEY");
  const url = new URL(NFL_EVENTS_URL);
  url.searchParams.set("apiKey", env.ODDS_API_KEY);
  url.searchParams.set("dateFormat", "iso");
  let response;
  try {
    response = await fetchImpl(url.toString(), { headers: { "User-Agent": "NFL-Analytics-Kickoff-Scheduler/1.0" } });
  } catch (error) {
    console.error(`Odds API events lookup: request failed: ${error instanceof Error ? error.message : String(error)}`);
    throw error;
  }
  console.log(`Odds API events lookup: HTTP ${response.status}`);
  if (!response.ok) {
    const detail = await response.text();
    console.error(`Odds API events lookup response body: ${detail || "<empty>"}`);
    throw new Error(`Odds API events lookup failed (HTTP ${response.status}): ${detail}`);
  }
  const events = await response.json();
  const due = events.filter((event) => {
    const kickoff = new Date(event.commence_time);
    const minutes = (kickoff.getTime() - scheduledAt.getTime()) / 60000;
    return Number.isFinite(minutes) && minutes >= CAPTURE_MIN_MINUTES && minutes < CAPTURE_MAX_MINUTES;
  });
  if (!due.length) return null;
  return {
    event_ids: due.map((event) => event.id).sort(),
    kickoff_times: [...new Set(due.map((event) => new Date(event.commence_time).toISOString()))].sort(),
    target_minutes_before_kickoff: 60,
  };
}

export async function runScheduledTasks(env, scheduledAt, fetchImpl = fetch) {
  const tasks = [];
  if (shouldDispatch(scheduledAt)) {
    tasks.push({
      name: "production refresh dispatch",
      promise: dispatchGitHub(env, scheduledAt, fetchImpl),
    });
  }
  tasks.push({
    name: "kickoff odds lookup",
    promise: (async () => {
      const kickoff = await upcomingKickoffWindow(env, scheduledAt, fetchImpl);
      if (kickoff) {
        await dispatchGitHub(env, scheduledAt, fetchImpl, KICKOFF_EVENT_TYPE, kickoff);
      } else {
        console.log(`No kickoff capture window at ${scheduledAt.toISOString()}.`);
      }
    })(),
  });

  const results = await Promise.allSettled(tasks.map(({ promise }) => promise));
  const failures = results.flatMap((result, index) => {
    if (result.status !== "rejected") return [];
    const reason = result.reason instanceof Error ? result.reason : new Error(String(result.reason));
    console.error(`Scheduled task rejected: ${tasks[index].name}: ${reason.stack || reason.message}`);
    return [new Error(`${tasks[index].name} failed: ${reason.message}`, { cause: reason })];
  });
  if (failures.length) {
    throw new AggregateError(failures, "One or more scheduled dispatcher tasks failed.");
  }
}

export default {
  async scheduled(controller, env, ctx) {
    const scheduledAt = new Date(controller.scheduledTime);
    ctx.waitUntil(runScheduledTasks(env, scheduledAt));
  },
};
