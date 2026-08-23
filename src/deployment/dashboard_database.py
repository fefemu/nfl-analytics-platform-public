"""Resolve or download the DuckDB artifact used by the public dashboard."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

DATABASE_PATH_ENV = "NFL_ANALYTICS_DASHBOARD_DATABASE"
DATABASE_URL_ENV = "NFL_ANALYTICS_DASHBOARD_DATABASE_URL"
GITHUB_REPOSITORY_ENV = "NFL_ANALYTICS_DASHBOARD_GITHUB_REPOSITORY"
GITHUB_TOKEN_ENV = "NFL_ANALYTICS_DASHBOARD_GITHUB_TOKEN"
GITHUB_ASSET_ENV = "NFL_ANALYTICS_DASHBOARD_GITHUB_ASSET"
DEFAULT_GITHUB_ASSET = "dashboard.duckdb"
GITHUB_API_ROOT = "https://api.github.com"
MAX_DATABASE_BYTES = 250 * 1024 * 1024
GITHUB_REQUEST_ATTEMPTS = 4
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATABASE_FILE = PROJECT_ROOT / "data" / "nfl_analytics.duckdb"
LOGGER = logging.getLogger(__name__)


def _request_headers(
    token: str | None = None,
    *,
    accept: str = "application/vnd.github+json",
) -> dict[str, str]:
    headers = {
        "Accept": accept,
        "User-Agent": "NFL-Analytics-Platform/1.0",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token and token.strip():
        headers["Authorization"] = f"Bearer {token.strip()}"
    return headers


def _github_error_message(error: HTTPError) -> str:
    """Return a safe, actionable description of a GitHub API failure."""

    try:
        payload = json.loads(error.read().decode("utf-8", errors="replace"))
        github_message = str(payload.get("message", "")).strip()
    except (json.JSONDecodeError, OSError, ValueError):
        github_message = ""

    rate_remaining = error.headers.get("X-RateLimit-Remaining")
    rate_reset = error.headers.get("X-RateLimit-Reset")
    details = [f"HTTP {error.code}"]
    if github_message:
        details.append(github_message)
    if rate_remaining is not None:
        details.append(f"rate-limit remaining={rate_remaining}")
    if rate_reset:
        details.append(f"reset={rate_reset}")
    return "; ".join(details)


def _is_retryable_github_error(error: HTTPError, description: str) -> bool:
    """Return whether a GitHub HTTP failure is likely to be transient."""

    lowered = description.lower()
    rate_limited = error.headers.get("X-RateLimit-Remaining") == "0"
    return (
        error.code in {408, 429}
        or 500 <= error.code < 600
        or (
            error.code == 403
            and (
                rate_limited
                or "rate limit" in lowered
                or "secondary rate" in lowered
                or "abuse" in lowered
            )
        )
    )


def _open_github_request(request: Request, *, timeout: int):
    """Open a GitHub request with bounded retries and useful diagnostics."""

    last_error: HTTPError | URLError | None = None
    final_attempt = 1
    for attempt in range(1, GITHUB_REQUEST_ATTEMPTS + 1):
        final_attempt = attempt
        try:
            return urlopen(request, timeout=timeout)
        except HTTPError as error:
            last_error = error
            description = _github_error_message(error)
            retryable = _is_retryable_github_error(error, description)
        except URLError as error:
            last_error = error
            description = str(error.reason)
            retryable = True

        if not retryable or attempt == GITHUB_REQUEST_ATTEMPTS:
            break
        delay = float(2 ** (attempt - 1))
        LOGGER.warning(
            "GitHub dashboard request attempt %s/%s failed: %s. "
            "Retrying in %.1f seconds.",
            attempt,
            GITHUB_REQUEST_ATTEMPTS,
            description,
            delay,
        )
        time.sleep(delay)

    if isinstance(last_error, HTTPError):
        raise RuntimeError(
            "GitHub dashboard artifact request failed after "
            f"{final_attempt} attempt(s): {description}. "
            "For a private repository, verify that the configured fine-grained "
            "token can access the selected repository with Contents: read-only."
        ) from last_error
    raise RuntimeError(
        "GitHub dashboard artifact request failed after "
        f"{final_attempt} attempt(s): {description}."
    ) from last_error


def _latest_github_asset_url(
    repository: str,
    token: str,
    asset_name: str,
) -> str:
    """Return the API URL of one asset from the latest private release."""

    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise ValueError("GitHub repository must use the 'owner/name' format.")
    if not token.strip():
        raise RuntimeError("Private GitHub dashboard access requires a token.")

    request = Request(
        f"{GITHUB_API_ROOT}/repos/{repository}/releases/latest",
        headers=_request_headers(token),
    )
    with _open_github_request(request, timeout=30) as response:
        release = json.load(response)

    matches = [
        asset
        for asset in release.get("assets", [])
        if asset.get("name") == asset_name
    ]
    if len(matches) != 1 or not matches[0].get("url"):
        raise RuntimeError(
            f"Latest GitHub release must contain exactly one {asset_name!r} asset."
        )
    return str(matches[0]["url"])


def _download_database(
    url: str,
    target: Path,
    *,
    token: str | None = None,
) -> Path:
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError("Dashboard database URL must be a valid HTTPS URL.")

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.downloading")
    if temporary.exists():
        temporary.unlink()
    request = Request(
        url,
        headers=_request_headers(
            token,
            accept="application/octet-stream" if token else "*/*",
        ),
    )
    downloaded = 0
    try:
        open_request = (
            _open_github_request(request, timeout=60)
            if token
            else urlopen(request, timeout=60)
        )
        with open_request as response, temporary.open("wb") as output:
            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > MAX_DATABASE_BYTES:
                raise RuntimeError("Dashboard database artifact exceeds 250 MB.")
            while chunk := response.read(1024 * 1024):
                downloaded += len(chunk)
                if downloaded > MAX_DATABASE_BYTES:
                    raise RuntimeError("Dashboard database artifact exceeds 250 MB.")
                output.write(chunk)
        if downloaded == 0:
            raise RuntimeError("Downloaded dashboard database artifact is empty.")
        os.replace(temporary, target)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    return target


def resolve_dashboard_database_file() -> Path:
    """Resolve local development DB or cache the configured public artifact."""

    configured_path = os.getenv(DATABASE_PATH_ENV)
    if configured_path:
        return Path(configured_path).expanduser().resolve()

    github_repository = os.getenv(GITHUB_REPOSITORY_ENV)
    if github_repository:
        github_token = os.getenv(GITHUB_TOKEN_ENV, "")
        github_asset = os.getenv(GITHUB_ASSET_ENV, DEFAULT_GITHUB_ASSET)
        asset_url = _latest_github_asset_url(
            github_repository,
            github_token,
            github_asset,
        )
        url_key = hashlib.sha256(asset_url.encode("utf-8")).hexdigest()[:12]
        cache_file = (
            Path(tempfile.gettempdir())
            / "nfl-analytics-platform"
            / f"dashboard-{url_key}.duckdb"
        )
        if cache_file.is_file() and cache_file.stat().st_size > 0:
            return cache_file
        return _download_database(asset_url, cache_file, token=github_token)

    configured_url = os.getenv(DATABASE_URL_ENV)
    if not configured_url:
        return DATABASE_FILE

    url_key = hashlib.sha256(configured_url.encode("utf-8")).hexdigest()[:12]
    cache_file = (
        Path(tempfile.gettempdir())
        / "nfl-analytics-platform"
        / f"dashboard-{url_key}.duckdb"
    )
    if cache_file.is_file() and cache_file.stat().st_size > 0:
        return cache_file
    return _download_database(configured_url, cache_file)
