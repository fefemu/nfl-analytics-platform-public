"""Resolve or download the DuckDB artifact used by the public dashboard."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
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
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATABASE_FILE = PROJECT_ROOT / "data" / "nfl_analytics.duckdb"


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
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


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
    with urlopen(request, timeout=30) as response:
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
        with urlopen(request, timeout=60) as response, temporary.open("wb") as output:
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
