"""Resolve or download the DuckDB artifact used by the public dashboard."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

DATABASE_PATH_ENV = "NFL_ANALYTICS_DASHBOARD_DATABASE"
DATABASE_URL_ENV = "NFL_ANALYTICS_DASHBOARD_DATABASE_URL"
MAX_DATABASE_BYTES = 250 * 1024 * 1024
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATABASE_FILE = PROJECT_ROOT / "data" / "nfl_analytics.duckdb"


def _download_database(url: str, target: Path) -> Path:
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError("Dashboard database URL must be a valid HTTPS URL.")

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.downloading")
    if temporary.exists():
        temporary.unlink()
    request = Request(url, headers={"User-Agent": "NFL-Analytics-Platform/1.0"})
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
