"""Tests for public dashboard database resolution and download caching."""

import json
from io import BytesIO
from pathlib import Path

import pytest

from src.deployment import dashboard_database


class FakeResponse(BytesIO):
    def __init__(self, payload: bytes) -> None:
        super().__init__(payload)
        self.headers: dict[str, str] = {
            "Content-Length": str(len(payload))
        }

    def __enter__(self):
        return self

    def __exit__(self, *args) -> None:
        self.close()


def clear_database_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(dashboard_database.DATABASE_PATH_ENV, raising=False)
    monkeypatch.delenv(dashboard_database.DATABASE_URL_ENV, raising=False)
    monkeypatch.delenv(dashboard_database.GITHUB_REPOSITORY_ENV, raising=False)
    monkeypatch.delenv(dashboard_database.GITHUB_TOKEN_ENV, raising=False)
    monkeypatch.delenv(dashboard_database.GITHUB_ASSET_ENV, raising=False)


def test_local_database_is_default(monkeypatch: pytest.MonkeyPatch) -> None:
    clear_database_environment(monkeypatch)

    assert (
        dashboard_database.resolve_dashboard_database_file()
        == dashboard_database.DATABASE_FILE
    )


def test_explicit_database_path_takes_priority(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    configured = tmp_path / "custom.duckdb"
    monkeypatch.setenv(
        dashboard_database.DATABASE_PATH_ENV,
        str(configured),
    )
    monkeypatch.setenv(
        dashboard_database.DATABASE_URL_ENV,
        "https://example.com/ignored.duckdb",
    )

    assert dashboard_database.resolve_dashboard_database_file() == configured


def test_https_artifact_is_downloaded_and_cached(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    clear_database_environment(monkeypatch)
    monkeypatch.setenv(
        dashboard_database.DATABASE_URL_ENV,
        "https://example.com/dashboard-v1.duckdb",
    )
    monkeypatch.setattr(dashboard_database.tempfile, "gettempdir", lambda: str(tmp_path))
    calls = []

    def fake_urlopen(request, timeout):
        calls.append((request.full_url, timeout, dict(request.header_items())))
        return FakeResponse(b"duckdb-artifact")

    monkeypatch.setattr(dashboard_database, "urlopen", fake_urlopen)

    first = dashboard_database.resolve_dashboard_database_file()
    second = dashboard_database.resolve_dashboard_database_file()

    assert first == second
    assert first.read_bytes() == b"duckdb-artifact"
    assert calls[0][0:2] == ("https://example.com/dashboard-v1.duckdb", 60)
    assert "Authorization" not in calls[0][2]


def test_private_github_release_asset_is_downloaded(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    clear_database_environment(monkeypatch)
    monkeypatch.setenv(
        dashboard_database.GITHUB_REPOSITORY_ENV,
        "fefemu/nfl-analytics-platform-data",
    )
    monkeypatch.setenv(dashboard_database.GITHUB_TOKEN_ENV, "secret-token")
    monkeypatch.setattr(dashboard_database.tempfile, "gettempdir", lambda: str(tmp_path))
    calls = []

    def fake_urlopen(request, timeout):
        calls.append((request.full_url, timeout, dict(request.header_items())))
        if request.full_url.endswith("/releases/latest"):
            return FakeResponse(
                json.dumps(
                    {
                        "assets": [
                            {
                                "name": "dashboard.duckdb",
                                "url": "https://api.github.com/repos/fefemu/data/releases/assets/42",
                            }
                        ]
                    }
                ).encode("utf-8")
            )
        return FakeResponse(b"private-duckdb")

    monkeypatch.setattr(dashboard_database, "urlopen", fake_urlopen)

    result = dashboard_database.resolve_dashboard_database_file()

    assert result.read_bytes() == b"private-duckdb"
    assert len(calls) == 2
    assert calls[0][1] == 30
    assert calls[1][1] == 60
    assert calls[0][2]["Authorization"] == "Bearer secret-token"
    assert calls[1][2]["Authorization"] == "Bearer secret-token"
    assert calls[1][2]["Accept"] == "application/octet-stream"


def test_private_github_release_requires_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_database_environment(monkeypatch)
    monkeypatch.setenv(
        dashboard_database.GITHUB_REPOSITORY_ENV,
        "fefemu/nfl-analytics-platform-data",
    )

    with pytest.raises(RuntimeError, match="requires a token"):
        dashboard_database.resolve_dashboard_database_file()


def test_private_github_release_rejects_missing_asset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_database_environment(monkeypatch)
    monkeypatch.setattr(
        dashboard_database,
        "urlopen",
        lambda request, timeout: FakeResponse(b'{"assets": []}'),
    )

    with pytest.raises(RuntimeError, match="exactly one"):
        dashboard_database._latest_github_asset_url(
            "fefemu/nfl-analytics-platform-data",
            "secret-token",
            "dashboard.duckdb",
        )


def test_non_https_url_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    clear_database_environment(monkeypatch)
    monkeypatch.setenv(
        dashboard_database.DATABASE_URL_ENV,
        "http://example.com/dashboard.duckdb",
    )
    monkeypatch.setattr(dashboard_database.tempfile, "gettempdir", lambda: str(tmp_path))

    with pytest.raises(ValueError, match="HTTPS"):
        dashboard_database.resolve_dashboard_database_file()


def test_empty_download_is_not_cached(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    clear_database_environment(monkeypatch)
    monkeypatch.setenv(
        dashboard_database.DATABASE_URL_ENV,
        "https://example.com/empty.duckdb",
    )
    monkeypatch.setattr(dashboard_database.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(
        dashboard_database,
        "urlopen",
        lambda request, timeout: FakeResponse(b""),
    )

    with pytest.raises(RuntimeError, match="empty"):
        dashboard_database.resolve_dashboard_database_file()

    assert not list(tmp_path.rglob("*.downloading"))
