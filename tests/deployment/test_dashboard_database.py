"""Tests for public dashboard database resolution and download caching."""

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
        calls.append((request.full_url, timeout))
        return FakeResponse(b"duckdb-artifact")

    monkeypatch.setattr(dashboard_database, "urlopen", fake_urlopen)

    first = dashboard_database.resolve_dashboard_database_file()
    second = dashboard_database.resolve_dashboard_database_file()

    assert first == second
    assert first.read_bytes() == b"duckdb-artifact"
    assert calls == [("https://example.com/dashboard-v1.duckdb", 60)]


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
