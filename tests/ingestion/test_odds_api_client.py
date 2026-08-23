"""Tests for The Odds API client."""

from typing import Any

import pytest
import requests

from src.ingestion.odds_api_client import (
    REQUEST_TIMEOUT_SECONDS,
    fetch_current_nfl_odds,
    parse_optional_integer,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("500", 500),
        (None, None),
        ("invalid", None),
    ],
)
def test_parse_optional_integer(
    value: str | None,
    expected: int | None,
) -> None:
    """Parse valid, missing and invalid quota headers."""

    assert parse_optional_integer(value) == expected


def test_fetch_current_nfl_odds_returns_events_and_quota(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return NFL events and API quota information."""

    captured_request: dict[str, Any] = {}

    class FakeResponse:
        headers = {
            "x-requests-remaining": "499",
            "x-requests-used": "1",
            "x-requests-last": "1",
        }

        def raise_for_status(self) -> None:
            return None

        def json(self) -> list[dict[str, str]]:
            return [{"id": "test-event"}]

    def fake_get(
        endpoint: str,
        params: dict[str, str],
        timeout: int,
    ) -> FakeResponse:
        captured_request["endpoint"] = endpoint
        captured_request["params"] = params
        captured_request["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setenv(
        "ODDS_API_KEY",
        "test-api-key",
    )
    monkeypatch.setattr(
        "src.ingestion.odds_api_client.requests.get",
        fake_get,
    )

    result = fetch_current_nfl_odds()

    assert result.events == [{"id": "test-event"}]
    assert result.requests_remaining == 499
    assert result.requests_used == 1
    assert result.requests_last == 1

    assert captured_request["endpoint"].endswith(
        "/sports/americanfootball_nfl/odds"
    )
    assert captured_request["timeout"] == REQUEST_TIMEOUT_SECONDS

    parameters = captured_request["params"]

    assert parameters["apiKey"] == "test-api-key"
    assert parameters["regions"] == "us"
    assert parameters["markets"] == "h2h,spreads,totals"
    assert parameters["oddsFormat"] == "american"


def test_fetch_current_nfl_odds_handles_request_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Convert request errors into a safe application error."""

    def fake_get(
        endpoint: str,
        params: dict[str, str],
        timeout: int,
    ) -> None:
        raise requests.Timeout("Request timed out.")

    monkeypatch.setenv(
        "ODDS_API_KEY",
        "test-api-key",
    )
    monkeypatch.setattr(
        "src.ingestion.odds_api_client.requests.get",
        fake_get,
    )

    with pytest.raises(
        RuntimeError,
        match="The Odds API request failed.",
    ):
        fetch_current_nfl_odds()


def test_fetch_current_nfl_odds_rejects_invalid_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail when the API response is not valid JSON."""

    class FakeResponse:
        headers: dict[str, str] = {}

        def raise_for_status(self) -> None:
            return None

        def json(self) -> None:
            raise ValueError("Invalid JSON.")

    monkeypatch.setenv(
        "ODDS_API_KEY",
        "test-api-key",
    )
    monkeypatch.setattr(
        "src.ingestion.odds_api_client.requests.get",
        lambda *args, **kwargs: FakeResponse(),
    )

    with pytest.raises(
        RuntimeError,
        match="returned invalid JSON",
    ):
        fetch_current_nfl_odds()


def test_fetch_current_nfl_odds_rejects_non_list_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail when the API payload is not a list of events."""

    class FakeResponse:
        headers: dict[str, str] = {}

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, str]:
            return {"message": "Unexpected payload"}

    monkeypatch.setenv(
        "ODDS_API_KEY",
        "test-api-key",
    )
    monkeypatch.setattr(
        "src.ingestion.odds_api_client.requests.get",
        lambda *args, **kwargs: FakeResponse(),
    )

    with pytest.raises(
        RuntimeError,
        match="must contain a list of events",
    ):
        fetch_current_nfl_odds()