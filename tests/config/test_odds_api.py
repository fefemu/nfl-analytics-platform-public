"""Tests for The Odds API configuration."""

import pytest

from src.config.odds_api import get_odds_api_key


def test_get_odds_api_key_returns_configured_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return the API key stored in the environment."""

    monkeypatch.setenv(
        "ODDS_API_KEY",
        "test-api-key",
    )

    assert get_odds_api_key() == "test-api-key"


def test_get_odds_api_key_strips_whitespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Remove accidental whitespace around the API key."""

    monkeypatch.setenv(
        "ODDS_API_KEY",
        "  test-api-key  ",
    )

    assert get_odds_api_key() == "test-api-key"


def test_get_odds_api_key_rejects_missing_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail when the API key is missing."""

    monkeypatch.delenv(
        "ODDS_API_KEY",
        raising=False,
    )

    with pytest.raises(
        RuntimeError,
        match="ODDS_API_KEY is missing",
    ):
        get_odds_api_key()


def test_get_odds_api_key_rejects_placeholder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail when the placeholder value is still configured."""

    monkeypatch.setenv(
        "ODDS_API_KEY",
        "IDE_JÖN_A_SAJÁT_VALÓDI_API_KULCSOD",
    )

    with pytest.raises(
        RuntimeError,
        match="placeholder value",
    ):
        get_odds_api_key()