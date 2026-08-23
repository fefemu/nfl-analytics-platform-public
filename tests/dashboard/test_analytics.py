import pytest

from src.dashboard.analytics import build_ga4_html, get_ga4_measurement_id


def test_measurement_id_is_loaded_from_environment(monkeypatch):
    monkeypatch.setenv("NFL_ANALYTICS_GA4_MEASUREMENT_ID", "g-abcd1234")

    assert get_ga4_measurement_id() == "G-ABCD1234"


def test_invalid_measurement_id_disables_analytics(monkeypatch):
    monkeypatch.setenv("NFL_ANALYTICS_GA4_MEASUREMENT_ID", "UA-legacy")

    assert get_ga4_measurement_id() is None


def test_ga4_html_uses_privacy_defaults_and_page_metadata():
    content = build_ga4_html("G-ABCD1234", "BETTING", "HU")

    assert "'analytics_storage': 'denied'" in content
    assert "'ad_storage': 'denied'" in content
    assert "'allow_google_signals': false" in content
    assert "'send_page_view': false" in content
    assert "gtag('event', 'page_view'" in content
    assert "BETTING:HU" in content
    assert "dashboard_language" in content
    assert "'/betting'" in content


def test_ga4_html_rejects_invalid_measurement_id():
    with pytest.raises(ValueError, match="Invalid GA4 Measurement ID"):
        build_ga4_html("UA-123", "OVERVIEW", "EN")
