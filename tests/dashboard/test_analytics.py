from src.dashboard.analytics import (
    DEFAULT_GA4_MEASUREMENT_ID,
    GA4_COMPONENT_JS,
    GA4_MEASUREMENT_ID,
    ga4_component_data,
)


def test_ga4_component_data_uses_measurement_id_and_page_context():
    data = ga4_component_data("BETTING", "HU")

    assert DEFAULT_GA4_MEASUREMENT_ID == "G-1X5E3S0J02"
    assert data == {
        "measurement_id": GA4_MEASUREMENT_ID,
        "page": "BETTING",
        "language": "HU",
    }


def test_ga4_v2_component_disables_advertising_features():
    assert "analytics_storage: 'granted'" in GA4_COMPONENT_JS
    assert "ad_storage: 'denied'" in GA4_COMPONENT_JS
    assert "allow_google_signals: false" in GA4_COMPONENT_JS
    assert "document.head.appendChild(script)" in GA4_COMPONENT_JS


def test_ga4_v2_component_deduplicates_virtual_page_views():
    assert "tracker.lastPage !== pageKey" in GA4_COMPONENT_JS
    assert "window.location.href" in GA4_COMPONENT_JS
    assert "window.parent" not in GA4_COMPONENT_JS
    assert "gtag('event', 'page_view'" in GA4_COMPONENT_JS
