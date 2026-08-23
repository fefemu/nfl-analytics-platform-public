from src.dashboard.analytics import GA4_MEASUREMENT_ID, build_ga4_component


def test_ga4_component_uses_public_measurement_id_and_page_context():
    html = build_ga4_component("BETTING", "HU")

    assert GA4_MEASUREMENT_ID == "G-1X5E3S0J02"
    assert GA4_MEASUREMENT_ID in html
    assert "BETTING" in html
    assert "dashboard_language" in html


def test_ga4_component_disables_advertising_features_and_deduplicates_pages():
    html = build_ga4_component("OVERVIEW", "EN")

    assert "analytics_storage: 'granted'" in html
    assert "ad_storage: 'denied'" in html
    assert "allow_google_signals: false" in html
    assert "sessionStorage.getItem(storageKey)" in html
