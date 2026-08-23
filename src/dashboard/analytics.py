"""Consent-aware GA4 tracking rendered inside the Streamlit application."""

from __future__ import annotations

import json

import streamlit as st
import streamlit.components.v1 as components


GA4_MEASUREMENT_ID = "G-1X5E3S0J02"
CONSENT_STATE_KEY = "dashboard_analytics_consent"


def build_ga4_component(page_key: str, language: str) -> str:
    """Build the GA4 component loaded only after explicit consent."""

    measurement_id = json.dumps(GA4_MEASUREMENT_ID)
    page = json.dumps(page_key)
    lang = json.dumps(language)
    return f"""<!doctype html><html><head><meta charset="utf-8"></head><body>
<script>
(function() {{
  const measurementId = {measurement_id};
  const page = {page};
  const language = {lang};
  const pageKey = page + ':' + language;
  const storageKey = 'nap_ga4_last_page';
  window.dataLayer = window.dataLayer || [];
  window.gtag = function() {{ dataLayer.push(arguments); }};
  gtag('js', new Date());
  gtag('consent', 'default', {{
    analytics_storage: 'granted',
    ad_storage: 'denied',
    ad_user_data: 'denied',
    ad_personalization: 'denied'
  }});
  gtag('config', measurementId, {{
    send_page_view: false,
    allow_google_signals: false,
    allow_ad_personalization_signals: false
  }});
  const script = document.createElement('script');
  script.async = true;
  script.src = 'https://www.googletagmanager.com/gtag/js?id=' + encodeURIComponent(measurementId);
  document.head.appendChild(script);
  if (sessionStorage.getItem(storageKey) !== pageKey) {{
    sessionStorage.setItem(storageKey, pageKey);
    gtag('event', 'page_view', {{
      page_title: page,
      page_location: document.referrer || window.location.href,
      page_path: '/' + page.toLowerCase(),
      dashboard_page: page,
      dashboard_language: language
    }});
  }}
}})();
</script></body></html>"""


def render_analytics(page_key: str, language: str) -> None:
    """Request consent once per session and render GA4 after acceptance."""

    if CONSENT_STATE_KEY not in st.session_state:

        @st.dialog(
            "Anonim látogatottsági statisztika"
            if language == "HU"
            else "Anonymous usage analytics",
            dismissible=False,
        )
        def consent_dialog() -> None:
            st.write(
                "Az oldal a Google Analytics segítségével anonim látogatottsági "
                "adatokat mérhet. Ezek az adatok a platform fejlesztését segítik; "
                "hirdetési és személyre szabási funkciókat nem használunk."
                if language == "HU"
                else "This site can use Google Analytics to measure anonymous visits and "
                "improve the platform. Advertising and personalization features are disabled."
            )
            essential, accept = st.columns(2)
            if essential.button(
                "Csak szükséges" if language == "HU" else "Essential only",
                use_container_width=True,
            ):
                st.session_state[CONSENT_STATE_KEY] = "denied"
                st.rerun()
            if accept.button(
                "Statisztikai mérés engedélyezése"
                if language == "HU"
                else "Allow usage analytics",
                type="primary",
                use_container_width=True,
            ):
                st.session_state[CONSENT_STATE_KEY] = "granted"
                st.rerun()

        consent_dialog()

    if st.session_state.get(CONSENT_STATE_KEY) == "granted":
        components.html(build_ga4_component(page_key, language), height=0, width=0)
