"""Consent-aware GA4 tracking integrated into the Streamlit page."""

from __future__ import annotations

import os

import streamlit as st


DEFAULT_GA4_MEASUREMENT_ID = "G-1X5E3S0J02"
GA4_MEASUREMENT_ID = os.getenv(
    "NFL_ANALYTICS_GA4_MEASUREMENT_ID",
    DEFAULT_GA4_MEASUREMENT_ID,
).strip()
CONSENT_STATE_KEY = "dashboard_analytics_consent"

GA4_COMPONENT_JS = r"""
export default function(component) {
  const data = component.data || {};
  const measurementId = data.measurement_id;
  const page = data.page;
  const language = data.language;

  if (!measurementId || !/^G-[A-Z0-9]+$/.test(measurementId)) {
    return;
  }

  const payload = JSON.stringify({ measurementId, page, language })
    .replace(/</g, '\\u003c');
  const bootstrap = document.createElement('script');
  bootstrap.dataset.nflAnalyticsGa4Bootstrap = 'true';
  bootstrap.textContent = `
    (function(data) {
      const trackerKey = '__nflAnalyticsGa4';
      const pageKey = data.page + ':' + data.language;
      const tracker = window[trackerKey] || { lastPage: null };
      window[trackerKey] = tracker;
      window.dataLayer = window.dataLayer || [];
      window.gtag = window.gtag || function() {
        window.dataLayer.push(arguments);
      };

      if (!tracker.initialized) {
        window.gtag('consent', 'default', {
          analytics_storage: 'granted',
          ad_storage: 'denied',
          ad_user_data: 'denied',
          ad_personalization: 'denied'
        });
        window.gtag('js', new Date());
        window.gtag('config', data.measurementId, {
          send_page_view: false,
          allow_google_signals: false,
          allow_ad_personalization_signals: false
        });
        const script = document.createElement('script');
        script.async = true;
        script.src = 'https://www.googletagmanager.com/gtag/js?id=' +
          encodeURIComponent(data.measurementId);
        script.dataset.nflAnalyticsGa4 = data.measurementId;
        script.addEventListener('load', function() {
          document.documentElement.dataset.nflAnalyticsGa4 = 'loaded';
        });
        document.head.appendChild(script);
        tracker.initialized = true;
      }

      if (tracker.lastPage !== pageKey) {
        tracker.lastPage = pageKey;
        const url = new URL(window.location.href);
        url.searchParams.set('language', data.language);
        url.searchParams.set('page', data.page);
        window.gtag('event', 'page_view', {
          page_title: data.page,
          page_location: url.toString(),
          page_path: '/' + data.page.toLowerCase(),
          dashboard_page: data.page,
          dashboard_language: data.language
        });
        document.documentElement.dataset.nflAnalyticsGa4Page = pageKey;
      }
    })(${payload});
  `;
  document.head.appendChild(bootstrap);
  bootstrap.remove();
}
"""

_ga4_component = st.components.v2.component(
    "nfl_analytics_ga4",
    html='<span data-nfl-analytics-ga4="mounted" hidden></span>',
    js=GA4_COMPONENT_JS,
    isolate_styles=False,
)


def ga4_component_data(page_key: str, language: str) -> dict[str, str]:
    """Return trusted, explicit data passed to the GA4 component."""

    return {
        "measurement_id": GA4_MEASUREMENT_ID,
        "page": page_key,
        "language": language,
    }


def render_analytics(page_key: str, language: str) -> None:
    """Request consent once per session and mount GA4 only after acceptance."""

    if not GA4_MEASUREMENT_ID:
        return

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
        _ga4_component(
            key="nfl_analytics_ga4_tracker",
            data=ga4_component_data(page_key, language),
        )
