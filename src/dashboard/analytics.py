"""Privacy-conscious Google Analytics 4 integration for the public dashboard."""

from __future__ import annotations

import html
import json
import os
import re

import streamlit as st
import streamlit.components.v1 as components


MEASUREMENT_ID_ENV = "NFL_ANALYTICS_GA4_MEASUREMENT_ID"
_MEASUREMENT_ID_PATTERN = re.compile(r"^G-[A-Z0-9]+$")


def get_ga4_measurement_id() -> str | None:
    """Return a validated GA4 web-stream ID from environment or Streamlit secrets."""

    value = os.getenv(MEASUREMENT_ID_ENV, "").strip().upper()
    if not value:
        try:
            value = str(st.secrets.get(MEASUREMENT_ID_ENV, "")).strip().upper()
        except (FileNotFoundError, KeyError):
            value = ""
    return value if _MEASUREMENT_ID_PATTERN.fullmatch(value) else None


def build_ga4_html(measurement_id: str, page_key: str, language: str) -> str:
    """Build an isolated, consented GA4 page-view tag with rerun deduping."""

    if not _MEASUREMENT_ID_PATTERN.fullmatch(measurement_id):
        raise ValueError("Invalid GA4 Measurement ID.")
    safe_page = page_key if page_key.isidentifier() else "UNKNOWN"
    safe_language = language if language in {"EN", "HU"} else "EN"
    event_key = f"{safe_page}:{safe_language}"
    return f"""
<!doctype html>
<html><head>
<script async src="https://www.googletagmanager.com/gtag/js?id={html.escape(measurement_id)}"></script>
<script>
window.dataLayer = window.dataLayer || [];
function gtag(){{dataLayer.push(arguments);}}
gtag('consent', 'default', {{
  'analytics_storage': 'granted',
  'ad_storage': 'denied',
  'ad_user_data': 'denied',
  'ad_personalization': 'denied'
}});
gtag('js', new Date());
gtag('config', {json.dumps(measurement_id)}, {{
  'send_page_view': false,
  'allow_google_signals': false,
  'allow_ad_personalization_signals': false
}});
try {{
  const storage = window.parent.sessionStorage;
  const key = 'nap_ga4_last_page_view';
  const previous = storage.getItem(key);
  if (previous !== {json.dumps(event_key)}) {{
    storage.setItem(key, {json.dumps(event_key)});
    gtag('event', 'page_view', {{
      'page_title': {json.dumps(safe_page)},
      'page_location': window.parent.location.href,
      'page_path': '/{safe_page.lower()}',
      'dashboard_page': {json.dumps(safe_page)},
      'dashboard_language': {json.dumps(safe_language)}
    }});
  }}
}} catch (error) {{
  gtag('event', 'page_view', {{
    'page_title': {json.dumps(safe_page)},
    'page_path': '/{safe_page.lower()}',
    'dashboard_page': {json.dumps(safe_page)},
    'dashboard_language': {json.dumps(safe_language)}
  }});
}}
</script>
</head><body></body></html>
""".strip()


def render_analytics_consent(language: str) -> bool:
    """Render an explicit opt-in control when GA4 is configured."""

    if get_ga4_measurement_id() is None:
        return False
    if language == "HU":
        label = "Anonim látogatottsági statisztika"
        help_text = (
            "Engedélyezés után a Google Analytics anonim oldalmegtekintési, "
            "eszköz- és hozzávetőleges országadatokat mér. Hirdetési és "
            "személyre szabási funkciókat nem használunk."
        )
    else:
        label = "Anonymous usage analytics"
        help_text = (
            "When enabled, Google Analytics measures anonymous page views, "
            "device information and approximate country. Advertising and "
            "personalization features remain disabled."
        )
    return st.toggle(
        label,
        value=False,
        key="dashboard_analytics_consent",
        help=help_text,
    )


def track_page_view(page_key: str, language: str, consent_granted: bool) -> bool:
    """Emit one GA4 page-view component after explicit analytics consent."""

    measurement_id = get_ga4_measurement_id()
    if measurement_id is None or not consent_granted:
        return False
    components.html(
        build_ga4_html(measurement_id, page_key, language),
        height=1,
        width=1,
    )
    return True
