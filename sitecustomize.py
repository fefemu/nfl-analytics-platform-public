"""Early process hook used by the hosted Streamlit deployment."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

from src.deployment.inject_ga4 import inject_ga4_into_index


def _install_ga4_shell() -> None:
    measurement_id = os.getenv("NFL_ANALYTICS_GA4_MEASUREMENT_ID", "").strip()
    if not measurement_id:
        return
    streamlit_spec = importlib.util.find_spec("streamlit")
    if streamlit_spec is None or streamlit_spec.origin is None:
        return
    index_file = Path(streamlit_spec.origin).parent / "static" / "index.html"
    if not index_file.is_file():
        return
    try:
        inject_ga4_into_index(index_file, measurement_id)
    except (OSError, ValueError):
        return


_install_ga4_shell()
