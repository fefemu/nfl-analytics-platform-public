from pathlib import Path

import pytest

from src.deployment.inject_ga4 import (
    END_MARKER,
    START_MARKER,
    build_ga4_shell,
    inject_ga4_into_index,
)


def test_shell_requires_valid_measurement_id():
    with pytest.raises(ValueError, match="Invalid GA4 Measurement ID"):
        build_ga4_shell("UA-123")


def test_shell_uses_explicit_basic_consent_and_disables_ads():
    content = build_ga4_shell("G-ABCD1234")

    assert "Statisztikai sütik engedélyezése" in content
    assert "Essential only" in content
    assert "localStorage.setItem(CONSENT_KEY,value)" in content
    assert "if(value==='granted')load()" in content
    assert "allow_google_signals:false" in content
    assert "allow_ad_personalization_signals:false" in content
    assert "setInterval(track,750)" in content


def test_injection_is_idempotent_and_can_replace_id(tmp_path: Path):
    index_file = tmp_path / "index.html"
    index_file.write_text("<html><head><title>App</title></head></html>", encoding="utf-8")

    assert inject_ga4_into_index(index_file, "G-FIRST123") is True
    assert inject_ga4_into_index(index_file, "G-FIRST123") is False
    assert inject_ga4_into_index(index_file, "G-SECOND456") is True

    content = index_file.read_text(encoding="utf-8")
    assert content.count(START_MARKER) == 1
    assert content.count(END_MARKER) == 1
    assert "G-SECOND456" in content
    assert "G-FIRST123" not in content


def test_injection_rejects_unknown_index_shape(tmp_path: Path):
    index_file = tmp_path / "index.html"
    index_file.write_text("<html></html>", encoding="utf-8")

    with pytest.raises(ValueError, match="does not contain"):
        inject_ga4_into_index(index_file, "G-ABCD1234")
