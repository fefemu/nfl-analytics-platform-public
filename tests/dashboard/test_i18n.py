import pytest

from src.dashboard.i18n import COPY, tr


def test_every_copy_key_has_both_languages() -> None:
    assert COPY
    assert all(set(translations) == {"EN", "HU"} for translations in COPY.values())


def test_hungarian_copy_preserves_technical_terms() -> None:
    assert tr("HU", "model_suite") == "Model suite"
    assert tr("HU", "top_expected") == "Legmagasabb várható győzelemszám"
    assert "Elo" in tr("HU", "dynamic_frozen")


def test_missing_translation_is_rejected() -> None:
    with pytest.raises(KeyError, match="Missing dashboard translation"):
        tr("HU", "missing")
