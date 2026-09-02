import pytest

from src.dashboard.i18n import COPY, tr


def test_every_copy_key_has_both_languages() -> None:
    assert COPY
    assert all(set(translations) == {"EN", "HU"} for translations in COPY.values())


def test_hungarian_copy_translates_prose_but_preserves_technical_terms() -> None:
    assert tr("HU", "model_suite") == "Modellcsomag"
    assert tr("HU", "top_expected") == "Legmagasabb várható győzelemszám"
    assert "Elo" in tr("HU", "dynamic_frozen")


def test_weekly_overview_describes_the_selected_week() -> None:
    assert "selected week" in tr("EN", "subtitle_overview")
    assert "kiválasztott hét" in tr("HU", "subtitle_overview")
    assert "Current-week" not in tr("EN", "subtitle_overview")
    assert "aktuális hét" not in tr("HU", "subtitle_overview")


def test_common_hungarian_labels_do_not_mix_english_prose() -> None:
    assert tr("HU", "last_refresh") == "Utolsó sikeres frissítés"
    assert tr("HU", "market_comparison") == "Piaci összehasonlítás"
    assert tr("HU", "simulations") == "Szimulációk"


def test_missing_translation_is_rejected() -> None:
    with pytest.raises(KeyError, match="Missing dashboard translation"):
        tr("HU", "missing")
