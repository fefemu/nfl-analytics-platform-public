import base64
from xml.etree import ElementTree
import re

from src.dashboard.pages.about import (
    ARCHITECTURE_DIAGRAM,
    DATA_FLOW_DIAGRAM,
    DATA_MODEL_DIAGRAM,
    _localized_diagram,
    EMAIL_ADDRESS,
    GITHUB_URL,
    LINKEDIN_URL,
)


def _decode_diagram(uri: str) -> str:
    prefix, encoded = uri.split(",", maxsplit=1)
    assert prefix == "data:image/svg+xml;base64"
    return base64.b64decode(encoded).decode("utf-8")


def test_about_diagrams_are_localized_in_english() -> None:
    diagrams = (
        (_decode_diagram(_localized_diagram(ARCHITECTURE_DIAGRAM, "EN")), "Technology architecture", "Technológiai architektúra"),
        (_decode_diagram(_localized_diagram(DATA_MODEL_DIAGRAM, "EN")), "DuckDB data model", "DuckDB adatmodell"),
        (_decode_diagram(_localized_diagram(DATA_FLOW_DIAGRAM, "EN")), "Data refresh workflow", "Adatfrissítési folyamat"),
    )
    for content, english_label, hungarian_label in diagrams:
        assert english_label in content
        assert hungarian_label not in content


def test_about_diagrams_remain_hungarian_in_hungarian_view() -> None:
    assert "Technológiai architektúra" in _decode_diagram(
        _localized_diagram(ARCHITECTURE_DIAGRAM, "HU")
    )


def test_about_assets_exist() -> None:
    assert ARCHITECTURE_DIAGRAM.is_file()
    assert DATA_MODEL_DIAGRAM.is_file()
    assert DATA_FLOW_DIAGRAM.is_file()
    ElementTree.parse(ARCHITECTURE_DIAGRAM)
    ElementTree.parse(DATA_MODEL_DIAGRAM)
    ElementTree.parse(DATA_FLOW_DIAGRAM)


def test_author_links_are_configured() -> None:
    assert GITHUB_URL.startswith("https://github.com/")
    assert LINKEDIN_URL.startswith("https://www.linkedin.com/in/")
    assert "@" in EMAIL_ADDRESS


def test_architecture_stays_at_technology_level() -> None:
    content = ARCHITECTURE_DIAGRAM.read_text(encoding="utf-8")
    for technology in ("Python", "pandas", "DuckDB", "SQL", "scikit-learn", "Streamlit"):
        assert technology in content
    assert "raw." not in content
    assert "analytics." not in content
    assert "current_game_predictions" not in content
    assert "pytest · Git · GitHub Actions" in content
    assert "automatizált minőségbiztosítás" in content


def test_data_model_contains_current_schema_table_counts() -> None:
    content = DATA_MODEL_DIAGRAM.read_text(encoding="utf-8")
    assert len(set(re.findall(r"raw\.[a-z0-9_]+", content))) == 9
    assert len(set(re.findall(r"processed\.[a-z0-9_]+", content))) == 12
    assert len(set(re.findall(r"analytics\.[a-z0-9_]+", content))) == 42
    assert "RAW — Forráshű adatok" in content
    assert "PROCESSED — Tisztított és egységesített adatok" in content
    assert "ANALYTICS — Feature-ök és modellezési adatok" in content
    assert "OUTPUT — Publikálható alkalmazási eredmények" in content


def test_data_flow_contains_success_and_failure_paths() -> None:
    content = DATA_FLOW_DIAGRAM.read_text(encoding="utf-8")
    assert "GitHub Actions" in content
    assert "Előkészítés és modellezés" in content
    assert "Oddsok frissítése" in content
    assert "Publikálás" in content
    assert "FAILED" in content
    assert "Az előző publikált adatállapot változatlan marad" in content
