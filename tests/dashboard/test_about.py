from src.dashboard.pages.about import (
    ARCHITECTURE_DIAGRAM,
    DATA_MODEL_DIAGRAM,
    DATA_FLOW_DIAGRAM,
    EMAIL_ADDRESS,
    GITHUB_URL,
    LINKEDIN_URL,
)
from xml.etree import ElementTree
import re


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


def test_data_model_contains_current_schema_table_counts() -> None:
    content = DATA_MODEL_DIAGRAM.read_text(encoding="utf-8")
    assert len(set(re.findall(r"raw\.[a-z0-9_]+", content))) == 9
    assert len(set(re.findall(r"processed\.[a-z0-9_]+", content))) == 12
    assert len(set(re.findall(r"analytics\.[a-z0-9_]+", content))) == 42


def test_data_flow_contains_success_and_failure_paths() -> None:
    content = DATA_FLOW_DIAGRAM.read_text(encoding="utf-8")
    assert "Modellezési pipeline" in content
    assert "Odds pipeline" in content
    assert "SUCCESS" in content
    assert "FAILED" in content
