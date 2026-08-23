from src.dashboard.team_branding import (
    NFLVERSE_TEAMS_LICENSE_URL,
    NFLVERSE_TEAMS_SOURCE_URL,
    TEAM_BRANDS,
    get_team_brand,
)


def test_all_current_teams_have_branding() -> None:
    assert len(TEAM_BRANDS) == 32
    assert all(
        brand.identity_mode == "NFLVERSE_REMOTE_LOGO"
        for brand in TEAM_BRANDS.values()
    )
    assert all(
        brand.logo_url and brand.logo_url.startswith("https://")
        for brand in TEAM_BRANDS.values()
    )


def test_logo_metadata_has_auditable_source_and_license() -> None:
    assert NFLVERSE_TEAMS_SOURCE_URL.startswith("https://github.com/nflverse/")
    assert NFLVERSE_TEAMS_LICENSE_URL.endswith("/LICENSE")


def test_historical_alias_uses_current_brand() -> None:
    assert get_team_brand("OAK").code == "LV"
    assert get_team_brand("SD").code == "LAC"
    assert get_team_brand("STL").code == "LA"


def test_unknown_team_has_safe_fallback() -> None:
    brand = get_team_brand("XYZ")
    assert brand.code == "XYZ"
    assert brand.display_name == "XYZ"
    assert brand.logo_url is None
    assert brand.identity_mode == "BADGE_FALLBACK"
