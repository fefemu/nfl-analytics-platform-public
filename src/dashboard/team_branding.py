"""Canonical team identity metadata for the dashboard."""

from dataclasses import dataclass

from src.config.nfl_team_mappings import ODDS_TEAM_TO_NFLVERSE, normalize_franchise_code


@dataclass(frozen=True)
class TeamBrand:
    code: str
    display_name: str
    primary_color: str
    secondary_color: str
    logo_url: str | None = None
    identity_mode: str = "NFLVERSE_REMOTE_LOGO"


NFLVERSE_TEAMS_SOURCE_URL = (
    "https://github.com/nflverse/nflverse-data/releases/tag/teams"
)
NFLVERSE_TEAMS_LICENSE_URL = (
    "https://github.com/nflverse/nflverse-data/blob/master/LICENSE"
)


_COLORS = {
    "ARI": ("#97233F", "#FFB612"), "ATL": ("#A71930", "#A5ACAF"),
    "BAL": ("#241773", "#9E7C0C"), "BUF": ("#00338D", "#C60C30"),
    "CAR": ("#0085CA", "#101820"), "CHI": ("#0B162A", "#C83803"),
    "CIN": ("#FB4F14", "#000000"), "CLE": ("#311D00", "#FF3C00"),
    "DAL": ("#003594", "#869397"), "DEN": ("#FB4F14", "#002244"),
    "DET": ("#0076B6", "#B0B7BC"), "GB": ("#203731", "#FFB612"),
    "HOU": ("#03202F", "#A71930"), "IND": ("#002C5F", "#A2AAAD"),
    "JAX": ("#006778", "#D7A22A"), "KC": ("#E31837", "#FFB81C"),
    "LV": ("#000000", "#A5ACAF"), "LAC": ("#0080C6", "#FFC20E"),
    "LA": ("#003594", "#FFA300"), "MIA": ("#008E97", "#FC4C02"),
    "MIN": ("#4F2683", "#FFC62F"), "NE": ("#002244", "#C60C30"),
    "NO": ("#D3BC8D", "#101820"), "NYG": ("#0B2265", "#A71930"),
    "NYJ": ("#125740", "#FFFFFF"), "PHI": ("#004C54", "#A5ACAF"),
    "PIT": ("#FFB612", "#101820"), "SF": ("#AA0000", "#B3995D"),
    "SEA": ("#002244", "#69BE28"), "TB": ("#D50A0A", "#FF7900"),
    "TEN": ("#0C2340", "#4B92DB"), "WAS": ("#5A1414", "#FFB612"),
}

_NAMES_BY_CODE = {code: name for name, code in ODDS_TEAM_TO_NFLVERSE.items()}

# These are team_logo_espn values published by nflverse's teams metadata.
# They remain remote assets; this repository does not redistribute logo files.
_LOGO_URLS = {
    code: f"https://a.espncdn.com/i/teamlogos/nfl/500/{slug}.png"
    for code, slug in {
        "ARI": "ari", "ATL": "atl", "BAL": "bal", "BUF": "buf",
        "CAR": "car", "CHI": "chi", "CIN": "cin", "CLE": "cle",
        "DAL": "dal", "DEN": "den", "DET": "det", "GB": "gb",
        "HOU": "hou", "IND": "ind", "JAX": "jax", "KC": "kc",
        "LA": "lar", "LAC": "lac", "LV": "lv", "MIA": "mia",
        "MIN": "min", "NE": "ne", "NO": "no", "NYG": "nyg",
        "NYJ": "nyj", "PHI": "phi", "PIT": "pit", "SF": "sf",
        "SEA": "sea", "TB": "tb", "TEN": "ten", "WAS": "wsh",
    }.items()
}
_LOGO_URLS["CAR"] = "https://a.espncdn.com/i/teamlogos/nfl/500-dark/car.png"

TEAM_BRANDS = {
    code: TeamBrand(code, _NAMES_BY_CODE[code], *colors, _LOGO_URLS[code])
    for code, colors in _COLORS.items()
}


def get_team_brand(team_code: str) -> TeamBrand:
    """Return current-franchise branding for a project team code."""

    code = normalize_franchise_code(team_code)
    try:
        return TEAM_BRANDS[code]
    except KeyError:
        return TeamBrand(
            code,
            code,
            "#334155",
            "#94A3B8",
            logo_url=None,
            identity_mode="BADGE_FALLBACK",
        )
