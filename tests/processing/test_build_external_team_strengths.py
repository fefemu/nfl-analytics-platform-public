from pathlib import Path

import pandas as pd
import pytest

from src.processing.build_external_team_strengths import (
    download_csv_with_retries,
)


def test_download_csv_supports_local_fixture(tmp_path: Path) -> None:
    source = tmp_path / "ratings.csv"
    source.write_text("team,season,rating\nKC,2026,1.0\n", encoding="utf-8")

    result = download_csv_with_retries(
        source_url=str(source),
        source_name="fixture",
    )

    assert len(result) == 1
    assert result.loc[0, "team"] == "KC"


def test_download_rejects_invalid_attempt_count() -> None:
    with pytest.raises(ValueError, match="positive"):
        download_csv_with_retries(
            source_url="fixture.csv",
            source_name="fixture",
            maximum_attempts=0,
        )
