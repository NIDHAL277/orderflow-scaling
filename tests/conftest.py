from __future__ import annotations

from pathlib import Path

import pytest

from helpers import make_project
from ofs.config import Project
from ofs.logs import configure_logging


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "OFS_ROOT",
        "OFS_CONFIG_DIR",
        "OFS_DATA_DIR",
        "OFS_CODE_VERSION",
        "OFS_HEALTHCHECKS_URL",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("OFS_PROTOCOL_GUARD", "off")
    configure_logging(None, level="WARNING")


@pytest.fixture
def project(tmp_path: Path) -> Project:
    return make_project(tmp_path / "repo")
