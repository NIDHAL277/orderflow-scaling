from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pytest

from helpers import REPO
from ofs.config import (
    Grid,
    Period,
    Project,
    RenameTable,
    SecondsGrid,
    date_range,
    dump_universe,
    load_estimation_config,
    load_pipeline_config,
    load_universe,
    model_hash,
)


def test_repository_configs_load() -> None:
    cfg = load_estimation_config(REPO / "config" / "estimation.yaml")
    assert cfg.windows_seconds == (1, 5, 10, 30, 60)
    assert cfg.h0.primary_window_s == 10
    assert cfg.inference.bootstrap.replications == 999
    assert cfg.split.development.end < cfg.split.confirmation.start
    assert set(cfg.volume_windows()) == {1, 5, 6, 8, 10, 12, 30, 60}
    pcfg = load_pipeline_config(REPO / "config" / "pipeline.yaml")
    assert pcfg.ingest.concurrency == 8 and pcfg.ingest.retries == 5


def test_missing_pipeline_config_gives_defaults(tmp_path: Path) -> None:
    assert load_pipeline_config(tmp_path / "nope.yaml").data_dir == "data"


def test_grid_values_and_integers() -> None:
    g = Grid(min=1, max=360, points=20)
    assert np.isclose(g.values()[0], 1) and np.isclose(g.values()[-1], 360)
    ints = g.integers()
    assert ints[0] == 1 and ints[-1] == 360 and np.all(np.diff(ints) > 0)
    assert Grid(min=2, max=2, points=1).values().tolist() == [2.0]
    assert Grid(min=1, max=3, points=3, spacing="linear").values().tolist() == [1.0, 2.0, 3.0]
    assert SecondsGrid(min_s=1, max_s=10, points=2).grid().max == 10
    with pytest.raises(ValueError, match="max"):
        Grid(min=5, max=1, points=2)


def test_period_and_split() -> None:
    p = Period.model_validate({"from": "2025-01-30", "to": "2025-02-02"})
    assert p.days() == date_range(dt.date(2025, 1, 30), dt.date(2025, 2, 2))
    assert len(p.days()) == 4
    with pytest.raises(ValueError, match="ends before"):
        Period.model_validate({"from": "2025-02-02", "to": "2025-01-01"})
    cfg = load_estimation_config(REPO / "config" / "estimation.yaml")
    rolling = cfg.split.period("rolling")
    assert (
        rolling.start == cfg.split.development.start and rolling.end == cfg.split.confirmation.end
    )
    assert cfg.split.period("confirmation") == cfg.split.confirmation


def test_rename_table() -> None:
    table = RenameTable.model_validate(
        {
            "version": 1,
            "renames": [
                {"old": "AUSDT", "new": "BUSDT", "effective": "2025-01-10"},
                {"old": "ZUSDT", "new": "AUSDT", "effective": "2024-01-01", "markets": ["spot"]},
            ],
        }
    )
    assert table.dataset_symbol("BUSDT", "spot", dt.date(2025, 1, 9)) == "AUSDT"
    assert table.dataset_symbol("BUSDT", "spot", dt.date(2025, 1, 10)) == "BUSDT"
    assert table.dataset_symbol("BUSDT", "spot", dt.date(2023, 6, 1)) == "ZUSDT"
    assert table.dataset_symbol("BUSDT", "um", dt.date(2023, 6, 1)) == "AUSDT"
    assert table.aliases("BUSDT") == ["AUSDT", "ZUSDT"]


def test_model_hash_is_stable_and_sensitive() -> None:
    cfg = load_estimation_config(REPO / "config" / "estimation.yaml")
    assert model_hash(cfg) == model_hash(cfg)
    other = cfg.model_copy(update={"seed": cfg.seed + 1})
    assert model_hash(other) != model_hash(cfg)
    assert len(model_hash({"a": 1}, length=8)) == 8


def test_project_discovery_and_universe(
    project: Project, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert project.data_dir == (project.root / "data").resolve()
    uni = project.universe
    assert [p.symbol for p in uni.top(2)] == ["AAAUSDT", "BBBUSDT"]
    assert [p.symbol for p in uni.assets("um")] == ["AAAUSDT", "BBBUSDT"]
    assert uni.get("BBBUSDT").market_symbol("um") == "1000BBBUSDT"
    assert uni.get("CCCUSDT").listed("spot") == dt.date(2025, 6, 1)
    with pytest.raises(KeyError):
        uni.get("ZZZUSDT")
    renames = project.renames
    assert renames.version == 2
    # the real table: the perpetual stopped trading before the spot pair (MATIC -> POL)
    assert renames.dataset_symbol("POLUSDT", "spot", dt.date(2024, 9, 10)) == "MATICUSDT"
    assert renames.dataset_symbol("POLUSDT", "um", dt.date(2024, 9, 10)) == "POLUSDT"
    assert renames.suspended("POLUSDT", "um", dt.date(2024, 9, 10))
    assert not renames.suspended("POLUSDT", "spot", dt.date(2024, 9, 10))
    assert not renames.suspended("POLUSDT", "um", dt.date(2024, 9, 13))
    monkeypatch.setenv("OFS_DATA_DIR", str(tmp_path / "elsewhere"))
    assert Project.discover(project.root).data_dir == (tmp_path / "elsewhere").resolve()
    out = tmp_path / "u.yaml"
    dump_universe(uni, out, "# header\n")
    assert out.read_text(encoding="utf-8").startswith("# header")
    assert load_universe(out).pairs[0].symbol == "AAAUSDT"


def test_project_without_renames(project: Project) -> None:
    (project.config_dir / "symbol_renames.yaml").unlink()
    fresh = Project.discover(project.root)
    assert fresh.renames.renames == ()
