"""The Streamlit dashboard renders its five pages without exceptions (M9)."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest
from test_forecast import small_forecast_project, write_development_estimates, write_history
from test_inference import write_results

from helpers import REPO
from ofs.config import Project
from ofs.inference.hypotheses import run_tests
from ofs.inference.robustness import specification_curve
from ofs.ingest.manifest import ManifestRecord, upsert
from ofs.paths import DataLayout
from ofs.simulate.study import load_simulation_config, run_study

APP = str(REPO / "dashboards" / "app.py")
PAGES = (
    "pages/tests.py",
    "pages/asset.py",
    "pages/forecast.py",
    "pages/health.py",
    "pages/simulation.py",
)


@pytest.fixture(scope="module")
def results_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from helpers import make_project

    project: Project = make_project(tmp_path_factory.mktemp("dash") / "repo")
    write_results(project, "confirmation", n=10, reps=30)
    run_tests(project, "confirmation")
    specification_curve(project, "confirmation", ["spot"])
    layout = DataLayout(project.data_dir)
    import datetime as dt

    upsert(
        layout.manifest_file,
        [
            ManifestRecord(
                "AAAUSDT", "AAAUSDT", "spot", "aggTrades", dt.date(2026, 10, 1), "u", "ok"
            ),
            ManifestRecord(
                "AAAUSDT", "AAAUSDT", "spot", "aggTrades", dt.date(2026, 10, 2), "u", "missing"
            ),
        ],
    )
    from ofs.io import write_parquet_atomic
    from ofs.quality import day_states, quality_table

    states = day_states(project, dt.date(2026, 10, 1), dt.date(2026, 10, 2))
    write_parquet_atomic(quality_table(states), layout.quality_dir / "quality.parquet")
    sim = load_simulation_config(REPO / "config" / "simulation.yaml")
    run_study(
        sim, project.estimation, "ci", layout.simulation_dir, exponents=("delta",), max_cells=2
    )
    # forecasting extension: development forecasts of two pairs and a tiny simulation check
    from ofs.forecast.models import ForecastSpec
    from ofs.forecast.run import run_forecasts
    from ofs.forecast.simulate import forecast_study

    small = small_forecast_project(project)
    for i, symbol in enumerate(("AAAUSDT", "BBBUSDT")):
        write_history(small, symbol, seed=30 + i)
    write_development_estimates(small)
    run_forecasts(small, "development", ("spot",))
    spec = ForecastSpec(horizons=(1, 5), window=150, min_obs=120, calibration=60)
    _, summary = forecast_study(
        spec, h_values=(0.1,), noise_values=(0.3,), reps=1, n_days=420, lags=(1, 2, 4, 8)
    )
    write_parquet_atomic(summary, layout.simulation_dir / "forecast_study.parquet")
    return project.data_dir


@pytest.mark.parametrize("page", PAGES)
def test_dashboard_pages_render(
    results_dir: Path, page: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OFS_DATA_DIR", str(results_dir))
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    at.switch_page(page).run()
    assert not at.exception, at.exception
    assert at.title[0].value  # every page has a title


def test_dashboard_on_empty_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OFS_DATA_DIR", str(tmp_path / "empty"))
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception
    assert any("No test results" in i.value for i in at.info)
