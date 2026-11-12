from __future__ import annotations

import datetime as dt
import json
import math

import numpy as np
import polars as pl
import pytest

from helpers import make_project
from ofs.config import Project
from ofs.estimators.registry import build_registry, role_of
from ofs.estimators.run import (
    BootstrapPlan,
    clean_json,
    day_masks,
    estimate_asset,
    load_bundle,
    make_plan,
    market_regimes,
    replicates,
    run_estimates,
)
from ofs.inference.bootstrap import rng_for
from ofs.io import write_parquet_atomic
from ofs.paths import DataLayout
from ofs.schemas import BOOTSTRAP, ESTIMATES
from ofs.simulate.fbm import mixed_fgn
from ofs.simulate.rough_vol import observed_prices, rough_bergomi
from ofs.simulate.signs import lmf_signs
from ofs.simulate.study import (
    flow_bundle,
    impact_bundle,
    signs_bundle,
    volatility_bundle,
    volume_bundle,
)
from ofs.simulate.volume import rough_rate_day, synthetic_runs

N_DAYS = 14


def synthetic_bundle(
    project: Project, days: list[dt.date], seed: int, H0: float = 0.8
) -> dict[str, pl.DataFrame]:
    cfg = project.estimation
    rng = rng_for(seed, "bundle")
    n = len(days)
    bundle = flow_bundle(mixed_fgn(n * 8640, H0, 1.0, 0.5, rng), days, cfg, 10)
    bundle |= signs_bundle([lmf_signs(20_000, H0, rng, n_orders=10) for _ in days], days, cfg)
    bundle |= volume_bundle(
        [rough_rate_day(8640, H0 - 0.5, 0.5, 0.0, rng) for _ in days], days, cfg, 10
    )
    logp, _ = rough_bergomi(n, 8640, 2 * H0 - 1.5, 0.6, 0.0009, -0.3, rng)
    bundle |= volatility_bundle(observed_prices(logp, 0.0, 0.0, rng), days, cfg, 8640)
    bundle |= impact_bundle([synthetic_runs(4000, 2 - 2 * H0, 0.5, rng) for _ in days], days, cfg)
    jumps = [1 if i == 3 else 0 for i in range(n)]
    bundle["day_summary"] = pl.DataFrame({"date": days, "n_jump_seconds": jumps})
    return bundle


def write_bundle(
    layout: DataLayout, asset: str, market: str, bundle: dict[str, pl.DataFrame]
) -> None:
    for kind, df in bundle.items():
        for (day,), part in df.group_by(["date"]):
            path = layout.day_file(layout.stats_table(kind), asset, market, day, "test")  # type: ignore[arg-type]
            write_parquet_atomic(
                part.with_columns(pl.lit(asset).alias("symbol"), pl.lit(market).alias("market")),
                path,
            )


@pytest.fixture(scope="module")
def stats_project(tmp_path_factory: pytest.TempPathFactory) -> tuple[Project, list[dt.date]]:
    project = make_project(tmp_path_factory.mktemp("stats") / "repo")
    layout = DataLayout(project.data_dir)
    days = [dt.date(2025, 11, 3) + dt.timedelta(days=i) for i in range(N_DAYS)]
    for i, (asset, market) in enumerate(
        [("AAAUSDT", "spot"), ("AAAUSDT", "um"), ("BBBUSDT", "spot")]
    ):
        write_bundle(layout, asset, market, synthetic_bundle(project, days, seed=i))
    return project, days


def test_registry_roles(project: Project) -> None:
    reg = build_registry(project.estimation)
    for exponent in ("H0", "Hvol", "Hsigma", "delta"):
        roles = [e.role for e in reg if e.exponent == exponent]
        assert roles.count("primary") == 1 and roles.count("control") == 1, exponent
    assert all(not e.bootstrap for e in reg if e.role == "diagnostic")
    # roles follow the configuration, as fixed by the simulation study (pre-registration §4.1)
    cfg = project.estimation
    assert role_of(reg, "Hsigma", "primary").name == cfg.hsigma.primary_estimator
    assert role_of(reg, "Hsigma", "control").name == cfg.hsigma.control_estimator
    assert role_of(reg, "Hvol", "primary").name == cfg.hvol.primary_estimator
    assert role_of(reg, "H0", "control").name == cfg.h0.concordance_control
    # deviations D3 and D4 (ADR 0015): volume-centred principal, control and negative
    # control; the pre-registered uncentred principal kept as a robustness variant
    primary = role_of(reg, "H0", "primary")
    assert primary.name == "mixed_fbm_vc" and getattr(primary, "flow_variant", "") == "signed_vc"
    assert getattr(role_of(reg, "H0", "control"), "flow_variant", "") == "signed_vc"
    uncentred = [e for e in reg if e.exponent == "H0" and e.variant == "uncentred"]
    assert [(e.name, e.role) for e in uncentred] == [("mixed_fbm", "robustness")]
    negative = [e for e in reg if e.role == "negative_control"]
    assert [(e.name, getattr(e, "flow_variant", "")) for e in negative] == [
        ("mixed_fbm_vc", "shuffled_vc")
    ]
    keys = [e.key for e in reg]
    assert len(keys) == len(set(keys))
    with pytest.raises(KeyError):
        role_of(reg, "H0", "nonexistent")


def test_registry_follows_config_switches(project: Project) -> None:
    cfg = project.estimation
    switched = cfg.model_copy(
        update={
            "hsigma": cfg.hsigma.model_copy(
                update={"primary_estimator": "whittle_fukasawa", "control_estimator": "gjr"}
            ),
            "hvol": cfg.hvol.model_copy(
                update={"primary_estimator": "acov_gmm", "control_estimator": "moments"}
            ),
            "h0": cfg.h0.model_copy(update={"concordance_control": "qv_ratio"}),
        }
    )
    reg = build_registry(switched)
    assert role_of(reg, "Hsigma", "primary").name == "whittle_fukasawa"
    assert role_of(reg, "Hsigma", "control").name == "gjr"
    assert role_of(reg, "Hvol", "primary").name == "acov_gmm"
    assert (
        role_of(reg, "H0", "control").name == "qv_ratio" and role_of(reg, "H0", "control").bootstrap
    )
    intraday = cfg.model_copy(
        update={"hsigma": cfg.hsigma.model_copy(update={"primary": "intraday"})}
    )
    assert role_of(build_registry(intraday), "Hsigma", "primary").name == "gjr_intraday"


def test_plan_is_synchronized_and_deterministic(project: Project) -> None:
    days = [dt.date(2025, 1, 1) + dt.timedelta(days=i) for i in range(30)]
    a = make_plan(project, "development", reps=25, days=days)
    b = make_plan(project, "development", reps=25, days=days)
    assert set(a.weights) == {"", "block=3", "block=14"}
    assert np.array_equal(a.indices[""], b.indices[""]) and a.main.shape == (25, 30)
    assert np.all(a.main.sum(axis=1) == 30)
    c = make_plan(project, "confirmation", reps=25, days=days)
    assert not np.array_equal(a.indices[""], c.indices[""])


def test_estimate_asset_in_memory(stats_project: tuple[Project, list[dt.date]]) -> None:
    project, days = stats_project
    layout = DataLayout(project.data_dir)
    bundle = load_bundle(layout, "AAAUSDT", "spot", days)
    assert bundle["flow_var"]["date"].n_unique() == N_DAYS
    assert load_bundle(layout, "AAAUSDT", "spot", days[:2])["rv"].height == 2
    plan = make_plan(project, "confirmation", reps=30, days=days)
    regimes = {d: ("high" if i % 2 else "low") for i, d in enumerate(days)}
    res = estimate_asset(project, "AAAUSDT", "spot", "confirmation", plan, regimes, bundle=bundle)
    rows = pl.DataFrame([{k: v for k, v in r.items() if k != "diagnostics"} for r in res.rows])
    prim = rows.filter(pl.col("role") == "primary")
    assert set(prim["exponent"]) == {"H0", "Hvol", "Hsigma", "delta"}
    h0 = prim.filter(pl.col("exponent") == "H0")
    assert h0["ci_low"][0] <= h0["value"][0] <= h0["ci_high"][0]
    variants = set(rows.filter(pl.col("role") == "robustness")["variant"])
    assert {"delta=10s|block=3", "delta=10s|block=14"} <= variants
    whittle = rows.filter(pl.col("estimator") == "whittle_fukasawa")
    role = (
        "primary"
        if project.estimation.hsigma.primary_estimator == "whittle_fukasawa"
        else "control"
    )
    assert whittle.height == 1 and whittle["role"][0] == role  # 13 increments < 30: honestly NaN
    assert whittle["value"][0] is None or math.isfinite(whittle["value"][0])
    diag = next(
        r["diagnostics"] for r in res.rows if r["role"] == "primary" and r["exponent"] == "H0"
    )
    assert "concordant" in diag or "control_value" not in diag
    assert {b[0] for b in res.boot} == set(range(30))


def test_masks_regimes_and_json(stats_project: tuple[Project, list[dt.date]]) -> None:
    project, days = stats_project
    layout = DataLayout(project.data_dir)
    regimes = market_regimes(layout, ["AAAUSDT", "BBBUSDT"], "spot", days)
    assert set(regimes.values()) <= {"high", "low"} and len(regimes) == N_DAYS
    masks = day_masks(load_bundle(layout, "AAAUSDT", "spot", days), days, regimes)
    assert set(masks) == {"regime=high", "regime=low", "no_jump_days"}
    assert masks["no_jump_days"].sum() == N_DAYS - 1
    assert market_regimes(layout, ["ZZZUSDT"], "spot", days) == {}
    assert clean_json(
        {"a": np.float64(np.nan), "b": [np.int64(2), np.array([1.5])], 3: (np.inf,)}
    ) == {"a": None, "b": [2, [1.5]], "3": [None]}


def test_replicates_modes(stats_project: tuple[Project, list[dt.date]]) -> None:
    project, days = stats_project
    bundle = load_bundle(DataLayout(project.data_dir), "AAAUSDT", "spot", days)
    plan: BootstrapPlan = make_plan(project, "confirmation", reps=10, days=days)
    reg = build_registry(project.estimation)
    whittle = next(e for e in reg if e.name == "whittle_fukasawa")
    f = whittle.features(bundle, days)
    assert replicates(whittle, f, f.meta, plan, mask=np.ones(N_DAYS)) is None
    reps = replicates(whittle, f, f.meta, plan)
    assert reps is not None and reps.shape == (10,)
    moments = next(e for e in reg if e.name == "moments" and e.role in ("primary", "control"))
    fm = moments.features(bundle, days)
    r = replicates(moments, fm, fm.meta, plan, "block=3", mask=np.ones(N_DAYS))
    assert r is not None and np.isfinite(r).mean() > 0.8


def test_run_estimates_end_to_end(stats_project: tuple[Project, list[dt.date]]) -> None:
    project, days = stats_project
    layout = DataLayout(project.data_dir)
    est, boot = run_estimates(
        project, "confirmation", ("spot", "um"), reps=12, days=days, symbols=["AAAUSDT", "BBBUSDT"]
    )
    ESTIMATES.validate(est)
    BOOTSTRAP.validate(boot)
    assert set(est["symbol"]) == {"AAAUSDT", "BBBUSDT"} and set(est["market"]) == {"spot", "um"}
    stored = pl.read_parquet(layout.estimates_file("confirmation"))
    assert stored.height == est.height and stored["code_version"][0] and stored["config_version"][0]
    json.loads(stored["diagnostics"][0])
    idx = pl.read_parquet(layout.bootstrap_index_file("confirmation"))
    assert idx.height == 12 * N_DAYS and set(idx["day"]) <= set(days)
    # the same replicate index means the same days for every asset (synchronized)
    b = boot.filter((pl.col("estimator") == "mixed_fbm_vc") & (pl.col("variant") == "delta=10s"))
    assert b.group_by("rep").len()["len"].max() == 3  # AAA spot, AAA um, BBB spot
