from __future__ import annotations

import datetime as dt
import json
import subprocess
from pathlib import Path

import httpx
import polars as pl
import pytest
from test_inference import write_results
from typer.testing import CliRunner

from helpers import DAY, make_trades, write_aggtrades_zip, write_klines_zip
from ofs import daily as daily_mod
from ofs.cli import app
from ofs.config import Project
from ofs.ingest.manifest import ManifestRecord, read_manifest, status_of, upsert
from ofs.monitoring import disk_fraction, ping, read_status, write_status
from ofs.paths import DataLayout
from ofs.protocol import ProtocolError, check_estimation_allowed
from ofs.versioning import code_version, config_unchanged_since, git_tag_exists, stage_version

runner = CliRunner()


def _cli(project: Project, *args: str) -> object:
    return runner.invoke(app, ["--root", str(project.root), "--log-level", "WARNING", *args])


def test_cli_help_and_status(project: Project) -> None:
    res = runner.invoke(app, ["--help"])
    assert res.exit_code == 0 and "ingest" in res.output and "simulate" in res.output
    res = _cli(project, "status")
    assert res.exit_code == 0 and json.loads(res.output)["status"] == "ok"


def test_cli_validate_reduce_and_verify(project: Project) -> None:
    layout = DataLayout(project.data_dir)
    trades = make_trades(DAY, n=1500, seed=3)
    write_aggtrades_zip(layout.raw_file("spot", "aggTrades", "AAAUSDT", DAY), trades)
    write_klines_zip(layout.raw_file("spot", "klines_1m", "AAAUSDT", DAY), trades)
    upsert(
        layout.manifest_file,
        [ManifestRecord("AAAUSDT", "AAAUSDT", "spot", "aggTrades", DAY, "u", "ok")],
    )
    day = DAY.isoformat()
    res = _cli(project, "validate", "--from", day, "--to", day, "--market", "spot")
    assert res.exit_code == 0 and "1 days checked, 0 blocked" in res.output
    res = _cli(project, "verify-naive", "--n-days", "1")
    assert res.exit_code == 0, res.output
    res = _cli(
        project, "reduce", "--from", day, "--to", day, "--market", "spot", "--symbol", "AAAUSDT"
    )
    assert res.exit_code == 0, res.output
    assert '"pass1_done": 1' in res.output
    assert not layout.raw_file("spot", "aggTrades", "AAAUSDT", DAY).exists()
    assert _cli(project, "verify-naive").exit_code == 1  # no raw file left


def test_cli_validate_reports_blocked_day(project: Project) -> None:
    layout = DataLayout(project.data_dir)
    trades = make_trades(DAY, n=500, seed=4)
    write_aggtrades_zip(
        layout.raw_file("um", "aggTrades", "AAAUSDT", DAY), trades, market="um", unit="ms"
    )
    res = _cli(
        project, "validate", "--from", DAY.isoformat(), "--to", DAY.isoformat(), "--market", "um"
    )
    assert (
        res.exit_code == 1 and "FAIL AAAUSDT um" in res.output
    )  # no manifest entry: sha256 not verified


def test_cli_estimate_respects_protocol(project: Project, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OFS_PROTOCOL_GUARD", "on")
    res = _cli(project, "estimate", "--sample", "development")
    assert res.exit_code == 2 and "prereg-v1" in res.output
    with pytest.raises(ProtocolError, match="prereg-v1"):
        check_estimation_allowed(project, "confirmation")


def test_protocol_guard_from_baked_environment(
    project: Project, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Inside the image there is no .git: CI bakes the tags and the frozen config hash."""
    import hashlib

    monkeypatch.setenv("OFS_PROTOCOL_GUARD", "on")
    monkeypatch.setenv("OFS_PROTOCOL_TAGS", "prereg-v1, config-frozen")
    digest = hashlib.sha256((project.config_dir / "estimation.yaml").read_bytes()).hexdigest()
    monkeypatch.setenv("OFS_FROZEN_CONFIG_SHA256", digest)
    check_estimation_allowed(project, "confirmation")
    monkeypatch.setenv("OFS_FROZEN_CONFIG_SHA256", "0" * 64)
    with pytest.raises(ProtocolError, match="differs"):
        check_estimation_allowed(project, "confirmation")
    monkeypatch.setenv("OFS_PROTOCOL_TAGS", "prereg-v1")
    check_estimation_allowed(project, "development")
    with pytest.raises(ProtocolError, match="config-frozen"):
        check_estimation_allowed(project, "confirmation")


def test_protocol_guard_with_git_tags(project: Project, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OFS_PROTOCOL_GUARD", "on")
    root = project.root

    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)

    git("init", "-q")
    git("-c", "user.email=t@t", "-c", "user.name=t", "add", "-A")
    git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "prereg")
    git("tag", "prereg-v1")
    check_estimation_allowed(project, "development")
    with pytest.raises(ProtocolError, match="config-frozen"):
        check_estimation_allowed(project, "confirmation")
    git("tag", "config-frozen")
    check_estimation_allowed(project, "confirmation")
    cfg = root / "config" / "estimation.yaml"
    cfg.write_text(
        cfg.read_text(encoding="utf-8").replace("seed: 20261106", "seed: 1"), encoding="utf-8"
    )
    with pytest.raises(ProtocolError, match="differs"):
        check_estimation_allowed(project, "confirmation")
    assert git_tag_exists("prereg-v1", root) and not config_unchanged_since(
        "config-frozen", "config/estimation.yaml", root
    )
    code_version.cache_clear()
    assert code_version(root).endswith("+dirty")


def test_cli_test_and_report(project: Project) -> None:
    write_results(project, "confirmation", n=12, reps=40)
    res = _cli(
        project, "test", "--sample", "confirmation", "--market", "spot", "--n-universes", "10"
    )
    assert res.exit_code == 0, res.output
    assert "H1" in res.output
    res = _cli(project, "report", "--sample", "confirmation", "--profile", "quick", "--no-pdf")
    assert res.exit_code == 0, res.output
    figures = sorted(p.name for p in (project.root / "paper" / "figures").glob("*.pdf"))
    assert len(figures) == 9 and (project.root / "paper" / "tables" / "tests.tex").exists()
    assert "\\resultHOne" in (project.root / "paper" / "tables" / "macros.tex").read_text(
        encoding="utf-8"
    )


def test_cli_simulate_study(project: Project) -> None:
    res = _cli(
        project, "simulate", "study", "--profile", "ci", "--exponent", "delta", "--max-cells", "2"
    )
    assert res.exit_code == 0, res.output
    assert (DataLayout(project.data_dir).simulation_dir / "summary_ci.parquet").exists()


# ---------------------------------------------------------------- daily run
def _fake_download(layout: DataLayout, statuses: dict[str, str], absent: set | None = None):
    """Binance Vision as seen at 02:30 on DAY + 1: ``statuses`` applies to yesterday's files
    ("pending" until published), older files of those pairs are 404 ("missing"); the
    (asset, day) pairs in ``absent`` are missing whatever the day."""

    def fake(tasks, lay, cfg, **kw):
        records = []
        for t in tasks:
            status = statuses.get(t.asset, "ok")
            if status == "pending" and t.day < DAY:
                status = "missing"
            if absent and (t.asset, t.day) in absent:
                status = "missing"
            if status == "ok":
                trades = make_trades(t.day, n=800, seed=hash((t.asset, t.market)) % 1000)
                if t.kind == "aggTrades":
                    write_aggtrades_zip(
                        lay.raw_file(t.market, t.kind, t.asset, t.day), trades, market=t.market
                    )
                else:
                    write_klines_zip(
                        lay.raw_file(t.market, t.kind, t.asset, t.day), trades, market=t.market
                    )
            records.append(
                ManifestRecord(t.asset, t.dataset_symbol, t.market, t.kind, t.day, "u", status)
            )  # type: ignore[arg-type]
        upsert(lay.manifest_file, records)
        return records

    return fake


def test_daily_success_and_ping(project: Project, monkeypatch: pytest.MonkeyPatch) -> None:
    layout = DataLayout(project.data_dir)
    pings: list[tuple[str, str]] = []
    monkeypatch.setattr("ofs.pipeline.download", _fake_download(layout, {}))
    monkeypatch.setattr(
        daily_mod, "disk_fraction", lambda p: 0.1
    )  # this test machine's D: is 80 % full
    monkeypatch.setattr(
        daily_mod, "ping", lambda url, kind, msg="": pings.append((kind, msg)) or True
    )
    now = dt.datetime.combine(DAY + dt.timedelta(days=1), dt.time(2, 30), tzinfo=dt.UTC)
    code = daily_mod.run_daily(project, now=now, markets=("spot",), estimate=False)
    assert code == daily_mod.EXIT_OK and [k for k, _ in pings] == ["start", "success"]
    status = read_status(layout.status_file)
    # first run: yesterday and the six days before it (catch-up window), 3 pairs
    assert status["last_day"] == DAY.isoformat() and status["counts"]["pass1_done"] == 3 * 7
    assert status["completed_day"] == DAY.isoformat()
    later = now + dt.timedelta(hours=1)
    assert (
        daily_mod.run_daily(project, now=later, markets=("spot",), estimate=False)
        == daily_mod.EXIT_OK
    )
    assert [k for k, _ in pings] == ["start", "success"]  # the retry is a no-op


def test_daily_waits_then_marks_missing(project: Project, monkeypatch: pytest.MonkeyPatch) -> None:
    layout = DataLayout(project.data_dir)
    pings: list[str] = []
    monkeypatch.setattr("ofs.pipeline.download", _fake_download(layout, {"BBBUSDT": "pending"}))
    monkeypatch.setattr(daily_mod, "disk_fraction", lambda p: 0.1)
    monkeypatch.setattr(daily_mod, "ping", lambda url, kind, msg="": pings.append(kind) or True)
    early = dt.datetime.combine(DAY + dt.timedelta(days=1), dt.time(3, 30), tzinfo=dt.UTC)
    assert (
        daily_mod.run_daily(project, now=early, markets=("spot",), estimate=False)
        == daily_mod.EXIT_RETRY
    )
    late = dt.datetime.combine(DAY + dt.timedelta(days=1), dt.time(12, 30), tzinfo=dt.UTC)
    assert (
        daily_mod.run_daily(project, now=late, markets=("spot",), estimate=True)
        == daily_mod.EXIT_FAIL
    )
    assert pings[-1] == "fail"
    man = read_manifest(layout.manifest_file)
    bbb = man.filter(pl.col("symbol") == "BBBUSDT")
    assert bbb.height == 2 * 7 and set(bbb["status"]) == {"missing"}  # 2 kinds x 7 days
    assert "missing" in read_status(layout.status_file)["last_error"]
    assert read_status(layout.status_file)["live_forecasts"] is True  # ran (no history yet)
    assert daily_mod.finalize_pending(layout, DAY) == 0


def test_daily_reports_failed_live_forecasts(
    project: Project, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = DataLayout(project.data_dir)
    monkeypatch.setattr("ofs.pipeline.download", _fake_download(layout, {}))
    monkeypatch.setattr(daily_mod, "disk_fraction", lambda p: 0.1)
    monkeypatch.setattr(daily_mod, "ping", lambda url, kind, msg="": True)

    def broken(*args: object, **kwargs: object) -> None:
        raise RuntimeError("model failure")

    monkeypatch.setattr("ofs.forecast.live.forecast_live", broken)
    now = dt.datetime.combine(DAY + dt.timedelta(days=1), dt.time(2, 30), tzinfo=dt.UTC)
    code = daily_mod.run_daily(project, now=now, markets=("spot",), estimate=True)
    assert code == daily_mod.EXIT_FAIL
    status = read_status(layout.status_file)
    assert status["live_forecasts"] is False and "live forecasts failed" in status["last_error"]
    assert status["rolling_estimates"] is True  # the data and the estimates of the day are safe


def test_daily_replays_a_file_published_late(
    project: Project, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Spec §11.4: a day missing at 12:00 UTC is replayed the next day."""
    from ofs.pipeline import StageSummary

    layout = DataLayout(project.data_dir)
    gap = DAY - dt.timedelta(days=2)
    absent = {("BBBUSDT", gap)}
    monkeypatch.setattr("ofs.pipeline.download", _fake_download(layout, {}, absent))
    monkeypatch.setattr(daily_mod, "disk_fraction", lambda p: 0.1)
    monkeypatch.setattr(daily_mod, "ping", lambda url, kind, msg="": True)
    monkeypatch.setattr(daily_mod, "apply_corrections", lambda *a, **k: StageSummary())
    now = dt.datetime.combine(DAY + dt.timedelta(days=1), dt.time(2, 30), tzinfo=dt.UTC)
    daily_mod.run_daily(project, now=now, markets=("spot",), estimate=False)
    man = read_manifest(layout.manifest_file)
    assert status_of(man, "BBBUSDT", "spot", "aggTrades", gap) == "missing"
    absent.clear()  # Binance publishes the file late
    daily_mod.run_daily(project, now=now + dt.timedelta(days=1), markets=("spot",), estimate=False)
    man = read_manifest(layout.manifest_file)
    assert status_of(man, "BBBUSDT", "spot", "aggTrades", gap) == "ok"
    from ofs.reduce.pipeline import Reducer

    assert Reducer(project, layout).is_reduced("BBBUSDT", "spot", gap)


def test_daily_applies_republished_files_once_a_week(
    project: Project, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ofs.pipeline import StageSummary

    layout = DataLayout(project.data_dir)
    calls: list[tuple[dt.date, dt.date]] = []

    def recheck(project, markets, start, end):
        calls.append((start, end))
        return StageSummary()

    monkeypatch.setattr("ofs.pipeline.download", _fake_download(layout, {}))
    monkeypatch.setattr(daily_mod, "disk_fraction", lambda p: 0.1)
    monkeypatch.setattr(daily_mod, "ping", lambda url, kind, msg="": True)
    monkeypatch.setattr(daily_mod, "apply_corrections", recheck)
    weekday = project.pipeline.daily.recheck_weekday
    sunday = DAY + dt.timedelta(days=(weekday - DAY.weekday()) % 7)
    for day in (sunday - dt.timedelta(days=1), sunday):
        now = dt.datetime.combine(day + dt.timedelta(days=1), dt.time(2, 30), tzinfo=dt.UTC)
        assert daily_mod.run_daily(project, now=now, markets=("spot",), estimate=False) == 0
    span = dt.timedelta(days=project.pipeline.daily.recheck_days - 1)
    assert calls == [(sunday - span, sunday)]  # only after the recheck weekday


def test_daily_failure_pings_fail(project: Project, monkeypatch: pytest.MonkeyPatch) -> None:
    pings: list[str] = []

    def boom(*a, **k):
        raise RuntimeError("network down")

    monkeypatch.setattr(daily_mod, "run_range", boom)
    monkeypatch.setattr(daily_mod, "ping", lambda url, kind, msg="": pings.append(kind) or True)
    with pytest.raises(RuntimeError):
        daily_mod.run_daily(project, markets=("spot",))
    assert pings == ["start", "fail"]
    assert "network down" in read_status(DataLayout(project.data_dir).status_file)["last_error"]
    assert len(daily_mod.rolling_days(DAY, 24)) == 730


# ---------------------------------------------------------------- monitoring / versioning
def test_ping_and_status(tmp_path: Path) -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200 if "ok" in str(request.url) else 500)

    transport = httpx.MockTransport(handler)
    assert ping("https://hc.test/ok", "start", transport=transport)
    assert ping("https://hc.test/ok", "success", "fine", transport=transport)
    assert not ping("https://hc.test/bad", "fail", transport=transport)
    assert seen == ["https://hc.test/ok/start", "https://hc.test/ok", "https://hc.test/bad/fail"]
    assert not ping(None, "start")

    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    assert not ping("https://hc.test/x", "start", transport=httpx.MockTransport(refuse))
    path = tmp_path / "s.json"
    write_status(path, a=1)
    assert write_status(path, b=2)["a"] == 1 and read_status(path)["b"] == 2
    assert read_status(tmp_path / "none.json") == {}
    assert 0 < disk_fraction(tmp_path / "nonexistent" / "x") <= 1


def test_versions(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    assert stage_version("bins") == stage_version("bins")
    assert stage_version("stats", {"a": 1}) != stage_version("stats", {"a": 2})
    monkeypatch.setenv("OFS_CODE_VERSION", "abc123")
    code_version.cache_clear()
    assert code_version() == "abc123"
    monkeypatch.delenv("OFS_CODE_VERSION")
    code_version.cache_clear()
    assert code_version(tmp_path).endswith("+nogit") or code_version(tmp_path)
    assert not git_tag_exists("x", tmp_path)
    code_version.cache_clear()
