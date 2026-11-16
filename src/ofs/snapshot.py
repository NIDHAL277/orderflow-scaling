"""Results snapshot: the derived tables the dashboard and the API read, in one zip file.

The pipeline needs two years of daily statistics and hours of computation; the results
themselves fit in a few hundred megabytes. :func:`export_snapshot` packs them with a
manifest (``snapshot.json``: every file with its size and SHA-256, the export date, the
code version, the last verified day and the licence), so that anyone can browse the results
without running the pipeline. :func:`fetch_snapshot` downloads such a file (or reads a local
one), checks every member against the manifest before writing anything, and unpacks it.

Raw Binance files are never included (Binance Vision terms); the derived tables are
published under CC BY-NC-SA 4.0, like the rest of the results.

    ofs snapshot export                     # -> results-snapshot.zip
    ofs snapshot fetch                      # latest GitHub release -> snapshot/
    OFS_DATA_DIR=snapshot uv run streamlit run dashboards/app.py
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import IO, Any

import httpx
import polars as pl

from ofs.paths import DataLayout
from ofs.versioning import code_version

MANIFEST_NAME = DataLayout(Path()).snapshot_file.name  # "snapshot.json", at the archive root

DEFAULT_URL = (
    "https://github.com/NIDHAL277/orderflow-scaling/releases/latest/download/results-snapshot.zip"
)
LICENCE = (
    "Derived from Binance Vision data (data.binance.vision), CC BY-NC-SA 4.0. Non-commercial "
    "research, not affiliated with Binance, not investment advice."
)
# What the dashboard and the API read, relative to the data directory. Left out: raw files,
# daily statistics and windows (the pipeline's working data), the bootstrap replicates,
# the validation reports (about 86,000 files), logs, locks and temporary files.
INCLUDE_DIRS = ("tests", "forecast", "simulation", "quality", "monitoring", "venues")
INCLUDE_GLOBS = ("estimates/estimates_*.parquet",)
INCLUDE_FILES = (
    "manifest/manifest.parquet",
    "status/pipeline_status.json",
    "cache/validation.parquet",
)
SUFFIXES = {".parquet", ".json"}
CHUNK = 1 << 20


def snapshot_members(data_dir: Path) -> list[PurePosixPath]:
    """Files of the snapshot, relative to ``data_dir``, in a stable order."""
    found: set[Path] = set()
    for name in INCLUDE_DIRS:
        root = data_dir / name
        if root.is_dir():
            found.update(p for p in root.rglob("*") if p.is_file() and p.suffix in SUFFIXES)
    for pattern in INCLUDE_GLOBS:
        found.update(p for p in data_dir.glob(pattern) if p.is_file())
    found.update(data_dir / rel for rel in INCLUDE_FILES if (data_dir / rel).is_file())
    return sorted(PurePosixPath(p.relative_to(data_dir).as_posix()) for p in found)


def _sha256(stream: IO[bytes]) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(CHUNK), b""):
        digest.update(chunk)
    return digest.hexdigest()


def _last_ok_day(data_dir: Path) -> str | None:
    """Most recent day downloaded and verified, from the manifest."""
    path = DataLayout(data_dir).manifest_file
    if not path.is_file():
        return None
    man = pl.read_parquet(path, columns=["date", "status"]).filter(pl.col("status") == "ok")
    last = man["date"].max() if man.height else None
    return last.isoformat() if isinstance(last, dt.date) else None


def export_snapshot(data_dir: Path, out: Path, *, now: dt.datetime | None = None) -> dict[str, Any]:
    """Write the result tables of ``data_dir`` to the zip file ``out``; return its manifest."""
    members = snapshot_members(data_dir)
    if not any(m.parts[0] == "tests" for m in members):
        raise ValueError(f"no test results under {data_dir}: nothing to publish")
    files = []
    for member in members:
        path = data_dir.joinpath(*member.parts)
        with path.open("rb") as stream:
            files.append(
                {"path": str(member), "bytes": path.stat().st_size, "sha256": _sha256(stream)}
            )
    manifest: dict[str, Any] = {
        "exported_at": (now or dt.datetime.now(dt.UTC)).isoformat(timespec="seconds"),
        "code_version": code_version(),
        "last_ok_day": _last_ok_day(data_dir),
        "licence": LICENCE,
        "files": files,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_name(out.name + ".part")
    with zipfile.ZipFile(part, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(MANIFEST_NAME, json.dumps(manifest, indent=1))
        for member in members:
            zf.write(data_dir.joinpath(*member.parts), str(member))
    part.replace(out)
    return manifest


def _check_name(name: str) -> PurePosixPath:
    """Refuse members that could land outside the target directory."""
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or ":" in name or "\\" in name or not path.parts:
        raise ValueError(f"unsafe path in the archive: {name!r}")
    return path


def _download(url: str, dest: Path) -> None:
    """Stream ``url`` to ``dest`` (redirects followed, the system's certificate store)."""
    from ofs.ingest.download import ssl_context

    headers = {"User-Agent": "orderflow-scaling/0.1"}
    with (
        httpx.Client(
            verify=ssl_context(), follow_redirects=True, timeout=120.0, headers=headers
        ) as client,
        client.stream("GET", url) as response,
    ):
        response.raise_for_status()
        with dest.open("wb") as out:
            for chunk in response.iter_bytes(CHUNK):
                out.write(chunk)


def fetch_snapshot(source: str, into: Path, *, force: bool = False) -> dict[str, Any]:
    """Unpack the snapshot ``source`` (URL or local zip file) into ``into``; return its manifest.

    Every member must be listed in the manifest with a matching SHA-256 and must stay inside
    ``into`` (no absolute path, no ``..``); a damaged or foreign archive is refused before
    anything is written. A pipeline data directory (a manifest of downloads but no
    ``snapshot.json``) is not overwritten unless ``force``.
    """
    layout = DataLayout(into)
    if layout.manifest_file.exists() and not layout.snapshot_file.exists() and not force:
        raise ValueError(
            f"{into} holds the pipeline's own data: unpack the snapshot elsewhere (or force)"
        )
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(source)
        if not archive.is_file():
            archive = Path(tmp) / "results-snapshot.zip"
            _download(source, archive)
        try:
            zf = zipfile.ZipFile(archive)
        except zipfile.BadZipFile as exc:
            raise ValueError(f"{source} is not a zip file") from exc
        with zf:
            try:
                manifest = json.loads(zf.read(MANIFEST_NAME))
            except KeyError as exc:
                raise ValueError(f"not a results snapshot: no {MANIFEST_NAME}") from exc
            if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), list):
                raise ValueError("not a results snapshot: its manifest lists no files")
            expected = {f["path"]: f["sha256"] for f in manifest["files"]}
            names = [n for n in zf.namelist() if n != MANIFEST_NAME and not n.endswith("/")]
            for name in names:
                _check_name(name)
                if name not in expected:
                    raise ValueError(f"{name} is not listed in the snapshot manifest")
            if missing := sorted(set(expected) - set(names)):
                raise ValueError(f"missing from the archive: {missing[:5]}")
            for name in names:
                with zf.open(name) as stream:
                    if _sha256(stream) != expected[name]:
                        raise ValueError(f"{name}: checksum mismatch, the archive is damaged")
            for name in names:
                target = into.joinpath(*_check_name(name).parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(name) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst, CHUNK)
    into.mkdir(parents=True, exist_ok=True)
    layout.snapshot_file.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest
