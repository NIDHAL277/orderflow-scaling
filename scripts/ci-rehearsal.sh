#!/bin/sh
# Rehearsal of the CI job "quality" (.github/workflows/ci.yml) in a Linux container, for a
# machine that develops on Windows or macOS. It caught forked pool workers that hang on
# Linux only (docs/incidents/2026-11-15-forked-workers-hang-on-linux.md).
#
# From the repository root, with the image built (docker build -t orderflow-scaling:local .):
#
#   mkdir -p /tmp/ofs-ci && git bundle create /tmp/ofs-ci/repo.bundle --all
#   docker run --rm --user 0 -v /tmp/ofs-ci:/ci:ro -v "$PWD/scripts:/scripts:ro" \
#     orderflow-scaling:local sh /scripts/ci-rehearsal.sh [branch]
#
# The repository comes from the bundle (commits and tags, no slow bind mount of the source
# tree). The image's environment already holds the locked runtime packages, so only the
# development ones are downloaded. The last line is "rehearsal exit=<code>".
set -u
branch="${1:-main}"
export DEBIAN_FRONTEND=noninteractive
# the CI runner has none of the image's runtime settings
unset OFS_ROOT OFS_DATA_DIR OFS_CODE_VERSION OFS_PROTOCOL_TAGS OFS_FROZEN_CONFIG_SHA256
unset NUMBA_CACHE_DIR MPLCONFIGDIR
export HOME=/root UV_COMPILE_BYTECODE=0

apt-get update -q >/dev/null 2>&1 && apt-get install -y -q --no-install-recommends git >/dev/null 2>&1 \
  || { echo "apt-get failed"; exit 90; }
rm -rf /src
git clone -q /ci/repo.bundle /src && cd /src && git checkout -q "$branch" || { echo "clone failed"; exit 91; }
echo "commit $(git rev-parse --short HEAD), tags: $(git tag -l | tr '\n' ' ')"
uname -srm
uv sync --frozen 2>&1 | tail -n 3 || { echo "uv sync failed"; exit 92; }

fail=0
run() {
  echo
  echo "\$ $*"
  "$@"
  rc=$?
  echo "[exit $rc]"
  [ "$rc" -eq 0 ] || fail=1
}
run uv run --frozen ruff check .
run uv run --frozen ruff format --check .
run uv run --frozen mypy
# -v, a time limit and faulthandler: a test that hangs is named in the log
run timeout 9000 uv run --frozen pytest --cov=ofs --cov-report=term -v -p no:cacheprovider \
  -o faulthandler_timeout=1500
run uv run --frozen ofs --data-dir /tmp/ofs-data simulate study --profile ci --max-cells 12
run uv run --frozen ofs --data-dir /tmp/ofs-data simulate check
echo
echo "rehearsal exit=$fail at $(date +%H:%M)"
exit "$fail"
