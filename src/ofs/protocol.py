"""Pre-registration guards (spec §9).

* Step 0: no estimator runs on real data before the git tag ``prereg-v1``
  (``config/preregistration.md`` committed first). Before that tag, real data
  only serve to test ingestion and quality.
* Step 2: no estimation on the confirmation sample before the tag
  ``config-frozen``, and ``config/estimation.yaml`` must be unchanged since then.

In a clone, git answers both questions. In the production image there is no
``.git``: CI records the tags present (``OFS_PROTOCOL_TAGS``) and the SHA-256 of
``config/estimation.yaml`` at ``config-frozen`` (``OFS_FROZEN_CONFIG_SHA256``)
at build time (docs/adr/0006).

``OFS_PROTOCOL_GUARD=off`` disables the guard for synthetic data and tests; any
real-data use of it must be reported in "Deviations from protocol".
"""

from __future__ import annotations

import hashlib
import os

from ofs.config import Project
from ofs.logs import get_logger
from ofs.versioning import config_unchanged_since, git_tag_exists

PREREG_TAG = "prereg-v1"
FROZEN_TAG = "config-frozen"


class ProtocolError(RuntimeError):
    """Raised when a step would break the pre-registered protocol."""


def guard_enabled() -> bool:
    """Tell whether the protocol guard is on (``OFS_PROTOCOL_GUARD``, off in tests)."""
    return os.environ.get("OFS_PROTOCOL_GUARD", "on").lower() not in {"off", "0", "false"}


def tag_present(tag: str, project: Project) -> bool:
    """Tell whether the git tag exists or is baked into the image (``OFS_PROTOCOL_TAGS``)."""
    baked = {t.strip() for t in os.environ.get("OFS_PROTOCOL_TAGS", "").split(",") if t.strip()}
    return tag in baked or git_tag_exists(tag, project.root)


def frozen_config_intact(project: Project) -> bool:
    """``config/estimation.yaml`` is the version frozen by ``config-frozen``."""
    expected = os.environ.get("OFS_FROZEN_CONFIG_SHA256", "").strip().lower()
    if expected:
        current = hashlib.sha256((project.config_dir / "estimation.yaml").read_bytes()).hexdigest()
        return current == expected
    return config_unchanged_since(FROZEN_TAG, "config/estimation.yaml", project.root)


def check_estimation_allowed(project: Project, sample: str) -> None:
    """Raise ``ProtocolError`` when ``sample`` may not be estimated yet."""
    if not guard_enabled():
        get_logger().warning("protocol_guard_disabled", sample=sample)
        return
    if not tag_present(PREREG_TAG, project):
        raise ProtocolError(
            f"git tag '{PREREG_TAG}' not found: commit config/preregistration.md and tag it "
            "before running any estimator on real data (spec §9, step 0)."
        )
    if sample in ("confirmation", "history"):  # the history sample includes the confirmation year
        if not tag_present(FROZEN_TAG, project):
            raise ProtocolError(
                f"git tag '{FROZEN_TAG}' not found: freeze the configuration before the first "
                "estimation on the confirmation sample (spec §9, step 2)."
            )
        if not frozen_config_intact(project):
            raise ProtocolError(
                f"config/estimation.yaml differs from tag '{FROZEN_TAG}': confirmatory runs must use "
                "the frozen configuration (record any change under 'Deviations from protocol')."
            )
