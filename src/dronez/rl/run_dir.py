"""Run directory ``$DRONEZ_DATA_DIR/runs/<run_id>/`` and its atomic ``status.json``."""

from __future__ import annotations

import json
import os
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dronez.utils import paths

RUN_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
STATUS_SCHEMA_VERSION = 1


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def default_run_id(stem: str) -> str:
    return f"{stem.lower().replace('_', '-')}-{datetime.now(UTC):%Y%m%d-%H%M%S}"


def validate_run_id(run_id: str) -> str:
    if not RUN_ID_RE.match(run_id):
        raise ValueError(f"run_id {run_id!r} must match [a-z0-9-]+")
    return run_id


def git_info() -> tuple[str, bool]:
    """``(short sha, dirty)``; ``("unknown", False)`` outside a git checkout."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
            cwd=paths.repo_root(),
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
                cwd=paths.repo_root(),
            ).stdout.strip()
        )
        return sha, dirty
    except (OSError, subprocess.SubprocessError):
        return "unknown", False


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


class RunDir:
    """Paths of one run plus ``write_status`` (atomic: tmp file + ``os.replace``)."""

    def __init__(self, run_id: str, create: bool = True) -> None:
        self.run_id = validate_run_id(run_id)
        self.root = paths.runs_dir(run_id, create=create)
        self.tb = self.root / "tb"
        self.checkpoints = self.root / "checkpoints"
        self.status_path = self.root / "status.json"
        self.config_path = self.root / "config.yaml"
        self.log_path = self.root / "train.log"
        if create:
            self.tb.mkdir(exist_ok=True)
            self.checkpoints.mkdir(exist_ok=True)
        self._status: dict[str, Any] = {}
        if self.status_path.exists():
            self._status = json.loads(self.status_path.read_text(encoding="utf-8"))

    def read_status(self) -> dict[str, Any]:
        return dict(self._status)

    def write_status(self, **fields: Any) -> dict[str, Any]:
        """Merge ``fields`` into the status and write it atomically."""
        self._status.update(fields)
        self._status.setdefault("schema_version", STATUS_SCHEMA_VERSION)
        self._status["run_id"] = self.run_id
        self._status["pid"] = os.getpid()
        self._status["updated_at"] = utc_now()
        atomic_write_json(self.status_path, self._status)
        return dict(self._status)
