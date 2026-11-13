"""Filesystem locations.

Heavy artifacts (generated models, runs, checkpoints, videos, eval reports)
live under ``DRONEZ_DATA_DIR``, never in the repo, so they are not synced by
OneDrive on Windows and never committed. Defaults:

* Windows: ``%LOCALAPPDATA%\\Dronez\\data``
* Linux/macOS: ``$XDG_DATA_HOME/dronez`` (``~/.local/share/dronez``)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ENV_VAR = "DRONEZ_DATA_DIR"


def repo_root() -> Path:
    """Repository root (the directory holding ``pyproject.toml``)."""
    return Path(__file__).resolve().parents[3]


def configs_dir() -> Path:
    return repo_root() / "configs"


def releases_dir() -> Path:
    return repo_root() / "releases"


def default_data_dir() -> Path:
    """Platform default for the data directory, ignoring ``DRONEZ_DATA_DIR``."""
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "Dronez" / "data"
    xdg = os.environ.get("XDG_DATA_HOME")
    base_path = Path(xdg) if xdg else Path.home() / ".local" / "share"
    return base_path / "dronez"


def data_dir(create: bool = True) -> Path:
    """The data directory, honouring ``DRONEZ_DATA_DIR`` when set."""
    override = os.environ.get(ENV_VAR)
    path = Path(override).expanduser() if override else default_data_dir()
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def _sub(name: str, *parts: str, create: bool = True) -> Path:
    path = data_dir(create=create).joinpath(name, *parts)
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def models_dir(*parts: str, create: bool = True) -> Path:
    """Generated CAD/MJCF assets, usually ``models/<config_name>-<hash>/``."""
    return _sub("models", *parts, create=create)


def runs_dir(*parts: str, create: bool = True) -> Path:
    """Training runs: TensorBoard events, checkpoints, logs."""
    return _sub("runs", *parts, create=create)


def eval_dir(*parts: str, create: bool = True) -> Path:
    """Evaluation reports, plots and videos."""
    return _sub("eval", *parts, create=create)
