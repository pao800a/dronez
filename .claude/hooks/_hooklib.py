"""Shared helpers for the Dronez Claude Code hooks (stdlib only, cross-platform).

Hooks receive one JSON object on stdin. Exit 0 = allow (stdout may be JSON),
exit 2 = block (stderr is the reason). See the Claude Code hooks reference.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def read_input() -> dict[str, Any]:
    """Parse the hook's stdin JSON; an empty or invalid payload yields ``{}``."""
    try:
        raw = sys.stdin.read()
    except (OSError, ValueError):
        return {}
    if not raw.strip():
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def project_dir() -> Path:
    """Project root: ``$CLAUDE_PROJECT_DIR``, else two levels above this file."""
    env = os.environ.get("CLAUDE_PROJECT_DIR")
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    """Mirror of ``dronez.utils.paths.data_dir`` without importing the package."""
    override = os.environ.get("DRONEZ_DATA_DIR")
    if override:
        return Path(override).expanduser().resolve()
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return (Path(base) / "Dronez" / "data").resolve()
    xdg = os.environ.get("XDG_DATA_HOME")
    return ((Path(xdg) if xdg else Path.home() / ".local" / "share") / "dronez").resolve()


def logs_dir() -> Path:
    path = project_dir() / ".claude" / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def emit(obj: dict[str, Any]) -> None:
    """Print a JSON decision object to stdout (exit code must then be 0)."""
    sys.stdout.write(json.dumps(obj))
    sys.stdout.flush()


def block(reason: str) -> None:
    """Block via exit code 2; ``reason`` goes to Claude through stderr."""
    sys.stderr.write(reason.rstrip() + "\n")
    sys.exit(2)


def run(cmd: list[str], cwd: Path | None = None, timeout: float = 10.0) -> tuple[int, str]:
    """Run a command, returning ``(returncode, combined output)``; never raises."""
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            encoding="utf-8",
            errors="replace",
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return 127, f"{type(exc).__name__}: {exc}"
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def git_branch(cwd: Path) -> str | None:
    """Current branch name (works before the first commit); ``None`` if detached/no repo."""
    code, out = run(["git", "branch", "--show-current"], cwd=cwd, timeout=5)
    return out.strip() if code == 0 and out.strip() else None


def venv_tool(name: str) -> list[str]:
    """Command prefix for a dev tool: the venv binary if present, else ``uv run``.

    Calling the venv binary directly avoids ~100 ms of ``uv run`` overhead,
    which matters for hooks that fire on every edit.
    """
    env_dir = os.environ.get("UV_PROJECT_ENVIRONMENT")
    venv = Path(env_dir) if env_dir else project_dir() / ".venv"
    sub, exe = ("Scripts", f"{name}.exe") if sys.platform == "win32" else ("bin", name)
    candidate = venv / sub / exe
    if candidate.exists():
        return [str(candidate)]
    uv = shutil.which("uv")
    if uv:
        return [uv, "run", "--no-sync", name]
    return [name]


def is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True
