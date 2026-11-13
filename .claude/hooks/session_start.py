"""SessionStart: inject a short Dronez status block into Claude's context.

Plain-text stdout on exit 0 is added to the context. It must stay fast: no
torch import (versions come from package metadata) and no network.
"""

from __future__ import annotations

import json
import re
import sys
from importlib import metadata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _hooklib import data_dir, git_branch, project_dir, read_input, run

MAX_NEXT_LINES = 12


def _version(pkg: str) -> str:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return "not installed"


def progress_next(root: Path) -> str:
    path = root / "docs" / "PROGRESS.md"
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return "(docs/PROGRESS.md missing)"
    m = re.search(r"^##\s+Next\b.*?$(.*?)(?=^##\s|\Z)", text, re.MULTILINE | re.DOTALL)
    if not m:
        return "(no '## Next' section in docs/PROGRESS.md)"
    lines = [ln for ln in m.group(1).strip().splitlines() if ln.strip()]
    extra = (
        f"\n  ... ({len(lines) - MAX_NEXT_LINES} more lines)" if len(lines) > MAX_NEXT_LINES else ""
    )
    return "\n".join(f"  {ln}" for ln in lines[:MAX_NEXT_LINES]) + extra


def latest_run() -> str:
    runs = data_dir() / "runs"
    try:
        candidates = [p for p in runs.iterdir() if p.is_dir()]
    except OSError:
        return "none"
    if not candidates:
        return "none"
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    status = "unknown"
    status_file = latest / "status.json"
    try:
        info = json.loads(status_file.read_text(encoding="utf-8"))
        status = str(info.get("status", "unknown"))
        if "timesteps" in info:
            status += f", {info['timesteps']:,} steps"
    except (OSError, ValueError, TypeError):
        pass
    return f"{latest.name} ({status})"


def torch_device() -> str:
    version = _version("torch")
    if "+xpu" in version:
        return "cpu (xpu-enabled torch build installed; opt-in via --device xpu)"
    return "cpu"


def main() -> None:
    data = read_input()
    root = project_dir()
    branch = git_branch(root) or "(no git)"
    code, porcelain = run(["git", "status", "--porcelain"], cwd=root, timeout=5)
    dirty = len(porcelain.splitlines()) if code == 0 else 0
    lines = [
        f"## Dronez session status ({data.get('source', 'startup')})",
        f"- git: branch `{branch}`, {'clean' if dirty == 0 else f'{dirty} changed file(s)'}",
        f"- mujoco {_version('mujoco')}, torch {_version('torch')}, "
        f"sb3 {_version('stable-baselines3')}; device: {torch_device()}",
        f"- data dir: {data_dir()}",
        f"- latest run: {latest_run()}",
        "- PROGRESS.md 'Next':",
        progress_next(root),
    ]
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
