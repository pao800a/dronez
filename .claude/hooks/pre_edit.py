"""PreToolUse(Edit|Write) guard for protected files.

Blocks edits to ``.env*`` files, ``uv.lock`` (only uv may change it),
``releases/`` and generated assets (``generated/`` in the repo or anything in
``DRONEZ_DATA_DIR``). Those are produced by the pipeline, not by hand.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _hooklib import block, data_dir, is_within, project_dir, read_input


def reason_for(path: Path) -> str | None:
    name = path.name
    if name == ".env" or name.startswith(".env."):
        return f"{name} holds secrets and is never edited by Claude"
    if name == "uv.lock":
        return "uv.lock is managed by uv only; run `uv add`/`uv lock`"
    root = project_dir()
    if is_within(path, root / "releases"):
        return "releases/ is written by `dronez-export`; regenerate instead of editing"
    if is_within(path, root / "generated"):
        return "generated/ is pipeline output; change the code/config and regenerate"
    if is_within(path, data_dir()):
        return "DRONEZ_DATA_DIR holds generated artifacts; regenerate instead of editing"
    return None


def main() -> None:
    data = read_input()
    tool_input = data.get("tool_input") or {}
    raw = tool_input.get("file_path") or tool_input.get("notebook_path")
    if not raw:
        return
    path = Path(str(raw))
    if not path.is_absolute():
        path = Path(data.get("cwd") or project_dir()) / path
    reason = reason_for(path)
    if reason:
        block(f"Blocked by .claude/hooks/pre_edit.py: {reason}.")


if __name__ == "__main__":
    main()
