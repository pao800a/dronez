"""Stop hook: run ``pytest -m fast -q`` and block the stop on failure.

* If ``stop_hook_active`` is true, Claude is already continuing because of a
  stop hook, so the stop is allowed (no infinite loops).
* If nothing relevant changed since the last green run (same ``git status``
  fingerprint and file mtimes), the run is skipped.
* ``DRONEZ_STOP_HOOK_PYTEST`` (a JSON list) overrides the pytest command;
  ``DRONEZ_STOP_HOOK_DISABLE=1`` turns the hook off (e.g. during long sessions
  of docs-only work).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _hooklib import emit, logs_dir, project_dir, read_input, run, venv_tool

TIMEOUT_S = 120
CACHE_FILE = "stop_hook_last_green.json"
FAIL_RE = re.compile(r"^(?:FAILED|ERROR) .*", re.MULTILINE)


def fingerprint(root: Path) -> str:
    code, status = run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=root)
    h = hashlib.sha256(status.encode() if code == 0 else os.urandom(8))
    for line in status.splitlines():
        rel = line[3:].strip().split(" -> ")[-1].strip('"')
        p = root / rel
        if p.suffix in {".py", ".yaml", ".yml", ".toml"} and p.exists():
            h.update(f"{rel}:{p.stat().st_mtime_ns}".encode())
    code, head = run(["git", "rev-parse", "HEAD"], cwd=root)
    h.update(head.encode())
    return h.hexdigest()


def pytest_cmd() -> list[str]:
    override = os.environ.get("DRONEZ_STOP_HOOK_PYTEST")
    if override:
        cmd = json.loads(override)
        if isinstance(cmd, list):
            return [str(c) for c in cmd]
    return venv_tool("pytest")


def main() -> None:
    data = read_input()
    if data.get("stop_hook_active") or os.environ.get("DRONEZ_STOP_HOOK_DISABLE") == "1":
        return
    root = Path(os.environ.get("DRONEZ_STOP_HOOK_ROOT") or project_dir())
    if not (root / "tests").is_dir() or not (root / "pyproject.toml").is_file():
        return

    cache = logs_dir() / CACHE_FILE
    fp = fingerprint(root)
    try:
        if json.loads(cache.read_text(encoding="utf-8")).get("fingerprint") == fp:
            return
    except (OSError, ValueError):
        pass

    code, out = run(
        [*pytest_cmd(), "-m", "fast", "-q", "-x", "--no-header", "-p", "no:cacheprovider"],
        cwd=root,
        timeout=TIMEOUT_S,
    )
    if code in (0, 5):  # 5 = no tests collected
        with contextlib.suppress(OSError):
            cache.write_text(json.dumps({"fingerprint": fp}), encoding="utf-8")
        return

    failures = FAIL_RE.findall(out)
    tail = "\n".join(out.strip().splitlines()[-15:])
    summary = "\n".join(failures[:10]) if failures else tail
    if code == 127 or "Timeout" in out:
        summary = f"pytest could not run or timed out ({TIMEOUT_S}s):\n{tail}"
    emit(
        {
            "decision": "block",
            "reason": (
                f"`pytest -m fast` is failing — fix before stopping (exit {code}):\n{summary}"
            ),
        }
    )


if __name__ == "__main__":
    main()
