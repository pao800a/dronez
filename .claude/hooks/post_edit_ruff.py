"""PostToolUse(Edit|Write): ruff format + ruff check --fix on the edited .py file.

Remaining lint issues are reported back to Claude via ``decision: block``
(the edit itself already happened; this only attaches feedback).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _hooklib import emit, is_within, project_dir, read_input, run, venv_tool

MAX_REPORT_LINES = 20


def main() -> None:
    data = read_input()
    tool_input = data.get("tool_input") or {}
    raw = tool_input.get("file_path")
    if not raw or not str(raw).endswith(".py"):
        return
    path = Path(str(raw))
    root = project_dir()
    if not path.exists() or not is_within(path, root):
        return
    ruff = venv_tool("ruff")
    run([*ruff, "format", "--quiet", str(path)], cwd=root, timeout=5)
    code, out = run(
        [*ruff, "check", "--fix", "--quiet", "--output-format", "concise", str(path)],
        cwd=root,
        timeout=5,
    )
    if code in {0, 127}:  # clean, or ruff missing (don't block on tooling gaps)
        return
    lines = [ln for ln in out.splitlines() if ln.strip()]
    shown = "\n".join(lines[:MAX_REPORT_LINES])
    more = (
        f"\n... and {len(lines) - MAX_REPORT_LINES} more" if len(lines) > MAX_REPORT_LINES else ""
    )
    emit(
        {
            "decision": "block",
            "reason": f"ruff found issues it could not auto-fix in {path.name}:\n{shown}{more}",
        }
    )


if __name__ == "__main__":
    main()
