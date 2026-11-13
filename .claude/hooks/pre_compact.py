"""PreCompact: make the compaction summary carry the resume protocol.

On exit 0, stdout is used as extra instructions for the compaction summary.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _hooklib import git_branch, project_dir, read_input


def main() -> None:
    read_input()
    branch = git_branch(project_dir()) or "(no git)"
    sys.stdout.write(
        "Dronez resume protocol — include these points verbatim in the summary:\n"
        "1. FIRST ACTION after compaction: update docs/PROGRESS.md (phase, done, next, "
        "open problems, latest run IDs + metrics with file paths) if anything changed "
        "since its last update.\n"
        f"2. Current git branch: {branch}. List any uncommitted work and background jobs "
        "(task IDs, run IDs, log paths) still running.\n"
        "3. List the Opus subagent calls made in the current phase (budget: ~3/phase).\n"
        "4. Keep exact numbers only if they came from files; cite the paths.\n"
    )


if __name__ == "__main__":
    main()
