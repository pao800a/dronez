"""SubagentStart/SubagentStop: audit log of subagent runs (for the Opus budget).

On start, the start time is recorded per ``agent_id``. On stop, one JSON line
``{timestamp, agent, agent_id, model, duration_s, summary_first_line}`` is
appended to ``.claude/logs/agents.jsonl`` (gitignored). Never blocks.
"""

from __future__ import annotations

import json
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _hooklib import logs_dir, project_dir, read_input

STARTS_FILE = "agent_starts.json"
LOG_FILE = "agents.jsonl"
MODEL_RE = re.compile(r"^model:\s*(\S+)", re.MULTILINE)


def agent_model(agent_type: str) -> str | None:
    """The ``model`` from ``.claude/agents/<agent>.md`` frontmatter, if any."""
    if not agent_type:
        return None
    path = project_dir() / ".claude" / "agents" / f"{agent_type}.md"
    try:
        head = path.read_text(encoding="utf-8").split("---", 2)
    except OSError:
        return None
    if len(head) < 3:
        return None
    m = MODEL_RE.search(head[1])
    return m.group(1) if m else None


def _load_starts(path: Path) -> dict[str, float]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {str(k): float(v) for k, v in data.items()} if isinstance(data, dict) else {}


def main() -> None:
    data = read_input()
    event = data.get("hook_event_name", "")
    agent_id = str(data.get("agent_id") or "")
    starts_path = logs_dir() / STARTS_FILE
    starts = _load_starts(starts_path)
    now = time.time()

    if event == "SubagentStart":
        if agent_id:
            starts[agent_id] = now
            # Drop entries older than a day so the file stays tiny.
            starts = {k: v for k, v in starts.items() if now - v < 86_400}
            starts_path.write_text(json.dumps(starts), encoding="utf-8")
        return

    if event != "SubagentStop":
        return
    agent = str(data.get("agent_type") or "")
    started = starts.pop(agent_id, None)
    message = str(data.get("last_assistant_message") or "").strip()
    first_line = next((ln.strip() for ln in message.splitlines() if ln.strip()), "")
    entry = {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "agent": agent or "(main/internal)",
        "agent_id": agent_id,
        "model": agent_model(agent),
        "duration_s": round(now - started, 1) if started else None,
        "summary_first_line": first_line[:200],
    }
    with (logs_dir() / LOG_FILE).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")
    starts_path.write_text(json.dumps(starts), encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # audit logging must never break a session
        sys.stderr.write(f"subagent_log: {exc}\n")
        sys.exit(1)
