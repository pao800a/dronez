"""Pipe sample hook JSON into every .claude/hooks script and check the behaviour.

Each hook runs as a subprocess, the way Claude Code runs it, with
``CLAUDE_PROJECT_DIR`` and ``DRONEZ_DATA_DIR`` pointing at temp dirs.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.fast

HOOKS = Path(__file__).resolve().parents[2] / ".claude" / "hooks"


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / "proj"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    return root


@pytest.fixture
def env(project: Path, tmp_path: Path) -> dict[str, str]:
    e = dict(os.environ)
    e["CLAUDE_PROJECT_DIR"] = str(project)
    e["DRONEZ_DATA_DIR"] = str(tmp_path / "data")
    e["UV_PROJECT_ENVIRONMENT"] = sys.prefix  # lets venv_tool find ruff/pytest
    e["DRONEZ_NOTIFY_DISABLE"] = "1"
    e.pop("DRONEZ_STOP_HOOK_DISABLE", None)
    return e


def run_hook(
    name: str, payload: dict[str, Any], env: dict[str, str], cwd: Path | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(HOOKS / name)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        env=env,
        cwd=cwd,
        timeout=180,
        check=False,
    )


def bash(cmd: str, cwd: Path, background: bool = False) -> dict[str, Any]:
    return {
        "hook_event_name": "PreToolUse",
        "tool_name": "Bash",
        "cwd": str(cwd),
        "tool_input": {"command": cmd, "run_in_background": background},
    }


# --- PreToolUse(Bash) ---------------------------------------------------------


@pytest.mark.parametrize(
    "cmd",
    [
        "pip install numpy",
        "pip3 install -U torch",
        "python -m pip install x",
        "uv pip install x",
        "cd src && pip install -e .",
    ],
)
def test_pip_install_blocked(cmd: str, project: Path, env: dict[str, str]) -> None:
    res = run_hook("pre_bash.py", bash(cmd, project), env)
    assert res.returncode == 2
    assert "uv add" in res.stderr


@pytest.mark.parametrize(
    "cmd", ["uv add numpy", "uv sync", "ls -la", "git status", "rm -rf build/", "rm -f x.txt"]
)
def test_harmless_commands_allowed(cmd: str, project: Path, env: dict[str, str]) -> None:
    res = run_hook("pre_bash.py", bash(cmd, project), env)
    assert res.returncode == 0, res.stderr
    assert res.stdout == ""


def test_rm_rf_protected(project: Path, env: dict[str, str], tmp_path: Path) -> None:
    for cmd in (
        "rm -rf .",
        f"rm -rf {project}",
        "rm -r -f ./",
        "rm -rf /",
        "rm -rf ~",
        "rm -rf $CLAUDE_PROJECT_DIR",
        f"rm -rf {tmp_path / 'data'}",
        "rm -rf $DRONEZ_DATA_DIR",
        f"rm -fr {tmp_path}",  # ancestor of both
        "echo hi && rm -rf ..",
    ):
        res = run_hook("pre_bash.py", bash(cmd, project), env)
        assert res.returncode == 2, cmd
    sub = project / "src"
    sub.mkdir()
    assert run_hook("pre_bash.py", bash("rm -rf ..", sub), env).returncode == 2
    assert run_hook("pre_bash.py", bash("rm -rf src/old", project), env).returncode == 0


def test_force_push_main_blocked(project: Path, env: dict[str, str]) -> None:
    for cmd in (
        "git push --force origin main",
        "git push -f origin main",
        "git push origin +main",
        "git push --force-with-lease origin HEAD:main",
        "git push -f",  # current branch is main
    ):
        assert run_hook("pre_bash.py", bash(cmd, project), env).returncode == 2, cmd
    for cmd in ("git push origin main", "git push -u origin phase-0", "git push"):
        assert run_hook("pre_bash.py", bash(cmd, project), env).returncode == 0, cmd


def test_force_push_feature_branch_allowed(project: Path, env: dict[str, str]) -> None:
    _git(project, "switch", "-q", "-c", "feature")
    assert run_hook("pre_bash.py", bash("git push -f", project), env).returncode == 0
    assert run_hook("pre_bash.py", bash("git push -f origin feat", project), env).returncode == 0


def test_reset_hard_on_main(project: Path, env: dict[str, str]) -> None:
    assert run_hook("pre_bash.py", bash("git reset --hard HEAD~1", project), env).returncode == 2
    _git(project, "switch", "-q", "-c", "feature")
    assert run_hook("pre_bash.py", bash("git reset --hard", project), env).returncode == 0


def test_foreground_training_warns(project: Path, env: dict[str, str]) -> None:
    res = run_hook("pre_bash.py", bash("uv run dronez-train --total-timesteps 5e6", project), env)
    assert res.returncode == 0
    out = json.loads(res.stdout)["hookSpecificOutput"]
    assert out["permissionDecision"] == "ask"
    assert "background" in out["permissionDecisionReason"]
    # Same command in the background: silent.
    res = run_hook(
        "pre_bash.py", bash("uv run dronez-train --total-timesteps 5e6", project, True), env
    )
    assert res.stdout == ""
    # Small smoke run: silent.
    res = run_hook("pre_bash.py", bash("uv run dronez-train --timesteps 20_000", project), env)
    assert res.stdout == ""


def test_foreground_training_reads_config(project: Path, env: dict[str, str]) -> None:
    cfg = project / "big.yaml"
    cfg.write_text("ppo:\n  total_timesteps: 30_000_000\n", encoding="utf-8")
    res = run_hook("pre_bash.py", bash(f"uv run dronez-train --config {cfg.name}", project), env)
    assert json.loads(res.stdout)["hookSpecificOutput"]["permissionDecision"] == "ask"


# --- PreToolUse(Edit|Write) ---------------------------------------------------


def edit(path: Path) -> dict[str, Any]:
    return {
        "hook_event_name": "PreToolUse",
        "tool_name": "Edit",
        "tool_input": {"file_path": str(path)},
    }


def test_protected_files(project: Path, env: dict[str, str], tmp_path: Path) -> None:
    for p in (
        project / ".env",
        project / ".env.local",
        project / "uv.lock",
        project / "releases" / "v0.1.0" / "policy_meta.json",
        project / "generated" / "x.stl",
        tmp_path / "data" / "models" / "m" / "drone.xml",
    ):
        res = run_hook("pre_edit.py", edit(p), env)
        assert res.returncode == 2, p
        assert "Blocked" in res.stderr
    for p in (project / "src" / "dronez" / "x.py", project / "docs" / "env.md"):
        assert run_hook("pre_edit.py", edit(p), env).returncode == 0, p


# --- PostToolUse(Edit|Write) ruff ---------------------------------------------


def test_post_edit_ruff_formats_and_reports(project: Path, env: dict[str, str]) -> None:
    f = project / "mod.py"
    f.write_text("import os\nx = {'a':1}\ny = undefined_name\n", encoding="utf-8")
    payload = {
        "hook_event_name": "PostToolUse",
        "tool_name": "Write",
        "tool_input": {"file_path": str(f)},
    }
    res = run_hook("post_edit_ruff.py", payload, env)
    assert res.returncode == 0
    text = f.read_text(encoding="utf-8")
    assert 'x = {"a": 1}' in text  # formatted
    assert "import os" not in text  # unused import auto-fixed
    out = json.loads(res.stdout)
    assert out["decision"] == "block"
    assert "F821" in out["reason"]


def test_post_edit_ruff_clean_and_non_python(project: Path, env: dict[str, str]) -> None:
    f = project / "ok.py"
    f.write_text("X = 1\n", encoding="utf-8")
    for p in (f, project / "README.md"):
        payload = {"hook_event_name": "PostToolUse", "tool_input": {"file_path": str(p)}}
        res = run_hook("post_edit_ruff.py", payload, env)
        assert res.returncode == 0
        assert res.stdout == ""


# --- Stop: pytest -m fast -------------------------------------------------------

PYPROJECT = textwrap.dedent(
    """
    [tool.pytest.ini_options]
    markers = ["fast: quick"]
    """
)


def _stop_env(env: dict[str, str]) -> dict[str, str]:
    e = dict(env)
    e["DRONEZ_STOP_HOOK_PYTEST"] = json.dumps([sys.executable, "-m", "pytest"])
    return e


def test_stop_blocks_on_failure(project: Path, env: dict[str, str]) -> None:
    (project / "pyproject.toml").write_text(PYPROJECT, encoding="utf-8")
    (project / "tests").mkdir()
    (project / "tests" / "test_x.py").write_text(
        "import pytest\n\n@pytest.mark.fast\ndef test_broken():\n    assert 1 == 2\n",
        encoding="utf-8",
    )
    res = run_hook(
        "stop_pytest.py", {"hook_event_name": "Stop", "stop_hook_active": False}, _stop_env(env)
    )
    assert res.returncode == 0
    out = json.loads(res.stdout)
    assert out["decision"] == "block"
    assert "test_broken" in out["reason"]
    # Already continuing because of a stop hook -> allow, no rerun.
    res = run_hook(
        "stop_pytest.py", {"hook_event_name": "Stop", "stop_hook_active": True}, _stop_env(env)
    )
    assert res.returncode == 0
    assert res.stdout == ""


def test_stop_allows_and_caches_on_pass(project: Path, env: dict[str, str]) -> None:
    (project / "pyproject.toml").write_text(PYPROJECT, encoding="utf-8")
    (project / "tests").mkdir()
    (project / "tests" / "test_ok.py").write_text(
        "import pytest\n\n@pytest.mark.fast\ndef test_ok():\n    assert True\n", encoding="utf-8"
    )
    res = run_hook(
        "stop_pytest.py", {"hook_event_name": "Stop", "stop_hook_active": False}, _stop_env(env)
    )
    assert res.returncode == 0
    assert res.stdout == ""
    assert (project / ".claude" / "logs" / "stop_hook_last_green.json").is_file()


def test_stop_without_tests_is_noop(project: Path, env: dict[str, str]) -> None:
    res = run_hook(
        "stop_pytest.py", {"hook_event_name": "Stop", "stop_hook_active": False}, _stop_env(env)
    )
    assert res.returncode == 0
    assert res.stdout == ""


# --- SubagentStart / SubagentStop ------------------------------------------------


def test_subagent_log(project: Path, env: dict[str, str]) -> None:
    agents = project / ".claude" / "agents"
    agents.mkdir(parents=True)
    (agents / "architect.md").write_text(
        "---\nname: architect\ndescription: d\nmodel: opus\n---\nbody\n", encoding="utf-8"
    )
    start = {"hook_event_name": "SubagentStart", "agent_id": "a1", "agent_type": "architect"}
    assert run_hook("subagent_log.py", start, env).returncode == 0
    stop = {
        "hook_event_name": "SubagentStop",
        "agent_id": "a1",
        "agent_type": "architect",
        "stop_hook_active": False,
        "last_assistant_message": "\nWrote docs/architecture.md\nmore detail",
    }
    res = run_hook("subagent_log.py", stop, env)
    assert res.returncode == 0
    assert res.stdout == ""  # never blocks
    lines = (project / ".claude" / "logs" / "agents.jsonl").read_text(encoding="utf-8").splitlines()
    entry = json.loads(lines[-1])
    assert entry["agent"] == "architect"
    assert entry["model"] == "opus"
    assert entry["duration_s"] is not None
    assert entry["summary_first_line"] == "Wrote docs/architecture.md"
    assert set(entry) >= {"timestamp", "agent", "duration_s", "summary_first_line"}


# --- SessionStart / PreCompact / Notification -------------------------------------


def test_session_start_status(project: Path, env: dict[str, str], tmp_path: Path) -> None:
    (project / "docs").mkdir()
    (project / "docs" / "PROGRESS.md").write_text(
        "# Progress\n\n## Next\n- build the CAD\n- run G1\n\n## Done\n- bootstrap\n",
        encoding="utf-8",
    )
    run_dir = tmp_path / "data" / "runs" / "att-001"
    run_dir.mkdir(parents=True)
    (run_dir / "status.json").write_text(
        json.dumps({"status": "running", "timesteps": 1234567}), encoding="utf-8"
    )
    res = run_hook(
        "session_start.py", {"hook_event_name": "SessionStart", "source": "startup"}, env
    )
    assert res.returncode == 0
    assert "branch `main`" in res.stdout
    assert "build the CAD" in res.stdout
    assert "bootstrap" not in res.stdout  # only the Next section
    assert "att-001 (running, 1,234,567 steps)" in res.stdout
    assert "mujoco" in res.stdout


def test_pre_compact_mentions_progress(project: Path, env: dict[str, str]) -> None:
    res = run_hook("pre_compact.py", {"hook_event_name": "PreCompact", "trigger": "auto"}, env)
    assert res.returncode == 0
    assert "docs/PROGRESS.md" in res.stdout


def test_notification_silent_when_disabled(project: Path, env: dict[str, str]) -> None:
    payload = {"hook_event_name": "Notification", "message": "Claude needs your permission"}
    res = run_hook("notify.py", payload, env)
    assert res.returncode == 0
    assert res.stdout == ""


def test_hooks_tolerate_empty_stdin(env: dict[str, str]) -> None:
    for script in (
        "pre_bash.py",
        "pre_edit.py",
        "post_edit_ruff.py",
        "pre_compact.py",
        "notify.py",
    ):
        res = subprocess.run(
            [sys.executable, str(HOOKS / script)],
            input="",
            capture_output=True,
            text=True,
            env=env,
            timeout=30,
            check=False,
        )
        assert res.returncode == 0, (script, res.stderr)


def test_settings_wire_every_hook() -> None:
    settings = json.loads((HOOKS.parent / "settings.json").read_text(encoding="utf-8"))
    hooks = settings["hooks"]
    for event in (
        "SessionStart",
        "PreToolUse",
        "PostToolUse",
        "Stop",
        "SubagentStop",
        "PreCompact",
        "Notification",
    ):
        assert event in hooks, event
    referenced = {
        Path(h["args"][-1]).name
        for groups in hooks.values()
        for group in groups
        for h in group["hooks"]
    }
    on_disk = {p.name for p in HOOKS.glob("*.py") if not p.name.startswith("_")}
    assert referenced == on_disk
