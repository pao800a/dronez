"""PreToolUse(Bash) guard.

Blocks: ``pip install`` (use ``uv add``), ``rm -rf`` on the repo root or the
data dir (or any ancestor of them), force-pushes to ``main``, and
``git reset --hard`` while on ``main``. Asks for confirmation when a training
command with more than 200k timesteps is launched in the foreground.
"""

from __future__ import annotations

import os
import re
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _hooklib import block, data_dir, emit, git_branch, project_dir, read_input

FOREGROUND_TIMESTEP_LIMIT = 200_000

PIP_RE = re.compile(r"(?:^|[\s;&|(])(?:pip3?|python3?(?:\.\d+)?\s+-m\s+pip|uv\s+pip)\s+install\b")
SEGMENT_SPLIT_RE = re.compile(r"&&|\|\||[;|\n]")
TRAIN_RE = re.compile(r"dronez-train|dronez\.rl\.train|rl/train\.py")
TIMESTEPS_RE = re.compile(r"(?:--total[-_]timesteps|--timesteps)[=\s]+([\d_.eE+]+)")
OVERRIDE_TIMESTEPS_RE = re.compile(r"total_timesteps\s*[=:]\s*([\d_.eE+]+)")
CONFIG_RE = re.compile(r"--config[=\s]+(\S+)")


def _segments(command: str) -> list[list[str]]:
    if sys.platform == "win32":  # keep Windows paths intact under POSIX shlex rules
        command = command.replace("\\", "/")
    out: list[list[str]] = []
    for seg in SEGMENT_SPLIT_RE.split(command):
        try:
            tokens = shlex.split(seg, posix=True)
        except ValueError:
            tokens = seg.split()
        if tokens:
            out.append(tokens)
    return out


def _expand(token: str, cwd: Path) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(token))
    if "$" in expanded:  # unresolved variable, e.g. $CLAUDE_PROJECT_DIR unset
        expanded = expanded.replace("${CLAUDE_PROJECT_DIR}", str(project_dir())).replace(
            "$CLAUDE_PROJECT_DIR", str(project_dir())
        )
    p = Path(expanded)
    if not p.is_absolute():
        p = cwd / p
    return Path(os.path.normpath(p))


def check_rm(tokens: list[str], cwd: Path) -> str | None:
    if not tokens or Path(tokens[0]).name != "rm":
        return None
    flags = "".join(t.lstrip("-") for t in tokens[1:] if t.startswith("-") and t != "--")
    long_flags = {t for t in tokens[1:] if t.startswith("--")}
    recursive = "r" in flags.lower() or "--recursive" in long_flags
    force = "f" in flags or "--force" in long_flags
    if not (recursive and force):
        return None
    protected = [project_dir(), data_dir()]
    for tok in tokens[1:]:
        if tok.startswith("-"):
            continue
        if tok in {"*", "/*", "./*", "~"}:
            return f"refusing `rm -rf {tok}`"
        target = _expand(tok, cwd).resolve()
        for root in protected:
            root_n = root.resolve()
            if target == root_n or root_n.is_relative_to(target):
                return f"refusing `rm -rf` on {target} (contains protected dir {root_n})"
    return None


def check_git(tokens: list[str], cwd: Path) -> str | None:
    if len(tokens) < 2 or Path(tokens[0]).name != "git":
        return None
    # Skip global options like `-C dir`.
    args = tokens[1:]
    while args and args[0].startswith("-"):
        args = args[2:] if args[0] in {"-C", "-c"} else args[1:]
    if not args:
        return None
    sub, rest = args[0], args[1:]
    if sub == "push":
        force = any(
            a in {"-f", "--force", "--force-with-lease", "--force-if-includes"}
            or a.startswith("--force-with-lease=")
            for a in rest
        ) or any(a.startswith("+") for a in rest)
        if not force:
            return None
        positional = [a for a in rest if not a.startswith("-")]
        refspecs = positional[1:]  # first positional is the remote
        targets_main = any(
            r.lstrip("+").split(":")[-1] in {"main", "refs/heads/main"} for r in refspecs
        )
        if targets_main or (not refspecs and git_branch(cwd) == "main"):
            return "force-pushing to `main` is forbidden; open a PR instead"
    if sub == "reset" and "--hard" in rest and git_branch(cwd) == "main":
        return "`git reset --hard` on `main` is forbidden; work on a branch"
    return None


def _num(text: str) -> float | None:
    try:
        return float(text.replace("_", ""))
    except ValueError:
        return None


def foreground_training_steps(command: str, cwd: Path) -> float | None:
    """Timesteps of a training command, or ``None`` if not a training command."""
    if not TRAIN_RE.search(command):
        return None
    for regex in (TIMESTEPS_RE, OVERRIDE_TIMESTEPS_RE):
        m = regex.search(command)
        if m and (n := _num(m.group(1))) is not None:
            return n
    m = CONFIG_RE.search(command)
    if m:
        cfg = _expand(m.group(1), cwd)
        if not cfg.exists():
            cfg = project_dir() / m.group(1)
        try:
            text = cfg.read_text(encoding="utf-8")
        except OSError:
            return None
        m2 = OVERRIDE_TIMESTEPS_RE.search(text)
        if m2:
            return _num(m2.group(1))
    return None


def main() -> None:
    data = read_input()
    tool_input = data.get("tool_input") or {}
    command = str(tool_input.get("command") or "")
    if not command.strip():
        return
    cwd = Path(data.get("cwd") or os.getcwd())

    if PIP_RE.search(command):
        block("`pip install` is not allowed in Dronez. Use `uv add <package>` (or `uv sync`).")

    for segment in _segments(command):
        # Strip a leading `sudo`/`env VAR=..` wrapper.
        tokens = segment
        while tokens and (tokens[0] in {"sudo", "env", "command"} or "=" in tokens[0]):
            tokens = tokens[1:]
        for check in (check_rm, check_git):
            reason = check(tokens, cwd)
            if reason:
                block(f"Blocked by .claude/hooks/pre_bash.py: {reason}.")

    if not tool_input.get("run_in_background"):
        steps = foreground_training_steps(command, cwd)
        if steps is not None and steps > FOREGROUND_TIMESTEP_LIMIT:
            emit(
                {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "ask",
                        "permissionDecisionReason": (
                            f"Training run with {steps:,.0f} timesteps launched in the foreground "
                            f"(limit {FOREGROUND_TIMESTEP_LIMIT:,}). Use run_in_background=true "
                            "and monitor via the train-policy skill."
                        ),
                    }
                }
            )


if __name__ == "__main__":
    main()
