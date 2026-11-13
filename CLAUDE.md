# Dronez

Parametric quadcopter designed in **build123d** → mass properties from the CAD solids → **MuJoCo** model built through `MjSpec` with a realistic motor/prop model → **Gymnasium** envs → **SB3 PPO** policy for stable attitude (level/commanded attitude, disturbance rejection, recovery up to inverted, hover) → sim-to-real hardening (domain randomization, latency, noise, motor dynamics) → **ONNX** export with a hardware-agnostic deployment contract. The primary dev platform is WSL2 Ubuntu; Windows native is supported (ADR-0001). There is no NVIDIA GPU: training runs on the CPU.

**Current state, next steps and open problems: `docs/PROGRESS.md`. Read it first.** The original spec is `docs/BUILD_PROMPT.md`; architecture is in `docs/architecture.md`; decisions are in `docs/adr/`.

## Commands

```bash
uv sync                                   # install (never pip install)
uv run dronez-build-model --drone default_5in   # params → CAD → mass props → meshes → MJCF
uv run dronez-train --config configs/train/<stage>.yaml   # long runs: run_in_background
uv run dronez-eval --run <run_id>         # scenario suite + report
uv run dronez-export --run <run_id>       # ONNX + parity + latency
uv run pytest -m fast                     # < 60 s, also run by the Stop hook
uv run pytest -m physics                  # physics sanity suite
uv run ruff check && uv run ruff format --check
uv run mypy src
```

Heavy artifacts go to `$DRONEZ_DATA_DIR` (default `~/.local/share/dronez`, Windows `%LOCALAPPDATA%\Dronez\data`). See `src/dronez/utils/paths.py`.

## Hard rules

- **`uv add <pkg>`, never `pip install`.** `uv.lock` is changed only by uv.
- **Generated files are never hand-edited** (meshes, MJCF, STEP, renders, ONNX, reports, anything in `releases/`). Change the code or config and regenerate.
- **Never fabricate results.** Every reported number cites the file that produced it.
- Long jobs (training, sweeps) run in the background and log to files. Never tail huge logs into context; use `log-summarizer`.
- Before using a MuJoCo / Gymnasium / SB3 / build123d / onnx / onnxruntime / PyTorch API you're not certain about, **check Context7** (`mcp__context7__*`).
- Keep `docs/PROGRESS.md` current: end of each phase, before each long job, and before compaction.
- Git: one branch per phase, conventional commits, PR → green CI → merge. Never force-push `main`.

## Conventions (full text: `docs/conventions.md`)

- SI units everywhere; degrees only in human-facing reports.
- World frame: MuJoCo default, z-up (ENU-like). Body frame: **FLU**.
- Quaternions are **(w, x, y, z)**. All quaternion math lives in `src/dronez/utils/rotations.py`; nothing else implements it.
- Attitude error is the geodesic angle on SO(3). No Euler angles in rewards or observations.
- Motor numbering and spin are defined once in the drone YAML (PX4 Quad-X order: M0 FR CCW, M1 RL CCW, M2 FL CW, M3 RR CW). The mixer and yaw-torque signs are derived from it, never typed by hand.
- The deployment contract documents the FLU/ENU → FRD/NED conversion explicitly.

## Layout

`src/dronez/{cad,sim,control,envs,rl,eval,deploy,utils}`, `configs/{drone,sim,train,eval}`, `tests/{unit,physics,integration,hooks}`, `tools/mcp_server/` (the `dronez-sim` MCP server), `scripts/`, `releases/` (committed release artifacts only), `docs/`.

## Subagents (`.claude/agents/`)

| Agent | Model | Use for |
|---|---|---|
| `architect` | opus | ADRs, obs/action/reward spec, curriculum/DR plan, PPO starting points, stuck training (after 2 failed fixes). Writes only `docs/` |
| `physics-auditor` | opus | Read-only review of frames, signs, units, mixer, inertia, motor lag, timestep. Never reviews its own code |
| `cad-engineer` | sonnet | `src/dronez/cad/` |
| `sim-engineer` | sonnet | `src/dronez/sim/` (MjSpec only) |
| `rl-engineer` | sonnet | `src/dronez/envs/`, `src/dronez/rl/` |
| `controls-engineer` | sonnet | `src/dronez/control/` (PID, geometric SO(3)) |
| `training-analyst` | sonnet | Diagnose runs from TensorBoard and eval reports |
| `deploy-engineer` | sonnet | `src/dronez/deploy/` (ONNX, parity, latency, contract) |
| `test-engineer` | sonnet | `tests/`, tooling, pre-commit, CI |
| `log-summarizer` | haiku | Condense logs and CI/pytest output (≤ 150 words) |
| `docs-writer` | haiku | README, docstrings, docs prose, CHANGELOG |
| `git-ops` | haiku | Commit and PR text, labels, issues, CI status |

**Opus budget (Claude Pro):** at most ~3 Opus subagent calls per phase, only for architecture, reward/obs/action design, physics review, or diagnosis after **two** failed fixes. Log every Opus call (date, agent, reason) in `docs/PROGRESS.md`. Do not set `CLAUDE_CODE_SUBAGENT_MODEL`. Subagents return ≤ 250 words plus file paths.

## Skills (`.claude/skills/`)

`drone-model-pipeline`, `physics-sanity`, `train-policy`, `training-diagnostics`, `evaluate-policy`, `export-policy`, `frames-and-units`.

## Hooks and MCP

Hooks are Python scripts in `.claude/hooks/`, wired in `.claude/settings.json` and tested in `tests/hooks/`. They inject session status, block `pip install`, destructive git and rm commands, and edits to protected files, run ruff on edited files and the fast tests on stop, and log subagent runs to `.claude/logs/agents.jsonl`. MCP servers (`.mcp.json`): `github` (needs `GITHUB_PAT`; `gh` is the fallback), `context7`, and from Phase 4 the `dronez-sim` server.
